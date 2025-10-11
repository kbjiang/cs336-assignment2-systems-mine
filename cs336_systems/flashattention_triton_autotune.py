import torch
import math
import triton
import triton.language as tl
from einops import einsum, rearrange

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=4, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_stages=4, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_stages=4, num_warps=8),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,  # This will be provided by autotune
    K_TILE_SIZE: tl.constexpr,  # This will be provided by autotune
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # the shape is still seq_len*d_model
    # different seq in a batch is located by `stride_qb`
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # on chip computation buffers with dtype `tl.float32`;
    # should not confuse with input dtype
    O_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_block = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')

    # Create offset indices for queries for causal masking
    q_end = query_tile_index * Q_TILE_SIZE + Q_TILE_SIZE - 1
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # if all falls inside the masked out area, no calculation needed
        k_start = i * K_TILE_SIZE
        if is_causal and (k_start > q_end):
            # Skip computation for this tile
            pass
        else:
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option='zero')
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')

            # `allow_tf32=False` is important
            # https://github.com/triton-lang/triton/issues/1840
            S = scale * tl.dot(Q_block, tl.trans(K_block), allow_tf32=False)

            if is_causal:
                # nice trick to get a triangular mask
                k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                causal_mask = k_indices[None, :] <= q_indices[:, None]
                S = tl.where(causal_mask, S, -1e6)
                # tl.device_print("S", S)

            m_curr = tl.maximum(m, tl.max(S, axis=-1))

            P = tl.exp(S - m_curr.expand_dims(axis=-1))

            alpha = tl.exp(m - m_curr)
            L_block = alpha * L_block  + tl.sum(P, axis=-1)

            # according to Claude, tl does not have `diag` so need to use broadcasting
            O_block = alpha[:, None] * O_block
            # using `acc` for `tl.float32`
            O_block = tl.dot(P.to(V_block.dtype), V_block, acc=O_block, allow_tf32=False)
            m = m_curr

        # Move the pointer to next tile (always executed)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block = (1 / L_block)[:, None] * O_block
    L_block = m + tl.log(L_block)

    # when store, ensure the final output matches the input precision
    tl.store(O_block_ptr, O_block.to(Q_block.dtype), boundary_check=(0,))
    tl.store(L_block_ptr, L_block.to(Q_block.dtype), boundary_check=(0,))

class FlashAttentionTritonAutotune(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # cache Q, K and V?
        batch_size, n_queries, D = Q.shape
        _, n_keys, _ = K.shape

        # reshape input tensor to 2D, i.e., remove batch dim
        Q_2d = rearrange(Q, "... d -> (...) d")
        K_2d = rearrange(K, "... d -> (...) d")
        V_2d = rearrange(V, "... d -> (...) d")

        for t in [Q_2d, K_2d, V_2d]:
            assert t.is_cuda, "Expected CUDA tensors"
            assert t.is_contiguous(), "Our pointer arithmetic will assume contiguous inputs"

        # Host-side tensor uses input dtype
        O = torch.empty(Q_2d.shape, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(batch_size * n_queries, device=Q.device, dtype=Q.dtype)

        stride_qb = n_queries * D
        stride_qq = D
        stride_qd = 1
        stride_kb = n_keys * D
        stride_kk = D
        stride_kd = 1
        stride_vb = n_keys * D
        stride_vk = D
        stride_vd = 1
        stride_ob = stride_qb
        stride_oq = stride_qq
        stride_od = 1
        stride_lb = n_queries
        stride_lq = 1
        scale = 1 / (D ** 0.5)
         
        # Autotune will determine the optimal tile sizes
        # Note: We use a reasonable default for grid calculation, but autotune will optimize internally
        def grid(meta):
            return (math.ceil(n_queries / meta['Q_TILE_SIZE']), batch_size)
        
        flash_fwd_kernel[grid](
            Q_2d, K_2d, V_2d, O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            n_queries, n_keys,
            scale, D,
            is_causal,
        )

        Q = rearrange(Q_2d, "(b q) d -> b q d", b=batch_size)
        K = rearrange(K_2d, "(b k) d -> b k d", b=batch_size)
        V = rearrange(V_2d, "(b k) d -> b k d", b=batch_size)
        O = rearrange(O, "(b q) d -> b q d", b=batch_size)
        L = rearrange(L, "(b q) -> b q", b=batch_size)

        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        _, n_queries, D = Q.shape
        _, n_keys, _ = K.shape
        DD = torch.sum(O * dO, axis=-1)
        S = einsum(Q, K, "... q d, ... k d -> ... q k") / D ** 0.5
        if ctx.is_causal:
            causal_mask = torch.arange(n_queries, device=S.device)[:, None] >= torch.arange(n_keys, device=S.device)[None, :]
            S = torch.where(causal_mask.to(S.device), S, -1e6)
        P = torch.exp(S - L[:, :, None])
        dV = einsum(P, dO, "... q k, ... q d -> ... k d")
        dP = einsum(dO, V, "... q d, ... k d -> ... q k")
        dS = P * (dP - DD[:, :, None])
        dQ = einsum(dS, K, "... q k, ... k d -> ... q d") / D ** 0.5
        dK = einsum(dS, Q, "... q k, ... q d -> ... k d") / D ** 0.5
        return dQ, dK, dV, None, None, None


# for benchmarking
def get_autotuned_config(n_heads, d_head, sequence_length, dtype=torch.bfloat16, device='cuda'):
    """Trigger autotuning and get the best configuration for given parameters."""
    print(f"🔍 Finding best autotuned config for (n_heads={n_heads}, d_head={d_head}, seq_len={sequence_length})...")
    
    # Create test tensors to trigger autotuning
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=device, dtype=dtype, requires_grad=True
    )
    
    # Trigger autotuning by running the kernel once
    flash = FlashAttentionTritonAutotune.apply
    # flash = torch.compile(FlashAttentionTritonAutotune.apply)
    _ = flash(q, k, v, True)
    
    # Access the best config from the autotuned kernel
    best_config = getattr(flash_fwd_kernel, 'best_config', None)
    if best_config:
        print(f"📊 Best autotuned config found:")
        print(f"   Q_TILE_SIZE: {best_config.kwargs['Q_TILE_SIZE']}")
        print(f"   K_TILE_SIZE: {best_config.kwargs['K_TILE_SIZE']}")
        print(f"   num_stages: {best_config.num_stages}")
        print(f"   num_warps: {best_config.num_warps}")
    else:
        print(f"⚠️  Autotuning config not accessible. Check if flash is wrapped in `torch.compile`.")

def test_timing_flash_forward_backward(
    test, n_heads, d_head, sequence_length, dtype=torch.bfloat16, device='cuda', track_memory=False
):
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=device, dtype=dtype, requires_grad=True
    )
    
    # For autotuned version, use the function directly since tile sizes are auto-determined
    flash = torch.compile(FlashAttentionTritonAutotune.apply)
    # sanity check; it would fail without compiling if precision in triton is not implemented right
    # flash = FlashAttentionTritonAutotune.apply
    
    def flash_forward():
        o = flash(q, k, v, True)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    # Clear cache and reset peak memory stats before benchmarking
    if track_memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Get initial memory state
        initial_memory = torch.cuda.memory_allocated(device)
        
    if test == "forward":
        results = triton.testing.do_bench(flash_forward, rep=1000, warmup=1000)
    elif test == "forward_backward":
        results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=1000)
    else:
        raise ValueError("Wrong selection.")
        
    if track_memory:
        # Get peak memory usage during benchmarking
        peak_memory = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory / 1024 / 1024
        initial_memory_mb = initial_memory / 1024 / 1024
        
        print(f"⏱️  Timing results: {results} ms")
        print(f"💾 Initial memory: {initial_memory_mb:.2f} MB")
        print(f"💾 Peak memory: {peak_memory_mb:.2f} MB")
        print(f"💾 Peak memory increase: {peak_memory_mb - initial_memory_mb:.2f} MB")
        
        return results, peak_memory_mb
    else:
        print(f"⏱️  Timing results: {results} ms")
        return results

if __name__ == "__main__":
    print("\n1️⃣  Small sequence configuration:")
    get_autotuned_config(16, 128, 1024, dtype=torch.bfloat16)
    
    # Run benchmark with autotuned version - config is determined automatically on first call
    print("\n--- Benchmarking Small Sequence (1024) ---")
    test_timing_flash_forward_backward(
        "forward", 16, 128, 1024, 
        dtype=torch.bfloat16, track_memory=True
    )
    
    print("\n2️⃣  Large sequence configuration:")  
    get_autotuned_config(16, 128, 4096, dtype=torch.bfloat16)
    
    # Compare with larger sequence length - may use different optimal config
    print("\n--- Benchmarking Large Sequence (4096) ---")
    test_timing_flash_forward_backward(
        "forward", 16, 128, 4096, 
        dtype=torch.bfloat16, track_memory=True
    )
    
    # Test forward + backward to see memory impact
    print("\n--- Forward + Backward Pass (4096) ---")
    test_timing_flash_forward_backward(
        "forward_backward", 16, 128, 4096, 
        dtype=torch.bfloat16, track_memory=True
    )
    