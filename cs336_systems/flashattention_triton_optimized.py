import torch
import math
import triton
import triton.language as tl
from einops import rearrange

AUTOTUNE_CONFIG=[
    triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=4, num_warps=4),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=4),
    triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_stages=4, num_warps=8),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}, num_stages=4, num_warps=8),
    triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_stages=4, num_warps=8),
]

@triton.autotune(
    configs=AUTOTUNE_CONFIG,
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

@triton.autotune(
    configs=AUTOTUNE_CONFIG,
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, DD_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_ddb, stride_ddq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,  # when using autotune, the tile sizes need to be at the end.
    K_TILE_SIZE: tl.constexpr,
):
    """Pass 1: Compute dQ - each block handles one Q tile across all K tiles"""
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Setup Q, O, dO, L, D block pointers for this tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
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
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    DD_block_ptr = tl.make_block_ptr(
        DD_ptr + batch_index * stride_ddb,
        shape=(N_QUERIES,),
        strides=(stride_ddq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load Q, O, dO, L, D for this tile
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
    O_block = tl.load(O_block_ptr, boundary_check=(0,), padding_option='zero')
    dO_block = tl.load(dO_block_ptr, boundary_check=(0,), padding_option='zero')
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
    DD_block = tl.load(DD_block_ptr, boundary_check=(0,), padding_option='zero')
    
    # Initialize dQ accumulator
    dQ_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    # Setup K, V block pointers - will iterate through these
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
    
    # Causal masking setup
    q_start = query_tile_index * Q_TILE_SIZE
    q_end = q_start + Q_TILE_SIZE - 1
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
    
    # Iterate through all K/V tiles
    for k_tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_start = k_tile_idx * K_TILE_SIZE
        
        # Skip if causal mask blocks entire tile
        if is_causal and (k_start > q_end):
            pass
        else:
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option='zero')
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
            
            # Recompute S for this Q-K tile pair
            S = scale * tl.dot(Q_block, tl.trans(K_block), allow_tf32=False)
            
            # Apply causal mask if needed
            if is_causal:
                k_indices = k_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = k_indices[None, :] <= q_indices[:, None]
                S = tl.where(causal_mask, S, float('-inf'))
            
            # Recompute P using saved L
            P = tl.exp(S - L_block[:, None])
            
            # Compute dP = dO @ V^T
            dP = tl.dot(dO_block, tl.trans(V_block), allow_tf32=False)
            
            # Compute dS = P * (dP - D)
            dS = P * (dP - DD_block[:, None])
            
            # Accumulate dQ += dS @ K * scale
            dQ_block += scale * tl.dot(dS.to(K_block.dtype), K_block, allow_tf32=False)
        
        # Advance K and V pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Store dQ result
    tl.store(dQ_block_ptr, dQ_block.to(Q_block.dtype), boundary_check=(0,))


@triton.autotune(
    configs=AUTOTUNE_CONFIG,
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, DD_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_ddb, stride_ddq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """Pass 2: Compute dK and dV - each block handles one K/V tile across all Q tiles"""
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Setup K, V, dK, dV block pointers for this tile
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Load K and V for this tile
    K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option='zero')
    V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
    
    # Initialize dK and dV accumulators
    dK_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    # Setup Q, O, dO, L, D block pointers - will iterate through these
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    DD_block_ptr = tl.make_block_ptr(
        DD_ptr + batch_index * stride_ddb,
        shape=(N_QUERIES,),
        strides=(stride_ddq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Causal masking setup for K indices
    k_start = key_tile_index * K_TILE_SIZE
    k_indices = k_start + tl.arange(0, K_TILE_SIZE)
    
    # Iterate through all Q tiles
    for q_tile_idx in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_start = q_tile_idx * Q_TILE_SIZE
        q_end = q_start + Q_TILE_SIZE - 1
        
        # Skip if causal mask blocks entire tile
        if is_causal and (k_start > q_end):
            pass
        else:
            # Load Q, O, dO, L, D for this tile
            Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
            O_block = tl.load(O_block_ptr, boundary_check=(0,), padding_option='zero')
            dO_block = tl.load(dO_block_ptr, boundary_check=(0,), padding_option='zero')
            L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
            DD_block = tl.load(DD_block_ptr, boundary_check=(0,), padding_option='zero')
            
            # Recompute S for this Q-K tile pair
            S = scale * tl.dot(Q_block, tl.trans(K_block), allow_tf32=False)
            
            # Apply causal mask if needed
            if is_causal:
                q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
                causal_mask = k_indices[None, :] <= q_indices[:, None]
                S = tl.where(causal_mask, S, float('-inf'))
            
            # Recompute P using saved L
            P = tl.exp(S - L_block[:, None])
            
            # Accumulate dV += P^T @ dO
            dV_block += tl.dot(tl.trans(P.to(dO_block.dtype)), dO_block, allow_tf32=False)
            
            # Compute dP = dO @ V^T
            dP = tl.dot(dO_block, tl.trans(V_block), allow_tf32=False)
            
            # Compute dS = P * (dP - D)
            dS = P * (dP - DD_block[:, None])
            
            # Accumulate dK += dS^T @ Q * scale
            dK_block += scale * tl.dot(tl.trans(dS.to(Q_block.dtype)), Q_block, allow_tf32=False)
        
        # Advance Q, O, dO, L, D pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        DD_block_ptr = tl.advance(DD_block_ptr, (Q_TILE_SIZE,))
    
    # Store dK and dV results
    tl.store(dK_block_ptr, dK_block.to(K_block.dtype), boundary_check=(0,))
    tl.store(dV_block_ptr, dV_block.to(V_block.dtype), boundary_check=(0,))


class FlashAttentionTritonOptimized(torch.autograd.Function):
    """FlashAttention with Triton backward pass (2-pass algorithm)"""
    
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
        batch_size, n_queries, D = Q.shape
        _, n_keys, _ = K.shape
        
        # Compute D = rowsum(dO * O)
        DD = torch.sum(O * dO, dim=-1)
        
        # Reshape for 2D processing
        Q_2d = rearrange(Q, "b q d -> (b q) d")
        K_2d = rearrange(K, "b k d -> (b k) d")
        V_2d = rearrange(V, "b k d -> (b k) d")
        O_2d = rearrange(O, "b q d -> (b q) d")
        dO_2d = rearrange(dO, "b q d -> (b q) d")
        L_2d = rearrange(L, "b q -> (b q)")
        DD_2d = rearrange(DD, "b q -> (b q)")
        
        # Initialize gradients
        dQ = torch.zeros_like(Q_2d)
        dK = torch.zeros_like(K_2d)
        dV = torch.zeros_like(V_2d)
        
        # Strides
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
        stride_ddb = n_queries
        stride_ddq = 1
        
        scale = 1 / (D ** 0.5)
        
        # Pass 1: Compute dQ
        # grid_dq = (math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size)
        def grid_dq(meta):
            return (math.ceil(n_queries / meta['Q_TILE_SIZE']), batch_size)
        flash_bwd_dq_kernel[grid_dq](
            Q_2d, K_2d, V_2d, O_2d, dO_2d, L_2d, DD_2d,
            dQ,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_ddb, stride_ddq,
            n_queries, n_keys,
            scale, D,
            ctx.is_causal,
            # ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
        )
        
        # Pass 2: Compute dK and dV
        # grid_dkv = (math.ceil(n_keys / ctx.K_TILE_SIZE), batch_size)
        def grid_dkv(meta):
            return (math.ceil(n_queries / meta['K_TILE_SIZE']), batch_size)
        flash_bwd_dkv_kernel[grid_dkv](
            Q_2d, K_2d, V_2d, O_2d, dO_2d, L_2d, DD_2d,
            dK, dV,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_ddb, stride_ddq,
            n_queries, n_keys,
            scale, D,
            ctx.is_causal,
            # ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
        )
        
        # Reshape back to original shape
        dQ = rearrange(dQ, "(b q) d -> b q d", b=batch_size)
        dK = rearrange(dK, "(b k) d -> b k d", b=batch_size)
        dV = rearrange(dV, "(b k) d -> b k d", b=batch_size)
        
        return dQ, dK, dV, None

# for benchmarking
# ...existing code...

def get_autotuned_config(n_heads, d_head, sequence_length, dtype=torch.bfloat16, device='cuda'):
    """Trigger autotuning for backward kernels and return discovered best configs.

    This runs a forward + backward to exercise the backward kernels' autotuning,
    then reads back `best_config` from each autotuned Triton kernel (if available).
    """
    print(f"🔍 Triggering autotune for backward kernels (n_heads={n_heads}, d_head={d_head}, seq_len={sequence_length})...")
    # Create small tensors to trigger kernel compilation/autotuning
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=device, dtype=dtype, requires_grad=True
    )

    # flash = torch.compile(FlashAttentionTritonOptimized.apply)
    flash = FlashAttentionTritonOptimized.apply

    # Run forward to populate ctx.saved_tensors (forward uses existing forward impl)
    o = flash(q, k, v, True)

    # Run backward to trigger backward kernels' autotune
    loss = o.sum()
    loss.backward()

    configs = {}

    # Try to read best_config for dQ kernel
    try:
        print(flash_fwd_kernel.bset_config)
        print("here")
        # cfg = getattr(flash_bwd_dq_kernel, "best_config", None)
        cfg = getattr(flash_fwd_kernel, "best_config", None)
        if cfg is not None:
            configs["bwd_dq"] = {
                "Q_TILE_SIZE": cfg.kwargs.get("Q_TILE_SIZE"),
                "K_TILE_SIZE": cfg.kwargs.get("K_TILE_SIZE"),
                "num_stages": cfg.num_stages,
                "num_warps": cfg.num_warps,
            }
            print("✅ flash_bwd_dq_kernel best config:", configs["bwd_dq"])
        else:
            print("⚠️ flash_bwd_dq_kernel completed autotune but no accessible best_config.")
    except Exception as e:
        print("⚠️ Error reading flash_bwd_dq_kernel.best_config:", e)

    # Try to read best_config for dK/dV kernel
    try:
        cfg = getattr(flash_bwd_dkv_kernel, "best_config", None)
        if cfg is not None:
            configs["bwd_dkv"] = {
                "Q_TILE_SIZE": cfg.kwargs.get("Q_TILE_SIZE"),
                "K_TILE_SIZE": cfg.kwargs.get("K_TILE_SIZE"),
                "num_stages": cfg.num_stages,
                "num_warps": cfg.num_warps,
            }
            print("✅ flash_bwd_dkv_kernel best config:", configs["bwd_dkv"])
        else:
            print("⚠️ flash_bwd_dkv_kernel completed autotune but no accessible best_config.")
    except Exception as e:
        print("⚠️ Error reading flash_bwd_dkv_kernel.best_config:", e)

    if not configs:
        print("ℹ️ No autotuned configs discovered or accessible.")
        return None
    return configs

# for benchmarking
def test_timing_flash_forward_backward(
    test, n_heads, d_head, sequence_length, dtype=torch.bfloat16, device='cuda', track_memory=False
):
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=device, dtype=dtype, requires_grad=True
    )
    
    # For autotuned version, use the function directly since tile sizes are auto-determined
    flash = torch.compile(FlashAttentionTritonOptimized.apply)
    # flash = FlashAttentionTritonBackward.apply

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
    # Test forward + backward to see memory impact
    print("\n--- Forward + Backward Pass (4096) ---")
    get_autotuned_config(16, 128, 4096, dtype=torch.float16)
    test_timing_flash_forward_backward(
        "forward_backward", 16, 128, 4096, 
        dtype=torch.bfloat16, track_memory=True
    )