import torch
import math
import triton
import triton.language as tl
from einops import einsum, rearrange

class AttentionAutogradFunctionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=16, B_k=16):
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        D = Q.shape[-1]
        scale = 1 / (D ** 0.5)
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
        if is_causal:
            causal_mask = torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :]
            S = torch.where(causal_mask, S, -1e6)
        P = torch.softmax(S, dim=-1)
        O = einsum(P, V, '... q k, ... k d -> ... q d')
        ctx.save_for_backward(Q, K, V, P, O)
        ctx.is_causal = is_causal
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P, O = ctx.saved_tensors
        D = Q.shape[-1]
        scale = 1 / (D ** 0.5)
        dV = einsum(P, dO, '... q k, ... q d -> ... k d')
        dP = einsum(dO, V, '... q d, ... k d -> ... q k')
        DD = torch.sum(O * dO, axis=-1)
        dS = P * (dP - DD[:, :, None])
        # Do not need masking here coz we are not recomputing
        # n_queries = Q.shape[-2]
        # n_keys = K.shape[-2]
        # if ctx.is_causal:
        #     causal_mask = torch.arange(n_queries, device=dS.device)[:, None] >= torch.arange(n_keys, device=dS.device)[None, :]
        #     dS = torch.where(causal_mask, dS, -1e6)

        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') * scale
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') * scale
        return dQ, dK, dV, None, None, None


class FlashAttentionAutogradFunctionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=16, B_k=16):
        device = Q.device
        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape
        # B_q = 16
        # B_k = 16
        O = torch.zeros_like(Q, device=device)
        L = torch.zeros(batch_size, N_q, device=device)
        for i in range(0, N_q, B_q):
            Q_i = Q[:, i:i+B_q, :]
            O_i = torch.zeros_like(Q_i, device=device)
            l_i = torch.zeros(batch_size, B_q, device=device)
            m_i = torch.full((batch_size, B_q), float('-inf'), device=device)
            Q_indices = torch.arange(i, i+B_q, device=device)
            for j in range(0, N_k, B_k):
                # if all falls inside the masked out area, no calculation needed
                if is_causal and j > i + B_q -1:
                    continue
                else:
                    K_j = K[:, j:j+B_k, :]
                    V_j = V[:, j:j+B_k, :]
                    S_j = 1 / d**(0.5) * einsum(Q_i, K_j, '... B_q d, ... B_k d -> ... B_q B_k')
                    if is_causal:
                        # nice trick to get a triangular mask
                        K_indices = torch.arange(j, j+B_k, device=device)
                        causal_mask = K_indices[None, :] <= Q_indices[:, None]
                        S_j = torch.where(causal_mask, S_j, -1e6)
                    # assert S_j.shape == (batch_size, B_q, B_k)

                    m_curr = torch.max(torch.cat([m_i[:, :, None], S_j], axis=-1), axis=-1).values
                    P_i = torch.exp(S_j - m_curr[:, :, None])
                    # assert P_i.shape == (batch_size, B_q, B_k)

                    l_i = torch.exp(m_i - m_curr)*l_i + torch.sum(P_i, axis=-1)
                    
                    _ = torch.diag_embed(torch.exp(m_i - m_curr))
                    O_i = einsum(_, O_i, '... B_q B_q, ... B_q d -> ... B_q d') + einsum(P_i, V_j, '... B_q B_k, ... B_k d -> ... B_q d')
                    # assert O_i.shape == (batch_size, B_q, d)

                    m_i = m_curr
            O_i = einsum(torch.diag_embed(1 / l_i), O_i, '... B_q B_q, ... B_q d -> ... B_q d')
            L_i = m_i + torch.log(l_i)

            O[:, i:i+B_q] += O_i
            L[:, i:i+B_q] += L_i
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

# reshape to 2D, but block size is still seq_len * d_model
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
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
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

    tl.store(O_block_ptr, O_block, boundary_check=(0,))
    tl.store(L_block_ptr, L_block, boundary_check=(0,))

class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=16, B_k=16):
        # cache Q, K and V?
        batch_size, n_queries, D = Q.shape
        _, n_keys, _ = K.shape

        # reshape input tensor to 2D, i.e., remove batch dim
        Q_input_shape = Q.shape
        Q = rearrange(Q, "... d -> (...) d")
        K_input_shape = K.shape
        K = rearrange(K, "... d -> (...) d")
        V = rearrange(V, "... d -> (...) d")

        ctx.save_for_backward(Q, K, V)
        ctx.is_causal = is_causal

        for t in [Q, K, V]:
            assert t.is_cuda, "Expected CUDA tensors"
            assert t.is_contiguous(), "Our pointer arithmetic will assume contiguous inputs"

        ctx.Q_TILE_SIZE = B_q
        ctx.K_TILE_SIZE = B_k
        ctx.Q_input_shape = Q_input_shape
        ctx.K_input_shape = K_input_shape

        O = torch.empty(Q.shape, device=Q.device)
        L = torch.zeros(batch_size * n_queries, device=Q.device)

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
         
        # Your launch grid should be set as (Tq,batch_size), meaning each Triton program instance 
        # will load only elements from a single batch index, i.e., one seq_len * d_model,
        # and only read/write to a single query tile of Q, O, and L.
        flash_fwd_kernel[(math.ceil(n_queries/ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V, O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            n_queries, n_keys,
            scale, D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal,
        )

        O = O.view(Q_input_shape).contiguous()
        Q = Q.view(Q_input_shape).contiguous()
        K = K.view(K_input_shape).contiguous()
        V = V.view(K_input_shape).contiguous()
        L = L.view(batch_size, n_queries).contiguous()
        ctx.save_for_backward(Q, K, V, L, O)
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