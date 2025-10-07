import torch
import math
import triton
import triton.language as tl
from einops import einsum, rearrange

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
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

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
    # tl.device_print("Q_block", Q_block)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option='zero')
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')

        S = scale * tl.dot(Q_block, tl.trans(K_block))
        m_curr = tl.maximum(m, tl.max(S, axis=-1))
        P = tl.exp(S - m_curr.expand_dims(axis=0))

        alpha = tl.exp(m - m_curr)
        L_block = alpha * L_block  + tl.sum(P, axis=-1)
        # according to Claude, tl does not have `diag` so need to use broadcasting
        O_block = alpha[:, None] * O_block
        # using `acc` for `tl.float32`
        O_block = tl.dot(P.to(V_block.dtype), V_block, acc=O_block)
        m = m_curr

        # Move the pointer to next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block = (1 / L_block)[:, None] * O_block
    L_block = m + tl.log(L_block)

    tl.store(O_block_ptr, O_block, boundary_check=(0,))
    tl.store(L_block_ptr, L_block, boundary_check=(0,))

class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # cache Q, K and V?
        D = Q.shape[-1]

        # reshape input tensor to 2D, i.e., remove batch dim
        Q_input_shape = Q.shape
        Q = rearrange(Q, "... d -> (...) d")
        K_input_shape = K.shape
        K = rearrange(K, "... d -> (...) d")
        V = rearrange(V, "... d -> (...) d")

        ctx.save_for_backward(Q, K, V)

        for t in [Q, K, V]:
            assert t.is_cuda, "Expected CUDA tensors"
            assert t.is_contiguous(), "Our pointer arithmetic will assume contiguous inputs"

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.Q_input_shape = Q_input_shape
        ctx.K_input_shape = K_input_shape

        O = torch.empty(Q.shape, device=Q.device)
        L = torch.zeros(Q.shape[:-1], device=Q.device)

        stride_qb = math.prod(Q_input_shape[-2:])
        stride_qq = D
        stride_kb = math.prod(K_input_shape[-2:])
        stride_kk = D
        stride_vb = math.prod(K_input_shape[-2:])
        stride_vk = D
        stride_qd = 1
        stride_kd = 1
        stride_vd = 1
        stride_ob = stride_qb
        stride_oq = stride_qq
        stride_od = 1
        stride_lb = Q_input_shape[0]
        stride_lq = 1
        N_QUERIES = math.prod(Q_input_shape[:-1])
        N_KEYS = math.prod(K_input_shape[:-1])
        scale = 1 / D ** 0.5
         
        flash_fwd_kernel[(math.ceil(O.shape[1]/ctx.Q_TILE_SIZE), Q_input_shape[0])](
            Q, K, V, O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale, D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
        )

        return O.view(Q_input_shape)
    def backward(ctx):
        raise NotImplementedError