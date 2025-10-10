import torch
import math
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
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
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load Q, O, dO, L, D for this tile
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
    O_block = tl.load(O_block_ptr, boundary_check=(0,), padding_option='zero')
    dO_block = tl.load(dO_block_ptr, boundary_check=(0,), padding_option='zero')
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
    D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')
    
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
            dS = P * (dP - D_block[:, None])
            
            # Accumulate dQ += dS @ K * scale
            dQ_block += scale * tl.dot(dS.to(K_block.dtype), K_block, allow_tf32=False)
        
        # Advance K and V pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Store dQ result
    tl.store(dQ_block_ptr, dQ_block.to(Q_block.dtype), boundary_check=(0,))


@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
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
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
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
            D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')
            
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
            dS = P * (dP - D_block[:, None])
            
            # Accumulate dK += dS^T @ Q * scale
            dK_block += scale * tl.dot(tl.trans(dS.to(Q_block.dtype)), Q_block, allow_tf32=False)
        
        # Advance Q, O, dO, L, D pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
    
    # Store dK and dV results
    tl.store(dK_block_ptr, dK_block.to(K_block.dtype), boundary_check=(0,))
    tl.store(dV_block_ptr, dV_block.to(V_block.dtype), boundary_check=(0,))


class FlashAttentionTritonBackward(torch.autograd.Function):
    """FlashAttention with Triton backward pass (2-pass algorithm)"""
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=16, B_k=16):
        # Use the existing forward kernel from your implementation
        from cs336_systems.flashattention import FlashAttentionTriton
        return FlashAttentionTriton.forward(ctx, Q, K, V, is_causal, B_q, B_k)
    
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
        stride_db = n_queries
        stride_dq = 1
        
        scale = 1 / (D ** 0.5)
        
        # Pass 1: Compute dQ
        grid_dq = (math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size)
        flash_bwd_dq_kernel[grid_dq](
            Q_2d, K_2d, V_2d, O_2d, dO_2d, L_2d, DD_2d,
            dQ,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_db, stride_dq,
            n_queries, n_keys,
            scale, D,
            ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
            ctx.is_causal,
        )
        
        # Pass 2: Compute dK and dV
        grid_dkv = (math.ceil(n_keys / ctx.K_TILE_SIZE), batch_size)
        flash_bwd_dkv_kernel[grid_dkv](
            Q_2d, K_2d, V_2d, O_2d, dO_2d, L_2d, DD_2d,
            dK, dV,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_db, stride_dq,
            n_queries, n_keys,
            scale, D,
            ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
            ctx.is_causal,
        )
        
        # Reshape back to original shape
        dQ = rearrange(dQ, "(b q) d -> b q d", b=batch_size)
        dK = rearrange(dK, "(b k) d -> b k d", b=batch_size)
        dV = rearrange(dV, "(b k) d -> b k d", b=batch_size)
        
        return dQ, dK, dV, None, None, None