import torch
from einops import einsum

class FlashAttentionAutogradFunctionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        device = Q.device
        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape
        B_q = 16
        B_k = 16
        O = torch.zeros_like(Q, device=device)
        L = torch.zeros(batch_size, N_q, device=device)
        for i in range(0, N_q, B_q):
            Q_i = Q[:, i:i+B_q, :]
            O_i = torch.zeros_like(Q_i, device=device)
            l_i = torch.zeros(batch_size, B_q, device=device)
            m_i = torch.full((batch_size, B_q), float('-inf'), device=device)
            for j in range(0, N_k, B_k):
                K_j = K[:, j:j+B_k, :]
                V_j = V[:, j:j+B_k, :]
                S_j = 1 / d**(0.5) * einsum(Q_i, K_j, '... B_q d, ... B_k d -> ... B_q B_k')
                assert S_j.shape == (batch_size, B_q, B_k)

                m_curr = torch.max(torch.cat([m_i[:, :, None], S_j], axis=-1), axis=-1).values
                P_i = torch.exp(S_j - m_curr[:, :, None])
                assert P_i.shape == (batch_size, B_q, B_k)

                l_i = torch.exp(m_i - m_curr)*l_i + torch.sum(P_i, axis=-1)
                
                _ = torch.diag_embed(torch.exp(m_i - m_curr))
                O_i = einsum(_, O_i, '... B_q B_q, ... B_q d -> ... B_q d') + einsum(P_i, V_j, '... B_q B_k, ... B_k d -> ... B_q d')
                assert O_i.shape == (batch_size, B_q, d)

                m_i = m_curr
            O_i = einsum(torch.diag_embed(1 / l_i), O_i, '... B_q B_q, ... B_q d -> ... B_q d')
            L_i = m_i + torch.log(l_i)

            O[:, i:i+B_q] += O_i
            L[:, i:i+B_q] += L_i
        ctx.save_for_backward(Q, K, V, L, O)
        return O
    def backward(ctx):
        raise NotImplementedError