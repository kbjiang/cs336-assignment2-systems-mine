import torch
from einops import einsum


class FlashAttentionAutogradFunctionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        device = Q.device
        O = torch.zeros(N_q, d, device=device)
        L = torch.zeros(N_q, device=device)
        for i in range(0, N_q, B_q):
            Q_i = Q[i:i+B_q, :]
            O_i = torch.zeros_like(Q_i, device=device)
            l_i = torch.zeros(B_q, device=device)
            m_i = torch.full((B_q,), float('-inf'), device=device)
            for j in range(0, N_k, B_k):
                K_j = K[j:j+B_k, :]
                V_j = V[j:j+B_k, :]
                S_j = torch.rsqrt(torch.tensor(d)) * einsum(Q_i, K_j, 'B_q d, B_k d -> B_q B_k')
                assert S_j.shape == (B_q, B_k)

                m_curr = torch.max(torch.cat([m_i[:, None], S_j], axis=-1), axis=-1).values
                P_i = torch.exp(S_j - m_curr[:, None])
                assert P_i.shape == (B_q, B_k)

                l_i = torch.exp(m_i - m_curr)*l_i + torch.sum(P_i, axis=-1)
                
                _ = torch.diag(torch.exp(m_i - m_curr))
                O_i = einsum(_, O_i, 'B_q B_q, B_q d -> B_q d') + einsum(P_i, V_j, 'B_q B_k, B_k d -> B_q d')
                assert O_i.shape == (B_q, d)

                m_i = m_curr
            O_i = einsum(torch.diag(1 / l_i), O_i, 'B_q B_q, B_q d -> B_q d')
            L_i = m_i + torch.log(l_i)

            O[i:i+B_q] += O_i
            L[i:i+B_q] += L_i
        ctx.save_for_backward(Q, K, V, L, O)
        return O, L
    def backward(ctx):
        raise NotImplementedError
            
if __name__ == "__main__":
    N_q = 32
    N_k = 32
    d = 64
    B_q = 16
    B_k = 16

    Q = torch.randn(N_q, d)
    K = torch.randn(N_k, d)
    V = torch.randn(N_k, d)
    O = torch.zeros(N_q, d)
    L = torch.zeros(N_q, )