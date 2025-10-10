# FlashAttention-2 Backward Pass Algorithm

## Algorithm 2: Tiled FlashAttention-2 Backward Pass (Corrected)

**Require:** $\mathbf{Q}, \mathbf{O}, d\mathbf{O} \in \mathbb{R}^{N_q \times d}$, $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_k \times d}$, $\mathbf{L} \in \mathbb{R}^{N_q}$, tile sizes $B_q, B_k$

Compute $D = \text{rowsum}(d\mathbf{O} \circ \mathbf{O}) \in \mathbb{R}^{N_q}$

Split $\mathbf{Q}, \mathbf{O}, d\mathbf{O}$ into $T_q = \left\lceil\frac{N_q}{B_q}\right\rceil$ tiles $\mathbf{Q}_1, \ldots, \mathbf{Q}_{T_q}$, $\mathbf{O}_1, \ldots, \mathbf{O}_{T_q}$, $d\mathbf{O}_1, \ldots, d\mathbf{O}_{T_q}$, each of size $B_q \times d$

Split $\mathbf{K}, \mathbf{V}$ into $T_k = \left\lceil\frac{N_k}{B_k}\right\rceil$ tiles $\mathbf{K}^{(1)}, \ldots, \mathbf{K}^{(T_k)}$ and $\mathbf{V}^{(1)}, \ldots, \mathbf{V}^{(T_k)}$, each of size $B_k \times d$

Split $L, D$ into $T_q$ tiles $L_1, \ldots, L_{T_q}$ and $D_1, \ldots, D_{T_q}$, each of size $B_q$

---

## Option 1: Single Pass with Atomics

$$
\begin{align*}
&\textbf{for } j = 1, \ldots, T_k \textbf{ do} \\
&\quad \text{Load } \mathbf{K}^{(j)}, \mathbf{V}^{(j)} \text{ from global memory} \\
&\quad \text{Initialize } d\mathbf{K}^{(j)} = d\mathbf{V}^{(j)} = \mathbf{0} \in \mathbb{R}^{B_k \times d} \\
&\quad \textbf{for } i = 1, \ldots, T_q \textbf{ do} \\
&\quad\quad \text{Load } \mathbf{Q}_i, \mathbf{O}_i, d\mathbf{O}_i, d\mathbf{Q}_i \text{ from global memory} \\
&\quad\quad \text{Compute tile of attention scores } \mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i(\mathbf{K}^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k} \\
&\quad\quad \text{Compute attention probabilities } \mathbf{P}_i^{(j)} = \exp\left(\mathbf{S}_i^{(j)} - L_i\right) \in \mathbb{R}^{B_q \times B_k} \\
&\quad\quad \text{Compute } d\mathbf{V}^{(j)} \mathrel{+}= (\mathbf{P}_i^{(j)})^\top d\mathbf{O}_i \in \mathbb{R}^{B_k \times d} \\
&\quad\quad \text{Compute } d\mathbf{P}_i^{(j)} = d\mathbf{O}_i \mathbf{V}_j^\top \in \mathbb{R}^{B_q \times B_k} \\
&\quad\quad \text{Compute } d\mathbf{S}_i^{(j)} = \mathbf{P}_i^{(j)} \circ \left(d\mathbf{P}_i^{(j)} - D_i\right) / \sqrt{d} \in \mathbb{R}^{B_q \times B_k} \\
&\quad\quad \text{Load } d\mathbf{Q}_i \text{ from global memory, then update } d\mathbf{Q}_i \mathrel{+}= d\mathbf{S}_i^{(j)} \mathbf{K}^{(j)} \in \mathbb{R}^{B_q \times d}, \\
&\quad\quad \text{and write back to global memory. Must be atomic for correctness!} \\
&\quad\quad \text{Compute } d\mathbf{K}^{(j)} \mathrel{+}= (d\mathbf{S}_i^{(j)})^\top \mathbf{Q}_i \in \mathbb{R}^{B_k \times d} \\
&\quad \textbf{end for} \\
&\quad \text{Write } d\mathbf{K}^{(j)} \text{ and } d\mathbf{V}^{(j)} \text{ to global memory as the } j\text{-th tiles of } d\mathbf{K} \text{ and } d\mathbf{V} \\
&\textbf{end for} \\
&\\
&\text{Return } d\mathbf{Q}, d\mathbf{K}, d\mathbf{V}
\end{align*}
$$

---

## Option 2: Two Passes (Recommended - No Atomics)

### Pass 1: Compute $d\mathbf{Q}$
$$
\begin{align*}
&\text{Initialize } d\mathbf{Q} = \mathbf{0} \text{ in global memory} \\
&\\
&\textbf{for } i = 1, \ldots, T_q \textbf{ do} \\
&\quad \text{Load } \mathbf{Q}_i, \mathbf{O}_i, d\mathbf{O}_i, L_i, D_i \text{ from global memory} \\
&\quad \text{Initialize } d\mathbf{Q}_i = \mathbf{0} \text{ in SRAM} \\
&\quad \textbf{for } j = 1, \ldots, T_k \textbf{ do} \\
&\quad\quad \text{Load } \mathbf{K}^{(j)}, \mathbf{V}^{(j)} \text{ from global memory} \\
&\quad\quad \text{Compute } \mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i(\mathbf{K}^{(j)})^\top}{\sqrt{d}} \\
&\quad\quad \text{Compute } \mathbf{P}_i^{(j)} = \exp\left(\mathbf{S}_i^{(j)} - L_i\right) \\
&\quad\quad \text{Compute } d\mathbf{P}_i^{(j)} = d\mathbf{O}_i \mathbf{V}_j^\top \\
&\quad\quad \text{Compute } d\mathbf{S}_i^{(j)} = \mathbf{P}_i^{(j)} \circ \left(d\mathbf{P}_i^{(j)} - D_i\right) / \sqrt{d} \\
&\quad\quad \text{Accumulate } d\mathbf{Q}_i \mathrel{+}= d\mathbf{S}_i^{(j)} \mathbf{K}^{(j)} \\
&\quad \textbf{end for} \\
&\quad \text{Write } d\mathbf{Q}_i \text{ to global memory} \\
&\textbf{end for}
\end{align*}
$$

### Pass 2: Compute $d\mathbf{K}$ and $d\mathbf{V}$

$$
\begin{align*}
&\textbf{for } j = 1, \ldots, T_k \textbf{ do} \\
&\quad \text{Load } \mathbf{K}^{(j)}, \mathbf{V}^{(j)} \text{ from global memory} \\
&\quad \text{Initialize } d\mathbf{K}^{(j)} = d\mathbf{V}^{(j)} = \mathbf{0} \text{ in SRAM} \\
&\quad \textbf{for } i = 1, \ldots, T_q \textbf{ do} \\
&\quad\quad \text{Load } \mathbf{Q}_i, \mathbf{O}_i, d\mathbf{O}_i, L_i, D_i \text{ from global memory} \\
&\quad\quad \text{Compute } \mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i(\mathbf{K}^{(j)})^\top}{\sqrt{d}} \\
&\quad\quad \text{Compute } \mathbf{P}_i^{(j)} = \exp\left(\mathbf{S}_i^{(j)} - L_i\right) \\
&\quad\quad \text{Accumulate } d\mathbf{V}^{(j)} \mathrel{+}= (\mathbf{P}_i^{(j)})^\top d\mathbf{O}_i \\
&\quad\quad \text{Compute } d\mathbf{P}_i^{(j)} = d\mathbf{O}_i \mathbf{V}_j^\top \\
&\quad\quad \text{Compute } d\mathbf{S}_i^{(j)} = \mathbf{P}_i^{(j)} \circ \left(d\mathbf{P}_i^{(j)} - D_i\right) / \sqrt{d} \\
&\quad\quad \text{Accumulate } d\mathbf{K}^{(j)} \mathrel{+}= (d\mathbf{S}_i^{(j)})^\top \mathbf{Q}_i \\
&\quad \textbf{end for} \\
&\quad \text{Write } d\mathbf{K}^{(j)} \text{ and } d\mathbf{V}^{(j)} \text{ to global memory} \\
&\textbf{end for} \\
&\text{Return } d\mathbf{Q}, d\mathbf{K}, d\mathbf{V}
\end{align*}
$$

---

## Key Insights

### Why Two Passes Are Superior

1. **No Atomic Operations Required**
   - Pass 1: Each thread block computes one complete $d\mathbf{Q}_i$ tile
   - Pass 2: Each thread block computes one complete $d\mathbf{K}^{(j)}, d\mathbf{V}^{(j)}$ tile pair
   - No race conditions between thread blocks

2. **Performance Benefits**
   - Atomic operations serialize memory access (extremely slow on GPUs)
   - Two passes allow fully parallel execution within each pass
   - Better memory coalescing and cache utilization

3. **Recomputation vs Storage Trade-off**
   - $\mathbf{P}$ is recomputed in each pass instead of stored
   - Recomputation cost < atomic synchronization overhead
   - Avoids $O(N_q \times N_k)$ memory requirement for storing $\mathbf{P}$

### Mathematical Formulation

The gradient computations follow from the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{ij}} = \mathbf{P}_{ij} \circ \left(\frac{\partial \mathcal{L}}{\partial \mathbf{P}_{ij}} - D_i\right)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}_i} = \frac{1}{\sqrt{d}} \sum_{j=1}^{T_k} \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{ij}} \mathbf{K}^{(j)}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}^{(j)}} = \frac{1}{\sqrt{d}} \sum_{i=1}^{T_q} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{ij}}\right)^\top \mathbf{Q}_i$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{V}^{(j)}} = \sum_{i=1}^{T_q} \mathbf{P}_{ij}^\top d\mathbf{O}_i$$

Where $D_i = \text{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i)$ compensates for the softmax normalization in the backward pass.

