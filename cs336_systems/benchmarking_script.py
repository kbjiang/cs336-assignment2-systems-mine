import timeit
import numpy as np
import torch
import argparse
import json
import math

import cs336_basics
from cs336_basics.model import BasicsTransformerLM, silu
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, softmax

import torch.cuda.nvtx as nvtx
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool
from contextlib import nullcontext

@nvtx.range("scaled dot production attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    with nvtx.range("computing attention score"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

@nvtx.range("SwiGLU forward")
class AnnotatedSwiGLU(cs336_basics.model.SwiGLU):
    def forward(self, x):
        with nvtx.range("SwiGLU_forward"):
            with nvtx.range("SwiGLU_w1"):
                w1_out = self.w1(x)
            with nvtx.range("SwiGLU_w3"):
                w3_out = self.w3(x)
            with nvtx.range("SwiGLU_gating"):
                gated = silu(w1_out) * w3_out
            with nvtx.range("SwiGLU_w2"):
                return self.w2(gated)

cs336_basics.model.SwiGLU = AnnotatedSwiGLU

def run_test(dataset, model, optimizer, batch_size, context_length, warmup_steps, train_steps, do_backward, use_mixed_precision):
    x, y = get_batch(
        dataset, batch_size, context_length, "cuda" 
    )

    # `nullcontext` to control mixed precision or not
    precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_mixed_precision else nullcontext()
    def train_step():
        with nvtx.range("Forward Pass"):
            with precision_context:
                y_hat = model(x)
        if do_backward:
            with nvtx.range("Backward Pass"):
                optimizer.zero_grad()
                loss = cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

    # Warmup steps
    with nvtx.range("Warmup Phase"):
        for _ in range(warmup_steps):
            train_step()

    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000_000)
    # Timed steps
    elapses = []
    with nvtx.range("Timing Phase"):
        for _ in range(train_steps):
            elapsed = timeit.timeit(train_step, number=1)
            elapses.append(elapsed)

    # Savea pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot("memory-snapshot.pickle")
    # Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)

    return np.mean(elapses).item(), np.std(elapses).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark transformer training')
    
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--rope-theta', type=int, default=10000, help='RoPE theta value')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--context-length', type=int, default=256, help='Context length')
    parser.add_argument('--d-model', type=int, default=768, help='Model dimension')
    parser.add_argument('--d-ff', type=int, default=3072, help='Feed forward dimension')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--warmup-steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--train-steps', type=int, default=10, help='Number of training steps to time')
    parser.add_argument('--warmup', action='store_true', help='If passed, do warmup.')
    parser.add_argument('--backward', action='store_true', help='If passed, do backward pass')
    parser.add_argument('--mixed-precision', action='store_true', help='If passed, use mixed precision')
    
    args = parser.parse_args()
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to("cuda:0")
    
    optimizer = AdamW(model.parameters())
    dataset = np.random.randint(0, args.vocab_size, 1024)
    
    elapsed_mean, elapsed_std = run_test(
        dataset, model, optimizer, args.batch_size, args.context_length, 
        args.warmup_steps if args.warmup else 0,
        args.train_steps, args.backward, args.mixed_precision)

    result = {
        "num_steps": args.train_steps,
        "warmup": args.warmup,
        "backward": args.backward,
        "d_model": args.d_model,
        "d_off": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "elapsed_mean": round(elapsed_mean, 4),
        "elapsed_std": round(elapsed_std, 4),
    }

    print(json.dumps(result))
