import timeit
import numpy as np
import torch
import argparse
import json
import math

import cs336_basics
from cs336_basics.model import BasicsTransformerLM, silu, Linear
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, softmax

import torch.nn as nn
import torch.cuda.nvtx as nvtx
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool
from contextlib import nullcontext
from types import SimpleNamespace

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

class PytorchAttention(nn.Module):
    """Single-Head Self-Attention
    Args:
        d_model: head embedding size
        vocab_size: for LM output

    Returns:
        Tensor of shape `(batch_size, sequence_length, vocab_size)`.
    """

    def __init__(
        self,
        d_model,
        vocab_size,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, Q, K, V) -> Float[Tensor, " ... seq d_v"]:
        seq_len = Q.shape[-2]  # Get actual sequence length from input
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
        
        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = annotated_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
        # return attn_output

        # Shape: (..., num_heads, sequence_length, vocab_size)
        output = self.lm_head(attn_output)

        return output

batch_size = 8
seq_len = 4096
d_model = 128
vocab_size = 10000
device = "cuda"

def run_test(
    model,
    batch_size: int, 
    d_model: int, 
    seq_len: int,
    vocab_size: int, 
    optimizer: AdamW, 
    warmup_steps: int, 
    train_steps: int, 
    do_backward: bool, 
    device: str,
    do_memory_profiling: bool,
    memory_profile_name: str,
) -> tuple[float, float]:
    Q = torch.randn((batch_size, seq_len, d_model), device=device)
    K = torch.randn((batch_size, seq_len, d_model), device=device)
    V = torch.randn((batch_size, seq_len, d_model), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=Q.device)
    # y = torch.randint(0, d_model, (batch_size, seq_len), device=Q.device)

    def forward_step():
        y_hat = model(Q, K, V)
        loss = cross_entropy(y_hat, y)
        torch.cuda.synchronize()
        return y_hat, loss

    def backward_step(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    def train_step():
        y_hat, loss = forward_step()
        if do_backward:
            backward_step(loss)

    # Warmup steps
    for _ in range(warmup_steps):
        train_step()

    # Timed steps - separate timing for forward and backward
    forward_times = []
    backward_times = []
    
    for _ in range(train_steps):
        # Time forward pass
        forward_time = timeit.timeit(lambda: forward_step(), number=1)
        forward_times.append(forward_time)
        
        if do_backward:
            if do_memory_profiling:
                # Start recording memory history
                torch.cuda.memory._record_memory_history(max_entries=1000_000)
    
            # For backward timing, we need the loss from forward pass
            y_hat, loss = forward_step()
            backward_time = timeit.timeit(lambda: backward_step(loss), number=1)
            backward_times.append(backward_time)

            if do_memory_profiling:
                # Save a pickle file to be loaded by PyTorch's online tool.
                torch.cuda.memory._dump_snapshot(memory_profile_name)
                # Stop recording history
                torch.cuda.memory._record_memory_history(enabled=None)

    forward_total = np.sum(forward_times).item()
    backward_total = np.sum(backward_times).item() if do_backward else 0.0
    
    return forward_total, backward_total

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Attention Benchmarking")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--memory-profiling", action="store_true")
    parser.add_argument("--memory-profile_name", type=str, default="memory-snapshot.pickle")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model = PytorchAttention(args.d_model, args.vocab_size).to(args.device)
    # model.compile()
    optimizer = AdamW(model.parameters())
    
    try:
        forward_time, backward_time = run_test(
            model=model, 
            d_model=args.d_model,
            seq_len=args.seq_len, 
            vocab_size=args.vocab_size, 
            optimizer=optimizer, 
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            train_steps=args.train_steps,
            do_backward=args.backward, 
            device=args.device,
            do_memory_profiling=args.memory_profiling,
            memory_profile_name=args.memory_profile_name,
        )

        result = {
            "d_model": args.d_model,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "backward": args.backward,
            "forward_time": round(forward_time, 4),
            "backward_time": round(backward_time, 4) if args.backward else None,
            "total_time": round(forward_time + backward_time, 4),
            "status": "success"
        }
        
    except torch.OutOfMemoryError:
        result = {
            "d_model": args.d_model,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "backward": args.backward,
            "forward_time": None,
            "backward_time": None,
            "total_time": None,
            "status": "OOM"
        }

    print(json.dumps(result))
    
