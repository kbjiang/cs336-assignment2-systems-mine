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

# cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
# cs336_basics.model.SwiGLU = AnnotatedSwiGLU

def run_test(
    dataset: np.ndarray, 
    model: BasicsTransformerLM, 
    optimizer: AdamW, 
    batch_size: int, 
    context_length: int, 
    warmup_steps: int, 
    train_steps: int, 
    do_backward: bool, 
    use_mixed_precision: bool, 
    mixed_precision_dtype: torch.dtype, 
    do_memory_profiling: bool,
    memory_profile_name: str = "memory-snapshot.pickle",
) -> tuple[float, float]:
    x, y = get_batch(
        dataset, batch_size, context_length, "cuda" 
    )

    def forward_step():
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        torch.cuda.synchronize()
        return loss

    def backward_step(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # `nullcontext` to control mixed precision or not
    precision_context = torch.autocast(device_type="cuda", dtype=mixed_precision_dtype) if use_mixed_precision else nullcontext()
    def train_step():
        with nvtx.range("Forward Pass"):
            with precision_context:
                loss = forward_step()
        if do_backward:
            with nvtx.range("Backward Pass"):
                backward_step(loss)


    # Warmup steps
    with nvtx.range("Warmup Phase"):
        for _ in range(warmup_steps):
            train_step()

    if do_memory_profiling:
        # Start recording memory history
        torch.cuda.memory._record_memory_history(max_entries=1000_000)
    # Timed steps - separate timing for forward and backward
    forward_times = []
    backward_times = []
    with nvtx.range("Timing Phase"):
        for _ in range(train_steps):
            # Time forward pass
            forward_time = timeit.timeit(lambda: forward_step(), number=1)
            forward_times.append(forward_time)

            if do_backward:
                # For backward timing, we need the loss from forward pass
                loss = forward_step()
                backward_time = timeit.timeit(lambda: backward_step(loss), number=1)
                backward_times.append(backward_time)
            else:
                backward_times.append(0.0)

    if do_memory_profiling:
        # Savea pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(memory_profile_name)
        # Stop recording history
        torch.cuda.memory._record_memory_history(enabled=None)

    return np.mean(forward_times).item(), np.std(forward_times).item(), np.mean(backward_times).item(), np.std(backward_times).item(), 

# Add this mapping after your imports
MODEL_CONFIGS = {
    "small": SimpleNamespace(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": SimpleNamespace(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": SimpleNamespace(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": SimpleNamespace(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": SimpleNamespace(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

# Add DTYPE_MAP here, right after parsing arguments
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark transformer training')
    
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--rope-theta', type=int, default=10000, help='RoPE theta value')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--context-length', type=int, default=256, help='Context length')

    # Add the size argument
    parser.add_argument('--size', choices=list(MODEL_CONFIGS.keys()), default='small', 
                        help='Model size configuration')

    # Keep individual arguments for override capability
    parser.add_argument('--d-model', type=int, help='Model dimension')
    parser.add_argument('--d-ff', type=int, help='Feed forward dimension')
    parser.add_argument('--num-layers', type=int, help='Number of layers')
    parser.add_argument('--num-heads', type=int, help='Number of attention heads')
    parser.add_argument('--warmup-steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--train-steps', type=int, default=10, help='Number of training steps to time')
    parser.add_argument('--warmup', action='store_true', help='If passed, do warmup.')
    parser.add_argument('--backward', action='store_true', help='If passed, do backward pass')
    parser.add_argument('--compile', action='store_true', help='If passed, do JIT compilation')
    parser.add_argument('--mixed-precision', action='store_true', help='If passed, use mixed precision')
    parser.add_argument('--mixed-precision-dtype', default='bfloat16', help='Mixed precision dtype')
    parser.add_argument('--memory-profiling', action='store_true', help='If passed, do memory profiling')
    parser.add_argument('--memory-profile-name', type=str, default='memory-snapshot.pickle', help='Name for memory profile output file (only used with --memory-profiling)')
    
    args = parser.parse_args()

    # Get config from size, then override with individual arguments if provided
    config = MODEL_CONFIGS[args.size]
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.num_heads is not None:
        config.num_heads = args.num_heads

    
    # Convert string to actual dtype
    mixed_precision_dtype = DTYPE_MAP[args.mixed_precision_dtype]

    # Validation (optional - warn if name provided without profiling)
    if args.memory_profile_name != 'memory-snapshot.pickle' and not args.memory_profiling:
        print("Warning: --memory-profile-name will be ignored without --memory-profiling")
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to("cuda:0")
    if args.compile:
        model.compile()
    
    optimizer = AdamW(model.parameters())
    dataset = np.random.randint(0, args.vocab_size, 1024)
    
    forward_mean, forward_std, backward_mean, backward_std = run_test(
        dataset=dataset, 
        model=model, 
        optimizer=optimizer, 
        batch_size=args.batch_size, 
        context_length=args.context_length, 
        warmup_steps=args.warmup_steps if args.warmup else 0,
        train_steps=args.train_steps, 
        do_backward=args.backward, 
        use_mixed_precision=args.mixed_precision,
        mixed_precision_dtype=mixed_precision_dtype,
        do_memory_profiling=args.memory_profiling,
        memory_profile_name=args.memory_profile_name)

    result = {
        "num_steps": args.train_steps,
        "warmup": args.warmup,
        "backward": args.backward,
        "compile": args.compile,
        "d_model": config.d_model,
        "d_off": config.d_ff,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "forward_mean": round(forward_mean, 4),
        "forward_std": round(forward_std, 4),
        "backward_mean": round(backward_mean, 4),
        "backward_std": round(backward_std, 4),
    }

    print(json.dumps(result))
