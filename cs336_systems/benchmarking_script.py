import timeit
import numpy as np
import torch
import argparse
import json

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


def run_test(dataset, model, optimizer, batch_size, context_length, warmup_steps, train_steps, do_backward):
    x, y = get_batch(
        dataset, batch_size, context_length, "cuda" 
    )
    def train_step():
        y_hat = model(x)
        if do_backward:
            optimizer.zero_grad()
            loss = cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Warmup steps
    for _ in range(warmup_steps):
        train_step()

    # Timed steps
    elapses = []
    for _ in range(train_steps):
        elapsed = timeit.timeit(train_step, number=1)
        elapses.append(elapsed)
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
    parser.add_argument('--warmup-unaware', action='store_true', help='If passed, do not consider warmup.')
    parser.add_argument('--skip-backward', action='store_true', help='If passed, exclude backward pass in timing')
    
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
        args.warmup_steps if not args.warmup_unaware else 0,
        args.train_steps, not args.skip_backward)

    result = {
        "num_steps": args.train_steps,
        "warmup_awareness": not args.warmup_unaware,
        "backward_inclusion": not args.skip_backward,
        "d_model": args.d_model,
        "d_off": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "elapsed_mean": round(elapsed_mean, 4),
        "elapsed_std": round(elapsed_std, 4),
    }

    print(json.dumps(result))
