import timeit
import numpy as np
import torch
import argparse
import json
import math

import torch.cuda.nvtx as nvtx
from torch import nn
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool

class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def run_test(model, optimizer, warmup_steps, train_steps, do_backward):
    x = torch.randn(BATCH_SIZE, INPUT_SIZE).cuda()
    y = torch.randn(BATCH_SIZE, OUTPUT_SIZE).cuda()
    def train_step():
        with nvtx.range("Forward Pass"):
            y_hat = model(x)
        if do_backward:
            with nvtx.range("Backward Pass"):
                optimizer.zero_grad()
                loss = nn.MSELoss()(y_hat, y)
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
    torch.cuda.memory._dump_snapshot("memory-snapshot-toy.pickle")
    # Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)

    return np.mean(elapses).item(), np.std(elapses).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark transformer training')
    
    parser.add_argument('--warmup-steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--train-steps', type=int, default=10, help='Number of training steps to time')
    parser.add_argument('--warmup', action='store_true', help='If passed, do warmup.')
    parser.add_argument('--backward', action='store_true', help='If passed, do backward pass')
    
    args = parser.parse_args()

    INPUT_SIZE = 256
    OUTPUT_SIZE = 512
    BATCH_SIZE = 32
    DTYPE = torch.bfloat16

    model = ToyModel(INPUT_SIZE, OUTPUT_SIZE).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    elapsed_mean, elapsed_std = run_test(
        model, optimizer,
        args.warmup_steps if args.warmup else 0,
        args.train_steps, args.backward)

    result = {
        "num_steps": args.train_steps,
        "warmup": args.warmup,
        "backward": args.backward,
        "elapsed_mean": round(elapsed_mean, 4),
        "elapsed_std": round(elapsed_std, 4),
    }

    print(json.dumps(result))
