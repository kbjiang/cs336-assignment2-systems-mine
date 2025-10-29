import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM, silu
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, softmax

import torch.cuda.nvtx as nvtx
import json

def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    # Use gloo backend for CPU, nccl for GPU
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def get_device(rank, backend="nccl"):
    if backend == "gloo":
        return torch.device("cpu")
    elif backend == "nccl":
        # return torch.device(f"cuda:{rank}")
        return f"cuda:{rank}"
    else:
        raise ValueError("wrong backend")

def get_equal_halves(arr):
    """
    Assuming world size is always 2.
    """
    i, j = 0, len(arr) - 1
    sl = sr = 0
    arrl, arrr = [], []
    while j > i:
        if sl > sr:
            sr += arr[j]
            arrr.append(j)
            j -= 1
        else:
            sl += arr[i]
            arrl.append(i)
            i += 1
    
    if sl > sr:
        sr += arr[j]
        arrr.append(j)
    else:
        sl += arr[i]
        arrl.append(i)

    # return the min of right half
    return arrr[-1]

class OptimizerSharded(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, **kwargs):
        self.params = list(params)
        # init super class for things like `zero_grad()`
        # this will zero out ALL parameters, not shards.
        defaults = dict(**kwargs)
        super().__init__(self.params, defaults)

        # now real stuff
        # self.params = list(params)
        self.idx_half = get_equal_halves([p.nelement() for p in self.params])
        self.params_0th = self.params[:self.idx_half]
        self.params_1st = self.params[self.idx_half:]
        self.rank = dist.get_rank()
        self._params = self.params_0th if self.rank == 0 else self.params_1st
        self._optimizer = optimizer_cls(self._params, **kwargs)

    def step(self, closure=None, **kwargs):
        if closure is not None:
            raise NotImplementedError
        
        # take a step on share of parameters
        self._optimizer.step(**kwargs)

        # both ranks need to define flat_params_* for broadcasting/receiving
        flat_params_0th = torch._utils._flatten_dense_tensors(self.params_0th)
        flat_params_1st = torch._utils._flatten_dense_tensors(self.params_1st)
        dist.broadcast(flat_params_0th, src=0, async_op=False)
        dist.broadcast(flat_params_1st, src=1, async_op=False)

        # update 0th half params if rank 1
        if self.rank == 1:
            recv_params_0th = torch._utils._unflatten_dense_tensors(flat_params_0th, self.params_0th)
            for p, new in zip(self.params_0th, recv_params_0th):
                p.data.copy_(new)
        
        # update 1st half params if rank 0
        if self.rank == 0:
            recv_params_1st = torch._utils._unflatten_dense_tensors(flat_params_1st, self.params_1st)
            for p, new in zip(self.params_1st, recv_params_1st):
                p.data.copy_(new)

def _all_reduce_max(value: int, device: torch.device) -> int:
    t = torch.tensor([value], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())

def report_memory(description: str, device: torch.device):
    torch.cuda.synchronize(device)
    current = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    current = round(_all_reduce_max(current, device)/1024.**3, 3)
    peak = round(_all_reduce_max(peak, device)/1024.**3, 3)
    return {"description": description, "current mem GB": current, "peak mem GB": peak}

def run_test(
    rank: int,
    world_size: int,
    sharding: bool,
    dataset: np.ndarray, 
    model: BasicsTransformerLM, 
    batch_size: int = 4, 
) -> tuple[float, float]:
    setup(rank, world_size, "nccl")

    # Move model to GPU BEFORE wrapping with DDP (for NCCL broadcast)
    device = get_device(rank, "nccl")
    model.to(device)
    memory_metrics = []
    torch.cuda.reset_peak_memory_stats(device)
    memory_metrics.append(report_memory("after_model_init", device))

    if sharding:
        optimizer = OptimizerSharded(
            params=model.parameters(),
            optimizer_cls=torch.optim.AdamW,
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    def train_step(step_i):
        x, y = get_batch(
            dataset, batch_size, model.context_length, get_device(rank, "nccl")
        )
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        memory_metrics.append(report_memory(f"before step {step_i}", device))
        optimizer.step()
        memory_metrics.append(report_memory(f"after step {step_i}", device))

    # start recording memroy history
    torch.cuda.memory._record_memory_history(max_entries=1000_000)
    for i in range(2):
        train_step(i)
    # Savea pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot("memory-sharded.pickle")
    # Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)

    if rank == 0:
        print(json.dumps(memory_metrics, indent=2))
        # return memory_metrics
    # Cleanup to avoid warning
    dist.destroy_process_group()  



if __name__ == "__main__":
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=512,
        d_model=1600,
        d_ff=6400,
        num_layers=48,
        num_heads=25,
        rope_theta=10_000,
    )
    dataset = np.random.randint(0, 10_000, 1024)
    world_size = 2
    sharding = True
    mp.spawn(fn=run_test, args=(world_size, sharding, dataset, model), nprocs=world_size, join=True)