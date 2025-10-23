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

class DDPIndividualParameters:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handles = []  # collective op handles
        def post_grad_hook(param) -> None:
            # return immediately after all_reduce to continue back-prop
            # need to append the handle for waiting later
            handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
            # print("hook is working.")
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(post_grad_hook)
            # broadcast model params from rank 0 to every other rank
            dist.broadcast(tensor=param.data, src=0, async_op=False)
            # torch.cuda.synchronize()
    def __call__(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    def finish_gradient_synchronization(self):
        """Wait for all async all_reduce operations to complete"""
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        self.handles.clear()
    # to pass test.
    def __getattr__(self, name):
        return getattr(self.module, name)

def run_test(
    rank: int,
    world_size: int,
    dataset: np.ndarray, 
    module: BasicsTransformerLM, 
    batch_size: int = 4, 
) -> tuple[float, float]:
    setup(rank, world_size, "nccl")

    # Move model to GPU BEFORE wrapping with DDP (for NCCL broadcast)
    module.to(get_device(rank, "nccl"))
    model = DDPIndividualParameters(module)
    optimizer = AdamW(model.module.parameters(), lr=0.01)
    def train_step():
        x, y = get_batch(
            dataset, batch_size, model.module.context_length, get_device(rank, "nccl")
        )
        total_start = time.time()
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        optimizer.zero_grad()
        with nvtx.range("backward pass"):
            loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()

        # Discussion!
        # If I added dist.barrier() before timing communication, I'd be measuring
        # from when all ranks are ready, not from when this rank reaches the communication.
        # This might hide differences in how long backward takes on different ranks.

        total_duration = time.time() - total_start

        return total_duration

    total_average = 0
    for i in range(15):
        total_duration = train_step()
        if i > 4:
            total_average = total_duration/(i-4) + total_average*(i-5)/(i-4)

    print(f"Rank {rank}: Total time {total_average:.5f}")
    # Cleanup to avoid warning
    dist.destroy_process_group()  

if __name__ == "__main__":
    module = BasicsTransformerLM(
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
    mp.spawn(fn=run_test, args=(world_size, dataset, module), nprocs=world_size, join=True)