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
        self.params_1st = self.params[:self.idx_half]
        self.params_2nd = self.params[self.idx_half:]
        self.rank = dist.get_rank()
        self._params = self.params_1st if self.rank == 0 else self.params_2nd
        self._optimizer = optimizer_cls(self._params, **kwargs)

    def step(self, closure=None, **kwargs):
        if closure is not None:
            raise NotImplementedError
        
        # take a step on share of parameters
        self._optimizer.step(**kwargs)

        # both ranks need to define flat_params_* for broadcasting/receiving
        flat_params_1st = torch._utils._flatten_dense_tensors(self.params_1st)
        flat_params_2nd = torch._utils._flatten_dense_tensors(self.params_2nd)
        dist.broadcast(flat_params_1st, src=0, async_op=False)
        dist.broadcast(flat_params_2nd, src=1, async_op=False)

        # update 1st half params if rank 1
        if self.rank == 1:
            recv_params_1st = torch._utils._unflatten_dense_tensors(flat_params_1st, self.params_1st)
            for p, new in zip(self.params_1st, recv_params_1st):
                p.data.copy_(new)
        
        # update 2nd half params if rank 0
        if self.rank == 0:
            recv_params_2nd = torch._utils._unflatten_dense_tensors(flat_params_2nd, self.params_2nd)
            for p, new in zip(self.params_2nd, recv_params_2nd):
                p.data.copy_(new)

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