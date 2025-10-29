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
        # `torch._utils._flatten_dense_tensors` created new tensors
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