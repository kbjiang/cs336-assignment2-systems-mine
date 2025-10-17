import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import itertools

class TopyModel(torch.nn.Module):
    def __init__(self, d_h):
        super.__init__()
        self.l1 = nn.Linear(d_h, d_h *4)
        self.l2 = nn.Linear(d_h * 4, d_h)
        self.relu = nn.Relu()

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Use gloo backend for CPU, nccl for GPU
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def get_device(rank, backend):
    if backend == "gloo":
        return torch.device("cpu")
    elif backend == "nccl":
        return torch.device(f"cuda:{rank}")
    else:
        raise ValueError("wrong backend")

def get_batch(batch_size, dh, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch_size, dh)

def naive_ddp_main(rank, backend, world_size, batch_size=32, d_h=256):
    setup(rank, world_size, backend)
    # -1. Initialize model on each rank
    model = TopyModel(d_h)

    # 0. broadcast model params from rank 0 to every other rank
    if rank == 0:  # do I need this?
        for param in model.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=True)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # optimizer after params has been synced
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(5):
        # 1. Shard and move data to the appropriate device
        local_batch_size = batch_size // world_size
        data = get_batch(batch_size, d_h).to(get_device(rank, backend))
        data = data[rank*local_batch_size:(rank+1)*local_batch_size, :]

        # 2. forward and backward to get gradient each rank
        output = model(data)
        loss = output.square().mean()
        optimizer.zero_grad()
        loss.backward()

        # Do I want to make sure all gradients have been calculated?
        dist.barrier() 

        # 2.1 all-reduce to average the gradients and copy to each rank
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()

        # 3. each rank do optimizer step and update its copy of gradient
        optimizer.step()

    # Cleanup to avoid warning
    dist.destroy_process_group()  


if __name__ == "__main__":
    # backends = ["gloo", "nccl"]
    backend = "gloo"
    world_size = 2
    mp.spawn(fn=naive_ddp_main, args=(backend, world_size), nprocs=world_size, join=True)