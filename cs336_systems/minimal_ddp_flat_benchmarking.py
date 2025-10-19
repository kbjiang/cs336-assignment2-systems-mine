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

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    # Use gloo backend for CPU, nccl for GPU
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def get_device(rank, backend):
    if backend == "gloo":
        return torch.device("cpu")
    elif backend == "nccl":
        # return torch.device(f"cuda:{rank}")
        return f"cuda:{rank}"
    else:
        raise ValueError("wrong backend")

def run_test(
    rank: int,
    world_size: int,
    dataset: np.ndarray, 
    model: BasicsTransformerLM, 
    optimizer: AdamW, 
    batch_size: int = 4, 
    context_length: int = 512, 
) -> tuple[float, float]:
    setup(rank, world_size, "nccl")

    model.to(get_device(rank, "nccl"))
    def train_step():
        x, y = get_batch(
            dataset, batch_size, context_length, get_device(rank, "nccl")
        )
        total_start = time.time()
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        # Discussion!
        # If I added dist.barrier() before timing communication, you'd be measuring
        # from when all ranks are ready, not from when this rank reaches the communication.
        # This might hide differences in how long backward takes on different ranks.

        # 2.1 all-reduce to average the gradients and copy to each rank
        # benchmarking just the communication part
        comm_start = time.time()
        # flatten gradients so only one `all_reduce` will suffice
        grads = [param.grad for param in model.parameters()]
        grads_flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(tensor=grads_flat, op=dist.ReduceOp.AVG, async_op=False)
        torch.cuda.synchronize()
        comm_duration = time.time() - comm_start

        # assign communicated gradients to model 
        grads_unflat = torch._utils._unflatten_dense_tensors(grads_flat, grads)
        for param, grad in zip(model.parameters(), grads_unflat):
            param.grad = grad

        optimizer.step()
        torch.cuda.synchronize()

        total_duration = time.time() - total_start

        return comm_duration, total_duration

    comm_average = 0
    total_average = 0
    for i in range(15):
        comm_duration, total_duration = train_step()
        if i > 4:
            comm_average = comm_duration/(i-4) + comm_average*(i-5)/(i-4)
            total_average = total_duration/(i-4) + total_average*(i-5)/(i-4)

    print(f"Rank {rank}: Total time {total_average:.5f}, Comm time {comm_average:.5f} ")

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
    optimizer = AdamW(model.parameters(), lr=0.01)
    dataset = np.random.randint(0, 10_000, 1024)
    world_size = 2
    mp.spawn(fn=run_test, args=(world_size, dataset, model, optimizer), nprocs=world_size, join=True)