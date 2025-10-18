import os
import timeit
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
    context_length: int = 256, 
) -> tuple[float, float]:
    setup(rank, world_size, "nccl")
    x, y = get_batch(
        dataset, batch_size, context_length, get_device(rank, "nccl")
    )

    model.to(get_device(rank, "nccl"))
    def train_step():
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()

        # 2.1 all-reduce to average the gradients and copy to each rank
        # how do I add benchmarking to just the communication part?
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            torch.cuda.synchronize()

        optimizer.step()
        torch.cuda.synchronize()

    timeit.timeit(lambda: train_step(), number=5) # warmup
    duration = timeit.timeit(lambda: train_step(), number=10)
    if rank == 0:
        print(f"Total time: {duration}")

    # Cleanup to avoid warning
    dist.destroy_process_group()  
    

if __name__ == "__main__":
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=256,
        d_model=1600,
        d_ff=6400,
        num_layers=16,
        num_heads=4,
        rope_theta=10_000,
    )
    optimizer = AdamW(model.parameters(), lr=0.01)
    dataset = np.random.randint(0, 10_000, 1024)
    world_size = 2
    mp.spawn(fn=run_test, args=(world_size, dataset, model, optimizer), nprocs=world_size, join=True)