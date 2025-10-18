import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import itertools

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

def get_data(rank, world_size, size_in_mb, backend, batch_size=64):
    num_elem = size_in_mb * 1024 // 4 // world_size // batch_size
    return torch.randn(batch_size, num_elem, device=get_device(rank, backend))

def all_reduce_benchmarking(rank, backend, world_size, data_size):
    setup(rank, world_size, backend)
    # Move data to the appropriate device
    data = get_data(rank, world_size, data_size, backend)

    # warmup - use the same tensor we'll benchmark with
    # different tensor sizes trigger different code paths and memory allocation
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        # dist.barrier()
    
    # benchmarking
    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()
    # dist.barrier()
    # print(data)
    duration = time.time() - start_time

    # Gather durations from all ranks
    # Note: all_gather_object only works with gloo backend
    if backend == "gloo":
        durations = [None] * world_size
        dist.all_gather_object(durations, duration)
        if rank == 0:
            print(f"{backend}, {data_size}MB, world_size={world_size}: {sum(durations)/world_size}")
    else:
        # For NCCL, use tensor-based gathering
        duration = torch.tensor([duration], device=f"cuda:{rank}")
        gathered = [torch.zeros(1, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(gathered, duration)
        if rank == 0:
            durations = [t.item() for t in gathered]
            print(f"{backend}, {data_size}MB, world_size={world_size}: {sum(durations)/world_size}")

    dist.destroy_process_group()  # Cleanup to avoid warning


if __name__ == "__main__":
    backends = ["gloo", "nccl"]
    world_sizes = [2]
    data_sizes = [1, 10, 100, 1024]
    # data_sizes = [1]
    for backend, world_size, data_size in itertools.product(
        backends, world_sizes, data_sizes
    ):
        mp.spawn(fn=all_reduce_benchmarking, args=(backend, world_size, data_size), nprocs=world_size, join=True)