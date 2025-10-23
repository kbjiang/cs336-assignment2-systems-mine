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

def get_buckets(model, bucket_size_mb:float) -> list[list[torch.tensor]]:
    # get bucket size limit in num of elements
    bucket_size_limit = (1024**2 * bucket_size_mb)//4
    buckets = []
    current_size = 0
    current_bucket = []
    for param in reversed(list(model.parameters())):
        # skip params do not need grad calculation
        if not param.requires_grad:
            continue
        # if current param is large
        if param.nelement() > bucket_size_limit:
            if current_bucket:  
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            buckets.append([param])
        # if current param leads to overflow
        elif param.nelement() + current_size > bucket_size_limit:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param.nelement()
        # if current param is ok
        else:
            current_bucket += [param]
            current_size += param.nelement()
    # final pieces if exists
    if current_bucket:  
        buckets.append(current_bucket)

    return buckets

class DDPBucketed:
    """
    Flow:
    - Bucketing: Parameters are organized into self.buckets (list of lists)
    - Hook registration: One hook per bucket, on the last parameter
    - During backward: When last param's gradient is ready, hook fires
    - In the hook: Flatten all gradients in that bucket → all_reduce
    - After backward: Wait for all_reduce → unflatten → copy back
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        self.module = module
        self.handles = []  # collective op handles
        self.bucket_size_mb = bucket_size_mb
        self.buckets = None
        self.bucket_states = [] # used to track if gradient is ready for each bucket

        # broadcast initial parameters
        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)
            # torch.cuda.synchronize()

    def __call__(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def ddp_bucketed_on_train_batch_start(self):
        if self.buckets:
            return

        self.buckets = get_buckets(self.module, self.bucket_size_mb)
        
        # For each bucket, register hook on the last parameter
        for bucket_id, bucket in enumerate(self.buckets):
            state = {
                "bucket_id": bucket_id,
                "params": bucket,
                "grad_counts": 0,
            }
            self.bucket_states.append(state)

            # Create a hook for each parameter in this bucket
            # but the all_reduce only run once when every gradient is ready in this bucket
            for param in bucket:
                def make_hook(bucket_id):
                    def hook(param):
                        # get the corresponding state from self
                        st = self.bucket_states[bucket_id]
                        bucket_params = st["params"]
                        # if grad for THIS param is ready, note it down
                        if param.grad is not None:
                            st["grad_counts"] += 1
                        
                        # if this param is the last one, ready for all_reduce
                        if st["grad_counts"] == len(bucket_params):
                            bucket_grads = [p.grad for p in bucket_params]
                            flat_grad = torch._utils._flatten_dense_tensors(bucket_grads)
                        
                            # 2. All-reduce the flattened gradient
                            handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG, async_op=True)
                        
                            # 3. Store handle and the info needed to unflatten later
                            self.handles.append((handle, flat_grad, bucket_id))
                    return hook
                param.register_post_accumulate_grad_hook(make_hook(bucket_id))

    def ddp_bucketed_on_after_backward(self):
        """Wait for all async all_reduce operations and copy back"""
        for handle, flat_grad, bucket_id in self.handles:
            bucket_params = self.bucket_states[bucket_id]["params"]
            # Wait for this bucket's all_reduce
            handle.wait()
            
            # Unflatten the all-reduced gradient
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, bucket_params)
            
            # Copy back to original parameters
            for param, grad in zip(bucket_params, unflat_grads):
                param.grad.copy_(grad)
        
            # restore this bucket's state
            self.bucket_states[bucket_id]["grad_counts"] = 0

        self.handles.clear()


    # to pass test.
    def __getattr__(self, name):
        return getattr(self.module, name)

def run_test(
    rank: int,
    world_size: int,
    dataset: np.ndarray, 
    module: BasicsTransformerLM, 
    bucket_size_mb: int,
    batch_size: int = 4, 
) -> tuple[float, float]:
    setup(rank, world_size, "nccl")

    # Move model to GPU BEFORE wrapping with DDP (for NCCL broadcast)
    module.to(get_device(rank, "nccl"))
    model = DDPBucketed(module, bucket_size_mb)
    model.ddp_bucketed_on_train_batch_start()

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
        model.ddp_bucketed_on_after_backward()
        optimizer.step()

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
    bucket_size_mb = 1.0
    mp.spawn(fn=run_test, args=(world_size, dataset, module, bucket_size_mb), nprocs=world_size, join=True)