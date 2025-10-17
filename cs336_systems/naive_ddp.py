import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import itertools

class ToyModel(torch.nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.l1 = nn.Linear(d_h, d_h *4)
        self.l2 = nn.Linear(d_h * 4, d_h)
        self.relu = nn.ReLU()

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

def validate_ddp_net_equivalence(net):
    # Helper to validate synchronization of nets across ranks.
    # net_module_states = list(net.module.state_dict().values())
    net_module_states = list(net.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)


def naive_ddp_main(rank, backend, world_size, batch_size=32, d_h=64):
    setup(rank, world_size, backend)
    # -1. Initialize model on each rank
    model = ToyModel(d_h).to(get_device(rank, backend))
    model_sp = ToyModel(d_h).to(get_device(rank, backend))
    model_sp.load_state_dict(model.state_dict())

    # 0. broadcast model params from rank 0 to every other rank
    for param in model.parameters():
        dist.broadcast(tensor=param.data, src=0, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()

    # Test: initial params are close on rank 0, different on other ranks
    for param_sp, param in zip(model_sp.parameters(),  model.parameters()):
        if rank == 0:
            assert torch.allclose(param_sp, param)
        else:
            assert not torch.allclose(param_sp, param)
    
    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(model)

    # optimizer after params has been synced
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_sp = torch.optim.SGD(model_sp.parameters(), lr=lr)

    # 1. Shard and move data to the appropriate device
    local_batch_size = batch_size // world_size
    
    data = get_batch(batch_size, d_h).to(get_device(rank, backend))
    data_sp = data.clone()
    data = data[rank*local_batch_size:(rank+1)*local_batch_size, :]
    for _ in range(5):

        # 2. forward and backward to get gradient each rank
        output = model(data)
        loss = output.square().mean()
        optimizer.zero_grad()
        loss.backward()

        # Do I want to make sure all gradients have been calculated before `all_reduce`?
        # not sure, but the lecture code does not. So I am skipping it for now.
        # dist.barrier() 

        # 2.1 all-reduce to average the gradients and copy to each rank
        for param in model.parameters():
            if backend == "nccl":
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
                torch.cuda.synchronize()
            else:
                # Gloo doesn't support AVG, so use SUM and manually divide
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad /= world_size

        # 3. each rank do optimizer step and update its copy of gradient
        optimizer.step()

        # Test: before training non-parallel model, params should be different
        if rank == 0:
            for param_sp, param in zip(model_sp.parameters(), model.parameters()):
                if param_sp.requires_grad and param.requires_grad:
                    assert not torch.allclose(param_sp, param)
                else:
                    assert torch.allclose(param_sp, param)

        # Test: after training non-parallel model, params should be close
        output = model_sp(data_sp)
        loss = output.square().mean()
        optimizer_sp.zero_grad()
        loss.backward()
        optimizer_sp.step()

        if rank == 0:
            for param_sp, param in zip(model_sp.parameters(), model.parameters()):
                assert torch.allclose(param_sp, param)

    # Cleanup to avoid warning
    dist.destroy_process_group()  


if __name__ == "__main__":
    # backends = ["gloo", "nccl"]
    backend = "nccl"
    world_size = 2
    mp.spawn(fn=naive_ddp_main, args=(backend, world_size), nprocs=world_size, join=True)