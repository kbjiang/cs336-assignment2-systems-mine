import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

import torch.cuda.nvtx as nvtx

###############################################
# Instrumented Bucketed Overlap Demo
# ----------------------------------
# This script adds NVTX ranges for:
#   - Entire backward pass
#   - Per-bucket readiness (gradient accumulation complete)
#   - Per-bucket all_reduce launch (async)
#   - Finalization phase (waiting + unflatten/avg)
# Use Nsight Systems (nsys) to visualize overlap:
#   nsys profile -t cuda,nvtx -o trace \
#       python cs336_systems/ddp_overlap_bucketed_instrumented.py --world-size 2 --bucket-mb 25
###############################################

def setup(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_buckets(model: torch.nn.Module, bucket_size_mb: float) -> list[list[torch.Tensor]]:
    elem_limit = int((1024 ** 2 * bucket_size_mb) // 4)  # float32 elements per bucket
    buckets: list[list[torch.Tensor]] = []
    cur_bucket: list[torch.Tensor] = []
    cur_size = 0
    for p in reversed(list(model.parameters())):  # reversed for output-side params earlier
        if not p.requires_grad:
            continue
        n = p.nelement()
        if n > elem_limit:  # large standalone bucket
            if cur_bucket:
                buckets.append(cur_bucket)
                cur_bucket = []
                cur_size = 0
            buckets.append([p])
        elif n + cur_size > elem_limit:  # overflow -> start new bucket
            buckets.append(cur_bucket)
            cur_bucket = [p]
            cur_size = n
        else:  # accumulate
            cur_bucket.append(p)
            cur_size += n
    if cur_bucket:
        buckets.append(cur_bucket)
    return buckets


class DDPBucketedInstrumented:
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets: list[list[torch.Tensor]] | None = None
        self.bucket_states = []  # dict per bucket
        self._global_seq = 0

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def init_buckets_and_hooks(self):
        if self.buckets is not None:
            return
        self.buckets = get_buckets(self.module, self.bucket_size_mb)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f"[Init] Created {len(self.buckets)} buckets (world_size={world_size})")

        for b_idx, bucket in enumerate(self.buckets):
            state = {
                "ready": 0,
                "params": bucket,
                "grads": [None] * len(bucket),
                "ready_seq": [-1] * len(bucket),
                "flat": None,
                "handle": None,
                "completed": False,
            }
            self.bucket_states.append(state)

            for p_idx, p in enumerate(bucket):
                def make_hook(bi=b_idx, pi=p_idx, pref=p):
                    def hook(_param_obj):
                        st = self.bucket_states[bi]
                        st["grads"][pi] = pref.grad
                        st["ready_seq"][pi] = self._global_seq
                        self._global_seq += 1
                        st["ready"] += 1
                        # If bucket completes
                        if st["ready"] == len(st["params"]) and not st["completed"]:
                            nvtx.range_push(f"bucket_{bi}_ready")
                            st["flat"] = torch._utils._flatten_dense_tensors(st["grads"]).detach()
                            nvtx.range_pop()  # readiness accounting done

                            # Launch async all_reduce if distributed initialized
                            nvtx.range_push(f"bucket_{bi}_allreduce_launch")
                            if dist.is_initialized():
                                st["handle"] = dist.all_reduce(st["flat"], async_op=True)
                            nvtx.range_pop()

                            st["completed"] = True
                    return hook
                p.register_post_accumulate_grad_hook(make_hook())

    def finalize(self):
        nvtx.range_push("finalize_buckets")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for bi, st in enumerate(self.bucket_states):
            if not st["completed"]:
                continue
            h = st["handle"]
            if h is not None:
                h.wait()
            flat = st["flat"]
            if flat is None:
                continue
            if world_size > 1:
                flat /= world_size
            # Unflatten and copy back
            unflat = torch._utils._unflatten_dense_tensors(flat, st["grads"])
            for p, new_grad in zip(st["params"], unflat):
                p.grad.copy_(new_grad)
        nvtx.range_pop()

    def __getattr__(self, name: str):  # delegate modules attributes
        return getattr(self.module, name)


def run(rank: int, world_size: int, args, shared_module: BasicsTransformerLM, dataset: np.ndarray):
    setup(rank, world_size, backend="nccl")
    # Use both a torch.device (for .to()) and a string (for helper utilities expecting substring checks)
    device = torch.device(f"cuda:{rank}")
    device_str = f"cuda:{rank}"  # data.py's get_batch expects a string it can substring match ('cuda' in device)
    shared_module.to(device)

    # IMPORTANT: parameters must be identical across ranks before training
    for p in shared_module.parameters():
        dist.broadcast(p.data, src=0)

    ddp = DDPBucketedInstrumented(shared_module, bucket_size_mb=args.bucket_mb)
    ddp.init_buckets_and_hooks()
    optimizer = AdamW(ddp.module.parameters(), lr=0.01)

    def step():
        # get_batch expects device string; passing torch.device caused TypeError ('in' not supported)
        x, y = get_batch(dataset, args.batch_size, ddp.module.context_length, device_str)
        optimizer.zero_grad()
        nvtx.range_push("forward")
        y_hat = ddp(x)
        nvtx.range_pop()
        nvtx.range_push("backward")
        loss = cross_entropy(y_hat, y)
        loss.backward()
        nvtx.range_pop()
        ddp.finalize()
        nvtx.range_push("optimizer_step")
        optimizer.step()
        nvtx.range_pop()
        torch.cuda.synchronize()  # for timing isolation

    # Warmup
    for _ in range(args.warmup):
        step()

    times = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    if rank == 0:
        print(f"[Result] world_size={world_size} bucket_mb={args.bucket_mb} avg_step={avg:.4f}s")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Instrumented bucketed gradient overlap demo")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--bucket-mb", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()

    # Shared model & data (model passed to each process)
    module = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=args.context,
        d_model=1600,
        d_ff=6400,
        num_layers=args.layers,
        num_heads=25,
        rope_theta=10_000,
    )
    np.random.seed(42)
    dataset = np.random.randint(0, 10_000, 50_000)  # synthetic token stream

    if args.world_size == 1:
        run(0, 1, args, module, dataset)
    else:
        mp.spawn(run, args=(args.world_size, args, module, dataset), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
