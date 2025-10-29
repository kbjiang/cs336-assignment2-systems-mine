from __future__ import annotations
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, List, Tuple

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

# Import optimizer sharding utilities
from cs336_systems.sharded_optimizer import OptimizerSharded, setup, get_device
from typing import Iterable
import pandas as pd

###############################################################################
# Helpers
###############################################################################
def tensor_nbytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()

def model_parameter_bytes(parameters: Iterable[torch.Tensor]) -> int:
    return sum(tensor_nbytes(p) for p in parameters)

def gradient_bytes(parameters: Iterable[torch.Tensor]) -> int:
    return sum(tensor_nbytes(p.grad) for p in parameters if p.grad is not None)

def adamw_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    # AdamW stores 'exp_avg' and 'exp_avg_sq' per param typically
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                total += tensor_nbytes(v)
    return total

def global_max_int(value: int, device: torch.device) -> int:
    """All-reduce MAX for an integer value across ranks."""
    t = torch.tensor([value], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())

def reset_and_measure(label: str, device: torch.device) -> Dict[str, int]:
    """Synchronize then capture current & peak memory (bytes) since last reset."""
    torch.cuda.synchronize(device)
    current = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    # global max across ranks
    current_g = global_max_int(current, device)
    peak_g = global_max_int(peak, device)
    return {"phase": label, "current_bytes_global_max": current_g, "peak_bytes_global_max": peak_g}

def bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 4)

###############################################################################
# Scenario runner
###############################################################################

def run_scenario(
        model_ctor, dataset: np.ndarray, device: torch.device, batch_size: int, sharded: bool,
        warmup_steps: int, train_steps: int,
    ) -> tuple[pd.DataFrame, Dict[str, object]]:
    # Fresh model per scenario to avoid residual optimizer state or allocated buffers
    model = model_ctor().to(device)
    metrics: List[Dict[str, object]] = []

    # Phase 1: After model init
    torch.cuda.reset_peak_memory_stats(device)
    metrics.append(reset_and_measure("after_model_init", device))

    # Construct optimizer (sharded or unsharded)
    if sharded:
        opt = OptimizerSharded(
            params=model.parameters(),
            optimizer_cls=torch.optim.AdamW,
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )


    # Optional warmup steps (allocate optimizer states, trigger JIT, stabilize allocator)
    context_len = getattr(model, "context_length")
    for _ in range(warmup_steps):
        x_w, y_w = get_batch(dataset, batch_size, context_len, device)
        y_hat_w = model(x_w)
        loss_w = cross_entropy(y_hat_w, y_w)
        opt.zero_grad()
        loss_w.backward()
        opt.step()

    # For unsharded AdamW: estimated state bytes = 2 * param_bytes (exp_avg & exp_avg_sq) same dtype as params
    # For sharded: only local shard states present; estimate actual via reading optimizer state dict
    # Logical breakdown (estimates & actual) captured after optimizer step
    param_bytes = model_parameter_bytes(model.parameters())
    grad_bytes = gradient_bytes(model.parameters())  # grads might be cleared or kept; after step some optimizers clear; this is for reference

    breakdown = {
        "sharded": sharded,
        "param_GB": bytes_to_gb(param_bytes),
        "grad_GB": bytes_to_gb(grad_bytes),
        "optimizer_state_actual_GB_series": [],
        "optimizer_state_estimated_GB_series": [],
    }
    iteration_times: List[float] = []

    for _ in range(train_steps):
        # Measured training step
        x, y = get_batch(dataset, batch_size, context_len, device)
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)
        opt.zero_grad()

        # Reset peak at start of measured step to isolate forward+backward post-warmup
        torch.cuda.reset_peak_memory_stats(device)
        iter_start = time.time()
        loss.backward()
        metrics.append(reset_and_measure("after backward/before_optimizer_step", device))

        opt.step()
        iteration_times.append(time.time() - iter_start)

        torch.cuda.reset_peak_memory_stats(device)
        metrics.append(reset_and_measure("after_optimizer_step", device))

        if sharded:
            # Logical breakdown (estimates & actual) captured after optimizer step
            actual_opt_state_bytes = adamw_state_bytes(opt._optimizer)
            estimated_state_bytes = 2 * model_parameter_bytes(opt._params)  # local shard only
        else:
            actual_opt_state_bytes = adamw_state_bytes(opt)
            estimated_state_bytes = 2 * param_bytes  # logical global

        breakdown["optimizer_state_actual_GB_series"].append(bytes_to_gb(actual_opt_state_bytes))
        breakdown["optimizer_state_estimated_GB_series"].append(bytes_to_gb(estimated_state_bytes))

    # Compute mean optimizer state sizes across measured train steps
    if breakdown["optimizer_state_actual_GB_series"]:
        actual_series = breakdown["optimizer_state_actual_GB_series"]
        est_series = breakdown["optimizer_state_estimated_GB_series"]
        breakdown["optimizer_state_actual_GB_mean"] = round(sum(actual_series) / len(actual_series), 4)
        breakdown["optimizer_state_estimated_GB_mean"] = round(sum(est_series) / len(est_series), 4)
    if iteration_times:
        breakdown["iteration_time_series_sec"] = [round(t, 6) for t in iteration_times]
        breakdown["iteration_time_mean_sec"] = round(sum(iteration_times) / len(iteration_times), 6)
        # Standard deviation (population) of iteration times
        mean_t = breakdown["iteration_time_mean_sec"]
        var_t = sum((t - mean_t) ** 2 for t in iteration_times) / len(iteration_times)
        breakdown["iteration_time_std_sec"] = round(var_t ** 0.5, 6)


    # Convert to DataFrame for aggregation
    metrics_df = pd.DataFrame(metrics)

    # Aggregate per phase (mean current & peak) across repeated steps
    phase_means = []
    if not metrics_df.empty:
        agg = metrics_df.groupby("phase").agg({
            "current_bytes_global_max": "mean",
            "peak_bytes_global_max": "mean",
        }).reset_index()
        for _, row in agg.iterrows():
            phase_means.append({
                "phase": row["phase"],
                "current_GB_mean": bytes_to_gb(int(row["current_bytes_global_max"])),
                "peak_GB_mean": bytes_to_gb(int(row["peak_bytes_global_max"])),
            })
    breakdown["phase_means"] = phase_means

    return metrics_df, breakdown

###############################################################################
# Main per-rank worker
###############################################################################

def worker(rank: int, world_size: int, batch_size: int, warmup_steps: int, train_steps: int):
    setup(rank, world_size, backend="nccl")
    device = torch.device(get_device(rank, "nccl"))

    # Model constructor closure so we can rebuild identical models per scenario
    def model_ctor():
        return BasicsTransformerLM(
            vocab_size=10_000,
            context_length=512,
            d_model=1600,
            d_ff=6400,
            num_layers=48,
            num_heads=25,
            rope_theta=10_000,
        )

    # Synthetic dataset
    dataset = np.random.randint(0, 10_000, size=4096)

    # Run both scenarios
    unsharded_metrics_df, unsharded_breakdown = run_scenario(
        model_ctor, dataset, device, batch_size, sharded=False,
        warmup_steps=warmup_steps, train_steps=train_steps,
    )
    # Barrier to avoid overlapping allocations influencing next scenario peak
    dist.barrier()
    sharded_metrics_df, sharded_breakdown = run_scenario(
        model_ctor, dataset, device, batch_size, sharded=True,
        warmup_steps=warmup_steps, train_steps=train_steps,
    )

    # Rank 0 aggregates & prints summary
    if rank == 0:
        def render(metrics_df: pd.DataFrame, breakdown: Dict[str, object], tag: str) -> Dict[str, object]:
            raw_phase_rows = []
            for _, row in metrics_df.iterrows():
                raw_phase_rows.append({
                    "phase": row["phase"],
                    "current_GB": bytes_to_gb(int(row["current_bytes_global_max"])),
                    "peak_GB": bytes_to_gb(int(row["peak_bytes_global_max"])),
                })
            return {
                "scenario": tag,
                # "raw_phases": raw_phase_rows,
                "phase_means": breakdown.get("phase_means", []),
                "breakdown": {k: v for k, v in breakdown.items() if k not in {"phase_means"}},
            }

        report = {
            "world_size": world_size,
            "results": [
                render(unsharded_metrics_df, unsharded_breakdown, "unsharded"),
                render(sharded_metrics_df, sharded_breakdown, "sharded"),
            ],
        }
        print(json.dumps(report, indent=2))

    dist.destroy_process_group()

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sharded optimizer memory & timing profiling")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps before measurement loop")
    parser.add_argument("--train-steps", type=int, default=100, help="Measured training iterations")
    args = parser.parse_args()

    # Spawn one process per rank
    world_size = 2
    mp.spawn(
        worker,
        nprocs=world_size,
        args=(world_size, args.batch_size, args.warmup_steps, args.train_steps),
    )
