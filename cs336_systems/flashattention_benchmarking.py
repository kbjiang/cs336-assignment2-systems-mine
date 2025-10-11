import json
import torch
import triton
from cs336_systems.flashattention import AttentionPytorch as NoFlashTorch
from cs336_systems.flashattention_triton_autotune import FlashAttentionTritonAutotune as FlashTriton
# from cs336_systems.flashattention_triton_optimized import FlashAttentionTritonOptimized as FlashTriton
# from cs336_systems.flashattention_triton_backward_autotune import FlashAttentionTritonBackward as FlashTriton
# from cs336_systems.flashattention_triton_backward import FlashAttentionTritonBackward as FlashTriton
import itertools
from tqdm.auto import tqdm

IMPL_DICT = {
    "NoFlashTorch": NoFlashTorch, 
    "FlashTriton_": FlashTriton,
}

def benchmark_flash(
    test, impl_name, n_heads, d_head, sequence_length, dtype, device='cuda'
):
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device=device, dtype=dtype, requires_grad=True
    )
    
    # For autotuned version, use the function directly since tile sizes are auto-determined
    impl = IMPL_DICT[impl_name]
    flash = torch.compile(impl.apply)
    # sanity check; it would fail without compiling if precision in triton is not implemented right
    flash = impl.apply
    
    def flash_forward():
        o = flash(q, k, v, True)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    # Clear cache and reset peak memory stats before benchmarking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
        
    # Get initial memory state
    initial_memory = torch.cuda.memory_allocated(device)
        
    try:
        if test == "forward":
            time = triton.testing.do_bench(flash_forward, rep=10000, warmup=1000)
        elif test == "forward_backward":
            time = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
        else:
            raise ValueError("Wrong selection.")
            
        # Get peak memory usage during benchmarking
        peak_memory = torch.cuda.max_memory_allocated(device) - initial_memory
        peak_memory_mb = peak_memory / 1024 / 1024
        
    except torch.OutOfMemoryError:
        time = peak_memory_mb = None

    result = {
        "Impl": impl_name,
        "test": test, 
        "d_head": d_head,
        "seq_len": sequence_length,
        "dtype": str(dtype),
        "time": round(time, 3) if time is not None else None,
        "memory_MB": round(peak_memory_mb, 3) if peak_memory_mb is not None else None
    }

    return result


if __name__ == "__main__":
    output_file = "flashattention_benchmarking.jsonl"

    tests = ["forward", "forward_backward"]
    dtypes = [torch.bfloat16, torch.float32]
    n_heads = 1
    d_heads = [2**n for n in range(6, 8)]
    seq_lens = [2**n for n in range(12, 17)]
    
    # Convert to list so tqdm can automatically detect the total
    # all_combinations = list(itertools.product(d_heads, seq_lens, dtypes, tests, IMPL_DICT.keys()))
    all_combinations = list(itertools.product(d_heads, seq_lens, dtypes, ["forward_backward"], ["FlashTriton_"]))
    print(f"Total configurations to test: {len(all_combinations)}")
    
    # Clean up before starting the benchmark suite
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with open(output_file, "w") as f:
        for d_head, seq_len, dtype, test, impl_name in tqdm(all_combinations):
            # print(f"Testing {impl_name} - {test}: d_head={d_head}, seq_len={seq_len}, dtype={dtype}")
            
            # Call your benchmark function
            result = benchmark_flash(
                test, impl_name, 1, d_head, seq_len, dtype
            )
            
            # print(result)
            # Write as single line JSON (JSONL format - no indentation)
            json.dump(result, f)
            f.write('\n')
            f.flush()  # Ensure immediate write to disk

            # Clean up between iteration
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

