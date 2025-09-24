#!/usr/bin/env python3
# filepath: /root/02-fun/assignment2-systems/cs336_systems/run_benchmark.py
import itertools
import subprocess
import sys
import os
import json
from datetime import datetime

def get_parameter_combinations():
    """Generate all parameter combinations"""
    model_configs = [
        {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        # {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        # {"name": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        # {"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    ]
    
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 256
    warmup_options = [False, True]  # warmup-unaware
    backward_options = [False, True]  # skip-backward
    
    combinations = []
    for config, warmup_unaware, skip_backward in itertools.product(
        model_configs, warmup_options, backward_options
    ):
        combinations.append({
            "d_model": config["d_model"],
            "d_ff": config["d_ff"], 
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "warmup_unaware": warmup_unaware,
            "skip_backward": skip_backward,
            "model_name": config["name"]
        })
    
    return combinations

def run_single_benchmark(params, task_id, total):
    """Run a single benchmark with given parameters"""
    cmd = [
        'python', 'cs336_systems/benchmarking_script.py',
        '--d-model', str(params['d_model']),
        '--d-ff', str(params['d_ff']),
        '--num-layers', str(params['num_layers']),
        '--num-heads', str(params['num_heads']),
        '--batch-size', str(params['batch_size']),
        '--context-length', str(params['context_length']),
        '--train-steps', '10',
        '--warmup-steps', '5'
    ]
    
    if params['warmup_unaware']:
        cmd.append('--warmup-unaware')
    if params['skip_backward']:
        cmd.append('--skip-backward')
    
    print(f"[{task_id+1}/{total}] Running: {params['model_name']} "
          f"(warmup_unaware={params['warmup_unaware']}, skip_backward={params['skip_backward']})")
    
    # Run benchmark
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr}")
        return None
    
    print(f"  ✓ Completed successfully")
    return result.stdout.strip()

def run_all_benchmarks():
    """Run all benchmark combinations"""
    print("=== Transformer Benchmark Suite ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    combinations = get_parameter_combinations()
    print(f"Total combinations to run: {len(combinations)}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    results_file = 'results/benchmark_results.jsonl'
    
    # Clear previous results
    if os.path.exists(results_file):
        print(f"Clearing previous results from {results_file}")
        open(results_file, 'w').close()
    
    successful = 0
    failed = 0
    
    for i, params in enumerate(combinations):
        result = run_single_benchmark(params, i, len(combinations))
        
        if result:
            # Append to results file
            with open(results_file, 'a') as f:
                f.write(result + '\n')
            successful += 1
        else:
            failed += 1
        
        print()  # Empty line for readability
    
    print("=== Benchmark Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    if len(sys.argv) == 1:
        # Run all benchmarks
        run_all_benchmarks()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--list":
            # List all combinations
            combinations = get_parameter_combinations()
            print(f"Total combinations: {len(combinations)}")
            for i, combo in enumerate(combinations):
                print(f"{i}: {combo['model_name']} (warmup_unaware={combo['warmup_unaware']}, skip_backward={combo['skip_backward']})")
        elif sys.argv[1] == "--count":
            # Just print count
            combinations = get_parameter_combinations()
            print(len(combinations))
        else:
            # Run single benchmark (backwards compatibility)
            try:
                task_id = int(sys.argv[1])
                combinations = get_parameter_combinations()
                
                if task_id >= len(combinations):
                    print(f'Task ID {task_id} exceeds number of combinations ({len(combinations)})')
                    sys.exit(1)
                
                params = combinations[task_id]
                result = run_single_benchmark(params, task_id, len(combinations))
                
                if result:
                    os.makedirs('results', exist_ok=True)
                    with open('results/benchmark_results.jsonl', 'a') as f:
                        f.write(result + '\n')
                else:
                    sys.exit(1)
                    
            except ValueError:
                print("Usage:")
                print("  python run_benchmarks.py           # Run all benchmarks")
                print("  python run_benchmarks.py --list    # List all combinations")
                print("  python run_benchmarks.py --count   # Show total count")
                print("  python run_benchmarks.py <id>      # Run single benchmark")
                sys.exit(1)
    else:
        print("Too many arguments!")
        sys.exit(1)

if __name__ == '__main__':
    main()