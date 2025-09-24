## 1.1.3 End-to-End Benchmarking
### Answers
1. see `cs336_systems/benchmarking_script.py` 
1. No significant deviation 
1. Without warmup the variation is significant. see `results/benchmark_results.jsonl` for full output

## 1.1.4 Nsight Systems Profiler
### Learnings
1. The versions of `nsys` need to match between Windows and Linux VM.
1. To install on Linux
    1. Download `.run` file from https://developer.nvidia.com/tools-downloads
    1. run and `export PATH=$PATH:$HOME/nsight-systems/bin`
1. `nvtx` is very helpful to isolate kernels for different parts of the calculation. To see operations by NVTX phase, 
    1. CLI: `nsys stats --report cuda_gpu_trace --format table --filter-nvtx "Forward Pass"  report.nsys-rep`
    1. UI: `apply filter` -> `Stats System View` in bottom dropdown -> `CUDA GPU trace`or whatever.
1. To see
### Answers
`nsys profile --python-backtrace=cuda --cudabacktrace=all python your_script.py`
> `nsys profile --python-sampling=true --python-sampling-frequency=1000 python benchmarking_script.py --d-model 1024 --d-ff 4096 --num-layers 24 --num-heads 16`
1. Yes, pretty close.
1. With `CUDA GPU Kernel Summary`, `matmul` took most time.
    ```
    Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
    69.7%	93.457 ms	146	640.117 μs	296.692 μs	280.117 μs	1.129 ms	404.363 μs	ampere_sgemm_128x64_tn
    ```
1. https://youtu.be/aQ1NYoRvp7o