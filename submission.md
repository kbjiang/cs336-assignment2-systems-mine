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
### Answers
1. Yes, pretty close.
1. `ampere_sgemm` takes most GPU time during forward pass; it is invoked 9 times during a single forward pass. It also takes most GPU time when both forward/backward passes are considered.
1. `elementwise_kernel` with `MulFunctor`, both vectorized or not, accounts for ~10% GPU time. 
1. skipped
1. FLOPs wise, `matmul` is $O(qdk)$ while `softmax` is $O(qk)$, former is way more intense; runtime wise, they are quite comparable at 1ms.
> `nsys profile --python-backtrace=cuda --cudabacktrace=all python your_script.py`
> `nsys profile --python-sampling=true --python-sampling-frequency=1000 python benchmarking_script.py --d-model 1024 --d-ff 4096 --num-layers 24 --num-heads 16`
1. https://youtu.be/aQ1NYoRvp7o

## 1.1.5 Mixed precision
### Learnings
1. Two sources of error. See example in `mixed_precision_accumulation`.
    1. The limited precision of binary machine representing decimal number
        ```python
        x = torch.tensor(0.01, dtype=torch.float32)
        print(f"0.01 in float32: {x.item():.10f}") 
        # 0.01 in float32: 0.0099999998 
        ```
    1. The accumulation. See how each accumulation gets rounded to nearest `float16/32` and ends up with different values.  Overall full precision is closer to exact value. Thus "a good idea to keep accumulations in higher precision even if the tensors themselves being accumulated have been downcasted"
        ```python
        s1 = torch.tensor(0, dtype=torch.float16)
        s2 = torch.tensor(0, dtype=torch.float32)
        x = torch.tensor(0.01, dtype=torch.float16)
        print("float16, float32")
        for i in range(5):
            s1 += x
            s2 += x
            print(f"{s1.item():.10f}, {s2.item():.10f}")
        # float16, float32
        # 0.0100021362, 0.0100021362
        # 0.0200042725, 0.0200042725
        # 0.0299987793, 0.0300064087
        # 0.0400085449, 0.0400085449  # identical by accident
        # 0.0500183105, 0.0500106812
        ```

### Answers
#### mixed_precision_accumulation
1. As long as accumulation `s` is `float32`, the result is close. 