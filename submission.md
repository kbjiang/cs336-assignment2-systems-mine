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
    1. To get the report, run `nsys profile --python-backtrace=cuda --cudabacktrace=all --pytorch=autograd-nvtx python your_script.py` 
        1. `backtrace` for..., `autograd-nvtx` for...
    1. To see by `nvtx` section
        1. CLI: `nsys stats --report cuda_gpu_trace --format table --filter-nvtx "Forward Pass"  report.nsys-rep`
        1. UI: `apply filter` -> `Stats System View` in bottom dropdown -> `CUDA GPU trace`or whatever.
1. See `report-toy.nsys-rep` for simpler profiling.
### Answers
1. Yes, pretty close.
1. `ampere_sgemm` takes most GPU time during forward pass; it is invoked 9 times during a single forward pass. It also takes most GPU time when both forward/backward passes are considered.
1. `elementwise_kernel` with `MulFunctor`, both vectorized or not, accounts for ~10% GPU time. 
1. skipped
1. FLOPs wise, `matmul` is $O(qdk)$ while `softmax` is $O(qk)$, the variables are sizes of $Q$, $K$ and model dimension; former is way more FLOPs; runtime wise, they are quite comparable at 1ms.
1. https://youtu.be/aQ1NYoRvp7o

## 1.1.5 Mixed precision
### Learnings
1. Two sources of error. 
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
1. Data Type Flow in Mixed Precision
    - **Parameters**: float32 (for precision)
    - **Forward activations**: float16 (for speed) 
        ```python
        # What happens inside autocast:
        x = torch.randn(4, 3, dtype=torch.float32).cuda()  # Input: float32
        weight = torch.randn(10, 3, dtype=torch.float32).cuda()  # Param: float32

        # During matrix multiplication, autocast converts both to float16:
        # result = torch.matmul(x.half(), weight.half())  # Both converted to float16
        # So the output is float16
        ```
    - **Gradients**: float32 (for preventing underflow and numerical stability)
    - **Optimizer updates**: float32 (for precision)
1. Where Speed Comes From
    - **GPU Tensor Cores**: 2-4x faster float16 matrix operations on modern GPUs
    - **Memory bandwidth**: Half the memory usage (2 bytes vs 4 bytes per element)
    - **Cache efficiency**: 2x more data fits in GPU cache
1. Key Insights
    - Autocast converts float32→float16→float32 during operations
    - Conversion overhead is minimal compared to compute speedup
    - Gradients stay float32 to prevent underflow and maintain stability
    - Mixed precision = speed of float16 + stability of float32

### Answers
#### mixed_precision_accumulation
1. As long as accumulation `s` is `float32`, the result is close. See [`Learnings`](#learnings-1) for more detail.
#### benchmarking_mixed_precision
1. data types
    - the model parameters within the autocast context: `float32`
    - the output of the first feed-forward layer (ToyModel.fc1): `float16`
    - the output of layer norm (ToyModel.ln): `float32`
    - the model's predicted logits: `float16`
    - the loss: `float32`
    - the model's gradients: `float32`
1. When `x` is large, `1/rms` in `float16` underflows, while `bfloat16` is fine. However, layer normalization is in `float32` in both cases. Why?
    ```python
    x = torch.tensor([1e3], dtype=torch.float16)
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True))
    print(rrms)
    x = torch.tensor([1e3], dtype=torch.bfloat16)
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True))
    print(rrms)
    # tensor([0.], dtype=torch.float16)
    # tensor([0.0010], dtype=torch.bfloat16)
    ```
1. 
