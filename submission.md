## 1.1.3 End-to-End Benchmarking
### Learnings
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
    1. To get the report, run `nsys profile --python-backtrace=cuda --cudabacktrace=all --pytorch=autograd-nvtx python your_script.py --warmup --backward --mixed-precision` 
        1. `backtrace` for..., `autograd-nvtx` for...; both are optional if only time is needed.
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
1. `torch.autocast`
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
1. Why Backward Pass Speeds Up Despite Being Outside Autocast
    1. **Stored Activations Are Lower Precision**: During forward pass with autocast, intermediate activations are stored in bfloat16. When backpropagation runs, these stored activations are loaded from memory using half the bandwidth.
    2. **Memory Bandwidth Benefits**: The reduced memory footprint of activations (2 bytes vs 4 bytes per element) means faster data movement during gradient computation.
    3. **Tensor Core Utilization**: Many gradient computation steps involve matrix operations with the stored bfloat16 activations, which can still leverage GPU Tensor Cores.
    4. **Cache Efficiency**: More activation data fits in GPU caches, reducing memory access latency during backprop.
1. `nullcontext` is useful for conditionally enabling/disabling other context managers
    ```python
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_mixed_precision else nullcontext()
    with context:
        ...
    ```

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
1. Run benchmarking [script](./cs336_systems/benchmarking.sh) and get result. 
    ```csv
    size,precision,forward_runs,forward_time_seconds,backward_runs,backward_time_seconds
    small,full,10,0.317784,10,0.733286
    small,mixed,10,0.486204,10,0.918253
    medium,full,10,0.655829,10,1.690529
    medium,mixed,10,0.962006,10,1.769694
    large,full,10,0.948486,10,4.092338
    large,mixed,10,1.330271,10,2.244423
    xl,full,10,1.766273,10,8.497722
    xl,mixed,10,1.509672,10,3.65094
    2.7b,full,10,1.837926,10,13.481286
    2.7b,mixed,10,0.932135,10,4.093555
    ```
    1. Only when model is big enough do the `mixed-precision` show benefits.
    1. Forward wise, `2.7b-mixed` is faster than `xl-mixed`. Why?
1. 

## 1.1.6 Profiling Memory
### Learnings
1. Understand `active Memory Timeline`
    1. Each entry is to be understand as the resultant *tensor* of some operation. Each has a start and an end (releases memory)
        1. E.g., `output = self.lm_head(attn_output)` the memory of `output` should be `batch_size * seq_len * vocab_size * 4 bytes`.
    1. Memories that persist are probably model weights? But why did it not change with `context_length`?
    1. To start, blow up `memory_256_forward.pickle` and see each entry. You can see the shape of the blocks and details within.
1. Observations of the memory plot
    1. zoom in to see the multiple `stairs` of memory, each corresponds to one Transformer block.
    1. `ffn` requires a lot of memory comparing to `attn`
### Answers
#### memory_profiling
1. See the `memory_*pickle` files. 
1. Table with full precision.
    | Context Length | Forward | Full |
    |:--------------:|:--------------:|:-----------:|
    | 128 | 19.3GB | 60.0GB |
    | 256 | 26.0GB | 67.0GB |
    | 512 | 42.8GB | - |
1. Table with mixed precision. The memory profile with mixed precision looked more fragmented. Why?
    | Context Length | Forward | Full |
    |:--------------:|:--------------:|:-----------:|
    | 128 | 24.0GB | 65.0GB |
    | 256 | 28.3GB | 69.0GB |
    | 512 | 39.6GB | - |
1. 5, 10, 20 MBs for context lengths 128, 256, 512 respectively.
    1. Corresponding line: `cs336_basics/model.py:385:forward`, which is `ffn_sublayer_output = attn_sublayer_output + x_ffn`.
    1. 4 seqs/batch x 128 toks/seq x 2560 floats/tok x 4 bytes/float = 5 MB
1. The largest allocations are those corresponds to `FFN`.


## 1.2.1 Benchmarking PyTorch Attention
### Learnings
### Ansers
#### pytorch_attention
1. Timing 100 forward/backward. 
    |    |   d_model |   seq_len |   batch_size | backward   |   forward_time |   backward_time |   total_time | status   |
    |---:|----------:|----------:|-------------:|:-----------|---------------:|----------------:|-------------:|:---------|
    |  0 |        16 |       256 |            8 | True       |         0.1012 |          0.1371 |       0.2383 | success  |
    |  1 |        16 |      1024 |            8 | True       |         0.2941 |          0.4229 |       0.7171 | success  |
    |  2 |        16 |      4096 |            8 | True       |         1.5719 |          1.5144 |       3.0863 | success  |
    |  3 |        16 |      8192 |            8 | True       |         4.4004 |          2.9757 |       7.3761 | success  |
    |  4 |        16 |     16384 |            8 | True       |        13.9563 |          5.8925 |      19.8488 | success  |
    |  5 |        32 |       256 |            8 | True       |         0.101  |          0.1372 |       0.2382 | success  |
    |  6 |        32 |      1024 |            8 | True       |         0.3067 |          0.427  |       0.7337 | success  |
    |  7 |        32 |      4096 |            8 | True       |         1.641  |          1.5514 |       3.1924 | success  |
    |  8 |        32 |      8192 |            8 | True       |         4.5796 |          2.958  |       7.5375 | success  |
    |  9 |        32 |     16384 |            8 | True       |        14.4986 |          5.8994 |      20.398  | success  |
    | 10 |        64 |       256 |            8 | True       |         0.1123 |          0.1452 |       0.2575 | success  |
    | 11 |        64 |      1024 |            8 | True       |         0.3375 |          0.441  |       0.7784 | success  |
    | 12 |        64 |      4096 |            8 | True       |         1.8024 |          1.6438 |       3.4462 | success  |
    | 13 |        64 |      8192 |            8 | True       |         4.9855 |          3.2003 |       8.1858 | success  |
    | 14 |        64 |     16384 |            8 | True       |        15.6728 |          6.3461 |      22.0189 | success  |
    | 15 |       128 |       256 |            8 | True       |         0.1377 |          0.1716 |       0.3093 | success  |
    | 16 |       128 |      1024 |            8 | True       |         0.404  |          0.518  |       0.922  | success  |
    | 17 |       128 |      4096 |            8 | True       |         2.1052 |          1.8568 |       3.9619 | success  |
    | 18 |       128 |      8192 |            8 | True       |         5.7687 |          3.6539 |       9.4226 | success  |
    | 19 |       128 |     16384 |            8 | True       |        17.9237 |          7.2563 |      25.1799 | success  |
1. Run `pytorch pytorch_attention.py --memory-profiling --train-steps 1 --d-model 128 --seq-len 16384 --backward --compile --memory-profile-name something.pickle` to get the memory profile.
    1. the `attention_scores` requires big chunk of memory when `seq-len` is big; long sequence is the bottleneck.
    1. the memory saved for backward seems to be linear with sequence length.