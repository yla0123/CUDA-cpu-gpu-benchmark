# CUDA Profiling Results

The `vector_add` benchmark was profiled using both **NVIDIA Nsight Systems** and **NVIDIA Nsight Compute** with:

```text
N = 10,000,000
```

The two tools were used for different purposes:

```text
Nsight Systems → Where is the application spending its time?

Nsight Compute → What is happening inside the CUDA kernel?
```

Together, they provide both application-level and kernel-level views of the CUDA implementation.

---

## Nsight Systems

Nsight Systems was used to analyze the complete CUDA application timeline, including:

- CUDA API calls
- kernel execution
- host-to-device transfers
- device-to-host transfers
- synchronization
- memory allocation and deallocation

The profile was generated using:

```bash
nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    -o results/vector_add_nsys \
    ./build/vector_add
```

The terminal summary was generated with:

```bash
nsys stats results/vector_add_nsys.nsys-rep
```

### Memory Transfers

For `N = 10,000,000`, each vector contains 10 million `int` values.

Since each `int` occupies 4 bytes:

```text
10,000,000 × 4 bytes = 40 MB
```

The application transfers:

```text
A: Host → Device = 40 MB
B: Host → Device = 40 MB
C: Device → Host = 40 MB
```

Nsight Systems reported:

| Operation | Transfers | Data | GPU Time |
|---|---:|---:|---:|
| Host-to-Device | 2 | 80 MB | ~9.91 ms |
| Device-to-Host | 1 | 40 MB | ~6.57 ms |
| **Total** | **3** | **120 MB** | **~16.48 ms** |

This shows that approximately **16.5 ms** of GPU activity was spent moving data between host and device.

### Kernel Execution

Nsight Systems recorded two executions of `vectorAdd`.

This is expected because the program performs:

```text
1 warm-up kernel
+
1 benchmark kernel
```

The real benchmark kernel took approximately:

```text
~0.79 ms
```

which is close to the normal CUDA-event measurements of approximately:

```text
0.86–0.90 ms
```

### CUDA Initialization

Nsight Systems recorded two calls to:

```cpp
cudaDeviceSynchronize()
```

One call took approximately:

```text
0.062 ms
```

while the other took approximately:

```text
523 ms
```

The long synchronization occurred during the warm-up path and included one-time CUDA runtime and kernel initialization costs.

Because the benchmark performs the warm-up before starting the end-to-end timer, this initialization cost is intentionally excluded from the steady-state benchmark results.

This demonstrates why separating **cold-start latency** from **steady-state execution time** is important when benchmarking CUDA applications.

### Allocation and Runtime Overhead

Nsight Systems also showed measurable time spent in CUDA runtime operations such as:

```text
cudaMalloc
cudaFree
cudaMemcpy
cudaLaunchKernel
cudaDeviceSynchronize
```

Although these operations are necessary, repeatedly allocating memory and transferring data can become expensive compared with the actual computation.

### Nsight Systems Conclusion

The most important result from Nsight Systems is:

```text
vectorAdd kernel:       ~0.8 ms
host-device transfers: ~16.5 ms
```

The kernel itself is therefore not the main bottleneck.

The majority of the GPU workflow is spent moving data rather than performing vector addition.

---

## Nsight Compute

Nsight Compute was used to inspect the `vectorAdd` kernel in more detail.

The profile was generated using:

```bash
ncu --set basic ./build/vector_add
```

Nsight Compute profiled two kernel launches:

```text
warm-up kernel
benchmark kernel
```

The warm-up kernel used:

```text
Grid size:  1
Block size: 1
```

and is not representative of the real workload.

The following analysis therefore focuses on the actual benchmark kernel.

### Kernel Launch Configuration

For `N = 10,000,000`:

```text
Block size:        256 threads
Grid size:         39,063 blocks
Total threads:     10,000,128
Registers/thread:  16
Shared memory:     0 bytes
SM count:          20
```

The total thread count is slightly larger than the number of vector elements because the number of blocks is calculated using ceiling division.

### Kernel Performance

Representative Nsight Compute measurements were:

| Metric | Value |
|---|---:|
| Kernel Duration | ~0.794 ms |
| DRAM Throughput | ~44.7% |
| Compute Throughput | ~7.8% |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | ~82% |

The measured kernel duration agrees well with the normal benchmark measurements, providing additional confidence that the CUDA-event timing is reasonable.

### Compute vs Memory Utilization

Nsight Compute reported approximately:

```text
Compute throughput: ~7.8%
DRAM throughput:   ~44.7%
```

The much lower compute utilization supports the conclusion that this workload is primarily **memory-oriented rather than compute-intensive**.

---

## Combined Profiling Analysis

The two profiling tools reveal different parts of the same performance problem.

### Kernel Level

Nsight Compute shows:

```text
vectorAdd kernel ≈ 0.8 ms
```

The kernel achieves approximately:

```text
82% occupancy
44.7% DRAM throughput
7.8% compute throughput
```

The low compute utilization is expected because vector addition performs very little arithmetic.

### Application Level

Nsight Systems shows:

```text
Host → Device transfers: ~9.91 ms
Device → Host transfer:  ~6.57 ms

Total transfer time:    ~16.48 ms
```

The memory-transfer cost is therefore more than an order of magnitude larger than the actual vector-add computation.

The normal benchmark results also showed:

```text
CPU median:              ~4.81 ms
GPU kernel median:       ~0.89 ms
GPU end-to-end median:  ~22.71 ms
```

The CUDA kernel itself is substantially faster than the CPU implementation.

However:

```text
fast GPU kernel
        ≠
fast GPU application
```

The complete GPU implementation is slower because a large amount of data must be moved between CPU and GPU to perform only one addition per element.

---

## Optimization Implications

The profiling results suggest that further optimization of the `vectorAdd` kernel itself would have limited impact on total application performance.

More meaningful optimization directions would therefore include:

- keeping data resident on the GPU across multiple kernels
- reducing unnecessary host-device transfers
- reusing allocated device memory
- avoiding repeated `cudaMalloc` and `cudaFree`
- combining multiple GPU operations before copying results back
- experimenting with pinned host memory
- using asynchronous memory transfers
- using CUDA streams to overlap communication and computation where appropriate
