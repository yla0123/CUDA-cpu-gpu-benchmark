# Vector Addition Benchmark Results

## Test Configuration

The benchmark compares a CPU implementation of vector addition against a CUDA GPU implementation.

The operation is:

```cpp
c[i] = a[i] + b[i];
```

The project was built in Release mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Hardware and software:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Environment: WSL2 Ubuntu
CUDA Toolkit: 13.4
```

Each problem size was executed four times.

The median is used as the main reported result to reduce the influence of run-to-run timing variation.

---

## N = 100,000

| Run | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) |
|---:|---:|---:|---:|
| 1 | 0.032361 | 0.176928 | 4.338214 |
| 2 | 0.022570 | 0.073376 | 3.429140 |
| 3 | 0.028700 | 0.045440 | 3.353541 |
| 4 | 0.027150 | 0.139712 | 4.001116 |
| **Median** | **0.027925** | **0.106544** | **3.715128** |

At this size, the CPU implementation is faster than both the CUDA kernel and the complete GPU execution path.

The workload is small enough that GPU parallelism does not compensate for the additional GPU execution overhead.

---

## N = 1,000,000

| Run | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) |
|---:|---:|---:|---:|
| 1 | 0.458551 | 0.187200 | 10.125072 |
| 2 | 1.826623 | 0.080096 | 8.456853 |
| 3 | 1.624575 | 0.061376 | 9.353392 |
| 4 | 0.846371 | 0.055168 | 10.285398 |
| **Median** | **1.235473** | **0.070736** | **9.739232** |

At one million elements, the GPU kernel becomes substantially faster than the CPU loop.

However, the complete GPU workflow remains slower because memory allocation, memory transfers, and other CUDA runtime operations dominate the total execution time.

---

## N = 10,000,000

| Run | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) |
|---:|---:|---:|---:|
| 1 | 4.662082 | 0.904352 | 23.729047 |
| 2 | 4.576930 | 0.897344 | 21.686741 |
| 3 | 5.288737 | 0.875840 | 24.373316 |
| 4 | 4.964858 | 0.856672 | 21.536104 |
| **Median** | **4.813470** | **0.886592** | **22.707894** |

At ten million elements, the CUDA kernel is approximately five times faster than the CPU implementation:

```text
CPU median:        4.81 ms
GPU kernel median: 0.89 ms
```

However, the end-to-end GPU time is approximately:

```text
22.71 ms
```

This shows that the kernel itself is not the main performance bottleneck.

---

## Summary

| Vector Size | CPU Median | GPU Kernel Median | GPU End-to-End Median |
|---:|---:|---:|---:|
| 100,000 | 0.028 ms | 0.107 ms | 3.72 ms |
| 1,000,000 | 1.24 ms | 0.071 ms | 9.74 ms |
| 10,000,000 | 4.81 ms | 0.887 ms | 22.71 ms |

The benchmark shows two different performance effects.

First, increasing the workload allows the GPU kernel to take advantage of parallel execution. At 100,000 elements, the CPU is faster than the kernel, while at one million and ten million elements the CUDA kernel is faster.

Second, kernel performance alone does not determine application performance.

The GPU end-to-end measurement includes approximately:

```text
cudaMalloc
+
host-to-device cudaMemcpy
+
kernel execution
+
device-to-host cudaMemcpy
+
cudaFree
```

For this simple vector-addition workload, these additional costs are larger than the computation itself.

Vector addition has low arithmetic intensity because only one arithmetic operation is performed for each output element while multiple values must be moved through memory.

Therefore, copying the input to the GPU, performing only one addition per element, and immediately copying the result back is not an efficient use of the GPU.

A more realistic GPU application would often keep data resident on the device and perform multiple kernels before transferring the final result back to the CPU.

## Main Observation

The experiment demonstrates the difference between:

```text
fast CUDA kernel
```

and:

```text
fast GPU application
```

The CUDA kernel can outperform the CPU while the overall GPU implementation remains slower.

This is why both kernel-level measurements and end-to-end measurements are important when evaluating GPU acceleration.