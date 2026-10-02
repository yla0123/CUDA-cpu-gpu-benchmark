# CUDA C++ CPU vs GPU Benchmark

A small CUDA C++ project for learning GPU programming fundamentals, CPU-vs-GPU benchmarking, memory-transfer overhead, and basic CUDA profiling.

The goal of this project is not to build highly optimized CUDA kernels, but to understand how CUDA programs execute and how to reason about GPU performance using measurements and profiling tools.

## Project Structure

```text
cuda-cpu-gpu-benchmark/
├── src/
│   ├── vector_add.cu
│   ├── vector_scale.cu
│   └── matrix_mul.cu
├── results/
│   └── benchmark_results.md
│   └── profiling_notes.md
├── CMakeLists.txt
├── README.md
└── .gitignore
```

## Examples

### Vector Addition

The main benchmark compares CPU and CUDA implementations of:

```cpp
c[i] = a[i] + b[i];
```

Each GPU thread computes one vector element.

The program measures:

- CPU execution time
- GPU kernel execution time
- GPU end-to-end execution time
- correctness of the CPU and GPU results

A CUDA warm-up is performed before timing so that one-time CUDA initialization overhead is excluded from the steady-state benchmark.

### Vector Scaling

Demonstrates the standard CUDA one-thread-per-element execution pattern.


### Matrix Multiplication

Used a two-dimensional CUDA grid for this example.

## Environment

Tested using:

- Windows 11
- WSL2 Ubuntu
- NVIDIA GeForce RTX 4050 Laptop GPU
- CUDA Toolkit 13.4
- C++17
- CMake
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute

## Build

From the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the programs with:

```bash
./build/vector_add
./build/vector_scale
./build/matrix_mul
```

## Benchmark Results

The vector addition benchmark was tested using three input sizes.

| Vector Size | CPU Time | GPU Kernel Time | GPU End-to-End Time |
|---:|---:|---:|---:|
| 100,000 | 0.028 ms | 0.107 ms | 3.72 ms |
| 1,000,000 | 1.24 ms | 0.071 ms | 9.74 ms |
| 10,000,000 | 4.81 ms | 0.887 ms | 22.71 ms |

The raw measurements are available in [`results/benchmark_results.md`](results/benchmark_results.md).

For the smallest workload, the CPU implementation is faster even when compared only with GPU kernel execution.

For the larger workloads, the CUDA kernel itself becomes faster than the CPU implementation. However, GPU end-to-end execution remains slower because the complete GPU workflow also includes:

```text
device allocation
+
host-to-device transfer
+
kernel execution
+
device-to-host transfer
+
device cleanup
```

At 10 million elements, for example:

```text
CPU:             ~4.81 ms
GPU kernel:      ~0.89 ms
GPU end-to-end: ~22.71 ms
```

The kernel is considerably faster than the CPU loop, but moving data between host and device dominates the total cost.

Vector addition has low arithmetic intensity: each element requires two input reads, one addition, and one output write. There is therefore relatively little computation available to compensate for memory-transfer and CUDA runtime overhead.

This experiment demonstrates an important CUDA performance principle:

> A faster GPU kernel does not necessarily mean a faster application.

## Profiling

### Nsight Systems

Nsight Systems is used to analyze the complete CUDA application timeline.

It helps identify time spent in:
- CUDA API calls
- memory transfers
- kernel execution
- synchronization
- allocation and deallocation

### Nsight Compute

Nsight Compute is used to analyze individual CUDA kernels in more detail.

It provides information such as:
- kernel duration
- memory throughput
- compute utilization
- occupancy
- launch configuration
- register usage
- shared-memory usage

Profiler execution time should not be treated as normal benchmark runtime because profiling can instrument and replay CUDA kernels.

## Future Work

This project focuses on CUDA fundamentals.
The next goal is to move from basic CUDA programming toward performance-oriented GPU development.

A longer-term goal is to implement a more realistic numerical workload related to signal processing, robotics, or FMCW radar and apply the same workflow:

```text
correct implementation
→ benchmark
→ profile
→ identify bottleneck
→ optimize
→ benchmark again
```
