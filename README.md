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

The vector scaling example performs:

```cpp
y[i] = scale * x[i];
```

It demonstrates the standard CUDA one-thread-per-element execution pattern.

This example focuses on correctness and CUDA indexing rather than detailed benchmarking.

### Matrix Multiplication

The matrix multiplication example uses a two-dimensional CUDA grid.

Each GPU thread computes one element of the output matrix:

```text
C[row][col]
```

The matrices are stored as contiguous one-dimensional arrays and indexed manually.

For example:

```cpp
a[row * K + i]
b[i * N + col]
c[row * N + col]
```

The current implementation is intentionally naive and does not use shared-memory tiling.

## CUDA Concepts Practiced

The project includes hands-on practice with:

- CUDA kernels and `__global__`
- kernel launch syntax
- threads, blocks, and grids
- `threadIdx`, `blockIdx`, and `blockDim`
- global thread indexing
- 1D and 2D CUDA grids
- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`
- `cudaDeviceSynchronize`
- CUDA error checking
- CUDA events
- global memory
- shared memory
- `__syncthreads()`
- parallel reduction
- memory coalescing
- memory-bound vs compute-bound workloads
- kernel time vs end-to-end time
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute

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

Each size was executed four times using the Release build. The median is reported to reduce the influence of run-to-run variation.

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

This is useful for answering:

> Where is the application spending its time?

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

This is useful for answering:

> Why is this kernel behaving this way?

Profiler execution time should not be treated as normal benchmark runtime because profiling can instrument and replay CUDA kernels.

## Performance Lessons

### GPU execution is not automatically faster

Launching a GPU kernel introduces additional costs. Small workloads may be faster on the CPU.

### Kernel time is only part of the application

A CUDA kernel can execute very quickly while the total GPU workflow remains relatively expensive.

### Data movement matters

Host-to-device and device-to-host transfers may cost considerably more than the computation itself.

Whenever possible, larger applications should keep data resident on the GPU across multiple operations instead of repeatedly copying it between CPU and GPU.

### Memory access matters

Neighboring GPU threads should ideally access neighboring memory addresses so memory transactions can be coalesced.

### Arithmetic intensity matters

Simple vector operations perform very little computation for each value transferred from memory and are commonly memory-oriented.

Matrix multiplication performs substantially more arithmetic and creates more opportunities for data reuse.

### Optimization should follow measurement

The optimization workflow used in this project is:

```text
measure
→ identify bottleneck
→ make a change
→ measure again
→ explain the result
```

Optimization should focus on the actual bottleneck rather than assuming the CUDA kernel itself is always the problem.

## Current Limitations

This project focuses on CUDA fundamentals.

The current implementations do not yet include:

- tiled shared-memory matrix multiplication
- pinned host memory
- asynchronous memory transfers
- CUDA streams
- overlapping computation and communication
- warp-level primitives
- detailed occupancy tuning
- Tensor Cores
- CUDA Graphs
- multi-GPU programming

## Future Work

The next goal is to move from basic CUDA programming toward performance-oriented GPU development.

Planned extensions include:

- implement tiled matrix multiplication using shared memory
- compare naive and tiled matrix multiplication performance
- study the effect of different CUDA block sizes
- investigate memory coalescing experimentally
- reuse allocated GPU memory instead of repeatedly allocating buffers
- reduce unnecessary CPU-GPU transfers
- experiment with pinned host memory
- introduce asynchronous transfers and CUDA streams
- investigate overlapping memory transfer with computation
- perform deeper analysis using Nsight Systems and Nsight Compute
- add larger and more systematic benchmark experiments

A longer-term goal is to implement a more realistic numerical workload related to signal processing, robotics, or FMCW radar and apply the same workflow:

```text
correct implementation
→ benchmark
→ profile
→ identify bottleneck
→ optimize
→ benchmark again
```

## Main Takeaway

The main lesson from this project is that CUDA performance is determined by more than kernel execution speed.

Effective GPU programming requires considering the complete system:

```text
parallelism
+
memory access
+
data movement
+
synchronization
+
computation
+
profiling
```

The goal is not simply to move computation onto the GPU, but to understand where the application spends its time and structure the workload so that GPU parallelism can be used effectively.