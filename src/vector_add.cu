#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

void checkCuda(cudaError_t error, const char* message){
    // error function: takes the result of a CUDA API call and prints a error message if failed.
    if (error != cudaSuccess){
        printf("%s: %s\n", message, cudaGetErrorString(error));
    }
}

__global__ void vectorAdd(const int* a, const int* b, int* c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        c[i] = a[i] + b[i];
    }
}

int main(){
    const int N = 10000000;
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> cpuResult(N);
    std::vector<int> gpuResult(N);

    for (int i = 0; i < N; ++i){
        a[i] = i;
        b[i] = 2 * i;
    }

    // CPU reference implementation
    auto cpuStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i){
        cpuResult[i] = a[i] + b[i];
    }

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    //========= Warm-up launch ============
    checkCuda(cudaDeviceSynchronize(), "Warm-up kernel");
    checkCuda(cudaFree(0), "CUDA initialization");

    int* d_warmup = nullptr;
    checkCuda(cudaMalloc(&d_warmup, sizeof(int)),"Warm-up cudaMalloc");

    vectorAdd<<<1, 1>>>(d_warmup, d_warmup, d_warmup,1);
    checkCuda(cudaDeviceSynchronize(), "Warm-up kernel");
    cudaFree(d_warmup);
    //======================

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    auto gpuStart = std::chrono::high_resolution_clock::now();

    checkCuda(
        cudaMalloc(&d_a, N * sizeof(int)),
        "cudaMalloc d_a"
    );

    checkCuda(
        cudaMalloc(&d_b, N * sizeof(int)),
        "cudaMalloc d_b"
    );

    checkCuda(
        cudaMalloc(&d_c, N * sizeof(int)),
        "cudaMalloc d_c"
    );

    checkCuda(
        cudaMemcpy(d_a, a.data(), N * sizeof(int), cudaMemcpyHostToDevice),
        "Memcpy a"
    );

    checkCuda(
        cudaMemcpy(d_b, b.data(), N * sizeof(int),cudaMemcpyHostToDevice),
        "Memcpy b"
    );

    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel-only timing
    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart);

    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    checkCuda(
        cudaGetLastError(),
        "Kernel launch"
    );

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);

    checkCuda(
        cudaMemcpy(gpuResult.data(), d_c, N * sizeof(int),  cudaMemcpyDeviceToHost),
        "Memcpy result"
    );

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpuTime = gpuEnd - gpuStart;

    // check the result
    bool correct = true;
    for (int i = 0; i < N; ++i){
        if (cpuResult[i] != gpuResult[i]){
            correct = false;
            break;
        }
    }

    printf("N = %d\n", N);
    printf("CPU time: %.6f ms\n", cpuTime.count());
    printf("GPU kernel time: %.6f ms\n", kernelTime);
    printf("GPU end-to-end time: %.6f ms\n", gpuTime.count());
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");

    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    return 0;
}