#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

void checkCuda(cudaError_t error, const char* message){
    // error function: takes the result of a CUDA API call and prints a error message if failed.
    if (error != cudaSuccess){
        printf("%s: %s\n", message, cudaGetErrorString(error));
    }
}

__global__ void VectorScale(const float* x, float* y, const float scale, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){y[i] = scale * x[i];}
}

int main(){
    const int N = 8;
    const float scale = 2.5f;

    std::vector<float> x(N);
    std::vector<float> y(N);

    for(int i = 0; i < N; ++i){
        x[i] = static_cast<float>(i + 1); //initialize
    }

    float* d_x = nullptr;
    float* d_y = nullptr;

    checkCuda(cudaMalloc(&d_x, N*sizeof(float)), "Malloc d_x");
    checkCuda(cudaMalloc(&d_y, N*sizeof(float)), "Malloc d_x");

    checkCuda(cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice), "Memcpy x");

    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorScale<<<numBlocks,threadsPerBlock>>>(d_x, d_y, scale, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    checkCuda(cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost), "Memcpy y");

    // Results
    printf("Input:  ");
    for (int i = 0; i < N; ++i){
        printf("%.1f ", x[i]);
    }

    printf("\nOutput: ");
    for (int i = 0; i < N; ++i){
        printf("%.1f ", y[i]);
    }
    printf("\n");

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}