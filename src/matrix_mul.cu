#include <cuda_runtime.h>

#include <cstdio>

void checkCuda(cudaError_t error, const char* message){
    // error function: takes the result of a CUDA API call and prints a error message if failed.
    if (error != cudaSuccess){
        printf("%s: %s\n", message, cudaGetErrorString(error));
    }
}

__global__ void matrixMulti(const int* a, const int* b, int* c, int M, int K, int N){
    // martix a is M*K (flattened to 1D)
    // matrix b is K*N 
    // matrix c is M*N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        for (int i = 0; i < K; ++i){
            c[row * N + col] += a[row * K + i] * b[i * N + col];
        }
    }
}

int main(){
    const int M = 2;
    const int K = 3;
    const int N = 2;

    int a[M*K] = {1,2,3,4,5,6};
    int b[K*N] = {7,8,9,10,11,12};
    int c[M*N] = {0,0,0,0};

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    checkCuda(cudaMalloc(&d_a, M * K * sizeof(int)), "Malloc d_a");
    checkCuda(cudaMalloc(&d_b, K * N * sizeof(int)), "Malloc d_b");
    checkCuda(cudaMalloc(&d_c, M * N * sizeof(int)), "Malloc d_c");

    checkCuda(cudaMemcpy(d_a, a, M * K * sizeof(int), cudaMemcpyHostToDevice),"Memcpy a");
    checkCuda(cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice),"Memcpy b");

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulti<<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    checkCuda(cudaMemcpy(c, d_c, M*N*sizeof(int), cudaMemcpyDeviceToHost), "Memcpy c");


    // Result
    printf("Result:\n");
    for(int row = 0; row < M; ++row){
        for(int col = 0; col < N; ++col){
            printf("%d ", c[row*N + col]);
        }
        printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}