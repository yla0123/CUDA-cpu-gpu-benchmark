cmake_minimum_required(VERSION 3.20)

project(cuda_cpu_gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add src/vector_add.cu)
add_executable(vector_scale src/vector_scale.cu)
add_executable(matrix_mul src/matrix_mul.cu)