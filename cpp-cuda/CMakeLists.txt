cmake_minimum_required(VERSION 3.20)
project(cpp_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cpp_cuda
        main.cu
        Matrix.cu
        Matrix.cuh
        cudaErrorCheck.cuh
        Tensor.cu
        Tensor.cuh
)