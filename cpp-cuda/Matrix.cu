//
// Created by mkato on 8/12/26.
//

#include <stdexcept>
#include "Matrix.cuh"

__global__ void matAddKernel(double* a, double* b, double* c, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i*cols + j] = a[i*cols + j] + b[i*cols + j];
    }
}

MatrixPtr matAdd(const MatrixPtr& a, const MatrixPtr& b) {
    if (a->cols != b->cols || a->rows != b->rows) {
        throw std::invalid_argument("Matrices are not compatible. ");
    }

    std::make_shared<Matrix>result(a->rows, a->cols);

    dim3 blockSize(16,16);
    dim3 gridSize(
        (blockSize.x + a->cols - 1) / blockSize.x,
        (blockSize.y + a->rows - 1) / blockSize.y
    );

    matAddKernel<<<gridSize,blockSize>>> (a->data, b->data, result->data, a->rows, a->cols);
    cudaDeviceSynchronize();
    return result;

}
