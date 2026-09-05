//
// Created by mkato on 8/12/26.
//

#include <stdexcept>
#include <iostream>
#include "Matrix.cuh"

dim3 calcGridSize2D(const dim3 &blockSize, int rows, int cols) {
    dim3 gridSize{};

    gridSize.x = (blockSize.x + cols - 1) / blockSize.x;
    gridSize.y = (blockSize.y + rows - 1) / blockSize.y;

    return gridSize;
}

Matrix::Matrix(int rows, int cols, const vector<double> &data_vector) : rows(rows), cols(cols), data(nullptr) {
    int bytes = rows * cols * sizeof(double);
    gpuErrchk(cudaMalloc(&this->data, bytes));
    gpuErrchk(cudaMemcpy(this->data, data_vector.data(), bytes, cudaMemcpyHostToDevice));
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(nullptr) {
    int bytes = rows * cols * sizeof(double);
    gpuErrchk(cudaMalloc(&this->data, bytes));
    gpuErrchk(cudaMemset(this->data, 0, bytes));
}

Matrix::Matrix(int rows, int cols, double *in_data) : rows(rows), cols(cols), data(nullptr) {
    int bytes = rows * cols * sizeof(double);
    gpuErrchk(cudaMalloc(&this->data, bytes));
    gpuErrchk(cudaMemcpy(this->data, in_data, bytes, cudaMemcpyHostToDevice));
}

Matrix::~Matrix() {
    cudaFree(data);
}

__global__ void matAddKernel(const double *a, const double *b, double *c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] + b[i * cols + j];
    }
}

__global__ void matAddBiasKernel(const double *a, const double *bias, double *c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] + bias[i];
    }
}

__global__ void matAddColBroadcastKernel(const double* a, const double* rowVector, double* c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] + rowVector[j];
    }
}

__global__ void matMulKernel(const double *a, const double *b, double *c, const int rows, const int cols,
                             const int inner) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = 0.0;
        for (int x = 0; x < inner; x++) {
            c[i * cols + j] += a[i * inner + x] * b[x * cols + j];
        }
    }
}

__global__ void matMulColBroadcastKernel(const double* a, const double* rowVector, double* c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] * rowVector[j];
    }
}

__global__ void scalMatMulKernel(const double *a, const double b, double *c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] * b;
    }
}

__global__ void scalMatAddKernel(const double *a, const double b, double *c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] + b;
    }
}

__global__ void matMulElementWiseKernel(const double *a, const double *b, double *c, const int rows, const int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        c[i * cols + j] = a[i * cols + j] * b[i * cols + j];
    }
}

__global__ void matExpKernel(const double *a, double *res, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        res[i * cols + j] = exp(a[i * cols + j]);
    }
}

__global__ void matLogKernel(const double *a, double *res, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        res[i * cols + j] = log(a[i * cols + j]);
    }
}

__global__ void matPowKernel(const double *a, double *res, double power, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        res[i * cols + j] = pow(a[i * cols + j], power);
    }
}

__global__ void matRELUKernel(const double *a, double *res, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        res[i * cols + j] = fmax(0.0, a[i * cols + j]);
    }
}

__global__ void RELUGradKernel(const double *data, double *res, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        res[i * cols + j] = (data[i * cols + j] > 0.0) ? 1.0 : 0.0;
    }
}

__global__ void matSumRowsKernel(const double *data, double *res, int rows, int cols) {
    int localIndex = threadIdx.x;
    extern __shared__ double s[];

    double sum = 0.0;
    for (int i = localIndex; i < cols; i += blockDim.x) {
        sum += data[i + blockIdx.x * cols];
    }
    s[localIndex] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            s[localIndex] += s[localIndex + stride];
        }
        __syncthreads();
    }

    if (localIndex == 0) {
        res[blockIdx.x] = s[0];
    }
}

__global__ void matSumColsKernel(const double *data, double *res, int rows, int cols) {
    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    double sum = 0.0;
    if (globalIndex < cols) {
        for (int i = 0; i < rows; i++) {
            sum += data[globalIndex + i * cols];
        }
    res[globalIndex] = sum;
    }

}

__global__ void matMaxRowsKernel(const double *data, double *res, int rows, int cols) {
    int localIndex = threadIdx.x;
    extern __shared__ double s[];

    double max = -INFINITY;
        for (int i = localIndex; i < cols; i += blockDim.x) {
            max = (data[i + blockIdx.x * cols] > max) ? data[i + blockIdx.x * cols] : max;
        }
        s[localIndex] = max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            s[localIndex] = (s[localIndex + stride] > s[localIndex]) ? s[localIndex + stride] : s[localIndex];
        }
        __syncthreads();
    }

    if (localIndex == 0) {
        res[blockIdx.x] = s[0];
    }
}

__global__ void matMaxColsKernel(const double *data, double *res, int rows, int cols) {
    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    double max = -INFINITY;
    if (globalIndex < cols) {
        for (int i = 0; i < rows; i++) {
            max = (data[globalIndex + i * cols] > max) ? data[globalIndex + i * cols] : max;
        }
    res[globalIndex] = max;
    }
}

__global__ void matTransposeKernel(const double *og, double *transposed, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < rows && j < cols) {
        transposed[j * rows + i] = og[i * cols + j];
    }
}

std::shared_ptr<Matrix> Matrix::transpose() {
    auto result = std::make_shared<Matrix>(this->cols, this->rows);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, this->rows, this->cols);

    matTransposeKernel<<<gridSize, blockSize>>>(this->data, result->data, this->rows, this->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matAdd(const MatrixPtr &a, const MatrixPtr &b) {
    if (!(a->cols == b->cols && a->rows == b->rows)) {
        throw std::invalid_argument("Matrices are incompatible. ");
    }

    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matAddKernel<<<gridSize,blockSize>>>(a->data, b->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matAddBias(const MatrixPtr &a, const MatrixPtr &bias) {
    if (bias->cols != 1 || bias->rows != a->rows) {
        throw std::invalid_argument("Bias should have same rows as a and 1 col");
    }
    auto result = std::make_shared<Matrix>(a->rows, a->cols);
    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);


    matAddBiasKernel<<<gridSize, blockSize>>>(a->data, bias->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matAddColBroadcast(const MatrixPtr& a, const MatrixPtr& rowVector) {
    if (rowVector->cols != a->cols || rowVector->rows != 1) {
        throw std::invalid_argument("Row vector should have same cols as a and 1 row");
    }
    auto result = std::make_shared<Matrix>(a->rows, a->cols);
    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);


    matAddColBroadcastKernel<<<gridSize, blockSize>>>(a->data, rowVector->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matSub(const MatrixPtr &a, const MatrixPtr &b) {
    return matAdd(a, -1 * b);
}

MatrixPtr matMul(const MatrixPtr &a, const MatrixPtr &b) {
    if (a->cols != b->rows) {
        throw std::invalid_argument("Matrices are not compatible. ");
    }

    auto result = std::make_shared<Matrix>(a->rows, b->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, b->cols);


    matMulKernel<<< gridSize, blockSize>>>(a->data, b->data, result->data, a->rows, b->cols, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matMulColBroadcast(const MatrixPtr &a, const MatrixPtr& rowVector) {
    if (rowVector->cols != a->cols || rowVector->rows != 1) {
        throw std::invalid_argument("Row vector should have same cols as a and 1 row");
    }
    auto result = std::make_shared<Matrix>(a->rows, a->cols);
    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);


    matMulColBroadcastKernel<<<gridSize, blockSize>>>(a->data, rowVector->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr operator*(const MatrixPtr &a, double b) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    scalMatMulKernel<<<gridSize, blockSize>>>(a->data, b, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr operator*(const double b, const MatrixPtr &a) {
    return a * b;
}

MatrixPtr operator/(const MatrixPtr &a, const double b) {
    return a * pow(b, -1.0);
}

MatrixPtr scalMatAdd(const MatrixPtr &a, const double b) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);
    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    scalMatAddKernel<<< gridSize, blockSize>>>(a->data, b, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr scalMatAdd(const double b, const MatrixPtr &a) {
    return scalMatAdd(a, b);
}

MatrixPtr matMulElementWise(const MatrixPtr &a, const MatrixPtr &b) {
    if (!(a->cols == b->cols && a->rows == b->rows)) {
        throw std::invalid_argument("Matrices are incompatible. ");
    }
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matMulElementWiseKernel<<< gridSize, blockSize>>>(a->data, b->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matExp(const MatrixPtr &a) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matExpKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matLog(const MatrixPtr &a) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matLogKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matPow(const MatrixPtr &a, const double power) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matPowKernel<<<gridSize, blockSize>>>(a->data, result->data, power, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matRELU(const MatrixPtr &a) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);

    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    matRELUKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matRELUGrad(const MatrixPtr &a) {
    auto result = std::make_shared<Matrix>(a->rows, a->cols);
    dim3 blockSize(16, 16);
    dim3 gridSize = calcGridSize2D(blockSize, a->rows, a->cols);

    RELUGradKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
    gpuErrchk(cudaPeekAtLastError());

    return result;
}

MatrixPtr matSumAxis(const MatrixPtr &a, AXIS axis) {
    if (axis == ROWS) {
        int length = (a->cols > 256) ? 1024 : 256;
        dim3 blockSize(length);
        dim3 gridSize(a->rows);
        auto memSize = length * sizeof(double);
        auto result = std::make_shared<Matrix>(a->rows, 1);
        matSumRowsKernel<<<gridSize, blockSize, memSize>>>(a->data, result->data, a->rows, a->cols);
        gpuErrchk(cudaPeekAtLastError());
        return result;
    } else {
        int length = (a->cols > 256) ? 1024 : 256;
        dim3 blockSize(length);
        dim3 gridSize = (blockSize.x + a->cols - 1) / blockSize.x;
        auto result = std::make_shared<Matrix>(1, a->cols);
        matSumColsKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
        gpuErrchk(cudaPeekAtLastError());
        return result;
    }
}

MatrixPtr matMaxAxis(const MatrixPtr &a, AXIS axis) {
    if (axis == ROWS) {
        int length = (a->cols > 256) ? 1024 : 256;
        dim3 blockSize(length);
        dim3 gridSize(a->rows);
        auto memSize = length * sizeof(double);
        auto result = std::make_shared<Matrix>(a->rows, 1);
        matMaxRowsKernel<<<gridSize, blockSize, memSize>>>(a->data, result->data, a->rows, a->cols);
        gpuErrchk(cudaPeekAtLastError());
        return result;
    } else {
        int length = (a->cols > 256) ? 1024 : 256;
        dim3 blockSize(length);
        dim3 gridSize = (blockSize.x + a->cols - 1) / blockSize.x;
        auto result = std::make_shared<Matrix>(1, a->cols);
        matMaxColsKernel<<<gridSize, blockSize>>>(a->data, result->data, a->rows, a->cols);
        gpuErrchk(cudaPeekAtLastError());
        return result;
    }
}

void printMatrix(const MatrixPtr &a) {
    auto b = vector<double>(a->rows * a->cols, 0);
    gpuErrchk(cudaMemcpy(b.data(), a->data, sizeof(double)*a->rows*a->cols, cudaMemcpyDeviceToHost));

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            std::cout << b[i * a->cols + j] << " ";
        }
        std::cout << "\n";
    }
}
