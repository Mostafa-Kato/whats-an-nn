//
// Created by mkato on 8/12/26.
//

#ifndef WHATS_AN_NN_MATRIX_CUH
#define WHATS_AN_NN_MATRIX_CUH
#include <vector>
#include <memory>
#include "cudaErrorCheck.cuh"

using std::vector;
using std::shared_ptr;

class Matrix {
public:
    int rows, cols;
    double* data;

    Matrix(int rows, int cols, const vector<double>& data_vector);
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double* data);

    // no copying allowed cause manual memory is scary
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    ~Matrix();

    std::shared_ptr<Matrix> transpose();

};

using MatrixPtr = shared_ptr<Matrix>;

__global__ void matAddKernel(const double* a, const double* b, double* c, int rows, int cols);
__global__ void matMulKernel(const double* a, const double* b, double* c, int rows, int cols, int inner);
__global__ void scalMatMulKernel(const double* a, double b, double* c, int rows, int cols);
__global__ void matMulElementWiseKernel(const double* a, const double* b, double* c, int rows, int cols);
__global__ void matExpKernel(const double* a, double* res, int rows, int cols);
__global__ void matLogKernel(const double* a, double* res, int rows, int cols);
__global__ void matPowKernel(const double* a, double* res, double pow, int rows, int cols);
__global__ void matRELUKernel(const double* a, double* res, int rows, int cols);
__global__ void RELUGradKernel(double* data, double* res, int rows, int cols)

dim3 calcGridSize2D(const dim3& blockSize, int rows, int cols);
MatrixPtr matAdd(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matSub(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matMul(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr operator*(const MatrixPtr& a, double b);
MatrixPtr operator*(double b, const MatrixPtr& a);
MatrixPtr matMulElementWise(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matExp(const MatrixPtr& a);
MatrixPtr matLog(const MatrixPtr& a);
MatrixPtr matPow(const MatrixPtr& a, double power);
MatrixPtr matRELU(const MatrixPtr& a);
MatrixPtr matRELUGrad(const MatrixPtr& a);
void printMatrix(const MatrixPtr& a);


#endif //WHATS_AN_NN_MATRIX_CUH
