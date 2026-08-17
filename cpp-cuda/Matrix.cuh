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

enum AXIS {ROWS, COLS};


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

dim3 calcGridSize2D(const dim3& blockSize, int rows, int cols);
MatrixPtr matAdd(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr scalMatAdd(const MatrixPtr& a, double b);
MatrixPtr scalMatAdd(double b, const MatrixPtr& a);
MatrixPtr matAddBias(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matSub(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matMul(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matMulElementWise(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr operator*(const MatrixPtr& a, double b);
MatrixPtr operator*(double b, const MatrixPtr& a);
MatrixPtr operator/(const MatrixPtr& a, double b);
MatrixPtr matExp(const MatrixPtr& a);
MatrixPtr matLog(const MatrixPtr& a);
MatrixPtr matPow(const MatrixPtr& a, double power);
MatrixPtr matRELU(const MatrixPtr& a);
MatrixPtr matRELUGrad(const MatrixPtr& a);
MatrixPtr matSumAxis(const MatrixPtr& a, AXIS axis);
MatrixPtr matMaxAxis(const MatrixPtr& a, AXIS axis);
void printMatrix(const MatrixPtr& a);


#endif //WHATS_AN_NN_MATRIX_CUH
