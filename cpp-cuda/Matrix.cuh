//
// Created by mkato on 8/12/26.
//

#ifndef WHATS_AN_NN_MATRIX_CUH
#define WHATS_AN_NN_MATRIX_CUH
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

class Matrix {
public:
    int rows, cols;
    double* data;

    Matrix(int rows, int cols, const vector<double>& data_vector) : rows(rows), cols(cols), data() {
        int bytes = rows*cols*sizeof(double);
        cudaMalloc(&data, rows*cols*sizeof(double));
        cudaMemcpy(data, data_vector.data(), bytes, cudaMemcpyHostToDevice);
    }

};

using MatrixPtr = shared_ptr<Matrix>;

__global__ void matAddKernel(double* a, double* b);
MatrixPtr matAdd(const MatrixPtr& a, MatrixPtr b);
MatrixPtr matMul(MatrixPtr a, MatrixPtr b);


#endif //WHATS_AN_NN_MATRIX_CUH
