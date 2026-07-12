//
// Created by mkato on 7/2/2026.
//

#ifndef WHATS_AN_NN_MATRIX_H
#define WHATS_AN_NN_MATRIX_H

#include <functional>
#include <memory>
#include <vector>
#include <iostream>

using std::shared_ptr;
using std::vector;
using std::function;

class Matrix {
public:
    vector<double> data;
    int rows{}, cols{};

    Matrix(const vector<double>& data, int rows, int cols);

    std::shared_ptr<Matrix> transpose();

};

using MatrixPtr = std::shared_ptr<Matrix>;

MatrixPtr operator+(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr operator-(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr operator*(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr operator*(const MatrixPtr& a, double b);
MatrixPtr operator*(double b, const MatrixPtr& a);
MatrixPtr elementwiseMul(const MatrixPtr& a, const MatrixPtr& b);
MatrixPtr matpow(const MatrixPtr& a, double power);
MatrixPtr matexp(const MatrixPtr& a);
MatrixPtr matlog(const MatrixPtr& a);
MatrixPtr matrelu(const MatrixPtr& a);
std::ostream& operator<<(std::ostream&, const MatrixPtr& a);

#endif //WHATS_AN_NN_MATRIX_H
