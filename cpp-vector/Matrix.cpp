//
// Created by mkato on 7/2/2026.
//

#include "Matrix.h"

#include <cmath>
#include <stdexcept>

Matrix::Matrix(const vector<double>& data, const int rows, const int cols)
    : data (data), rows (rows), cols(cols) {}

MatrixPtr operator+(const MatrixPtr& a, const MatrixPtr& b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    vector<double> result(a->rows * a->cols, 0.0);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result[i * a->cols + j] = (a->data[i*a->cols + j] + b->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr operator-(const MatrixPtr& a, const MatrixPtr &b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    vector<double> result(a->rows * a->cols, 0.0);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result[i * a->cols + j] = (a->data[i*a->cols + j] - b->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr operator*(const MatrixPtr& a, const MatrixPtr &b) {
    if (a->cols != b->rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible");
    }

    vector<double> result(a->rows * b->cols, 0.0);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < b->cols; j++)
            for (int k = 0; k < a->cols; k++)
                result[i * b->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
    return std::make_shared<Matrix>(result, a->rows, b->cols);
}

MatrixPtr operator*(const MatrixPtr& a, const double b) {

    vector<double> result(a->rows * a->cols, 0.0);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            result[i * a->cols + j] = a->data[i*a->cols + j] * b;
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr operator*(const double b, const MatrixPtr& a) {
    return a*b;
}

MatrixPtr elementwiseMul(const MatrixPtr& a, const MatrixPtr& b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    vector<double> result(a->rows * a->cols, 0.0);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result[i * a->cols + j] = (a->data[i*a->cols + j] * b->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr matpow(const MatrixPtr& a, double power) {

    vector<double> result(a->rows*a->cols, 0.0);
    for (int i=0; i < a->rows; i++) {
        for (int j=0; j < a->cols; j++) {
            result[i * a->cols + j] = std::pow(a->data[i*a->cols + j], power);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr matexp(const MatrixPtr& a) {
    vector<double> result(a->rows*a->cols, 0.0);
    for (int i=0; i < a->rows; i++) {
        for (int j=0; j < a->cols; j++) {
            result[i * a->cols + j] = std::exp(a->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr matlog(const MatrixPtr& a) {
    vector<double> result(a->rows*a->cols, 0.0);
    for (int i=0; i < a->rows; i++) {
        for (int j=0; j < a->cols; j++) {
            result[i * a->cols + j] = std::log(a->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

MatrixPtr matrelu(const MatrixPtr& a) {
    vector<double> result(a->rows*a->cols, 0.0);
    for (int i=0; i < a->rows; i++) {
        for (int j=0; j < a->cols; j++) {
            result[i * a->cols + j] = std::max(0.0 ,a->data[i*a->cols + j]);
        }
    }
    return std::make_shared<Matrix>(result, a->rows, a->cols);
}

std::ostream& operator<<(std::ostream& os, const MatrixPtr &m) {
    for (int i = 0; i < m->rows; i++) {
        os << "[";
        for (int j = 0; j < m->cols; j++) {
            os << m->data[i * m->cols + j];
            if (j < m->cols - 1) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

std::shared_ptr<Matrix> Matrix::transpose() {
    vector<double> transpose(this->cols * this->rows, 0);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            transpose[j*this->rows + i] = this->data[i*this->cols + j];
        }
    }

    return std::make_shared<Matrix>(transpose, this->cols, this->rows);
}
