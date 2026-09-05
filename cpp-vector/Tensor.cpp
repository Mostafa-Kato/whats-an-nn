//
// Created by mkato on 7/2/2026.
//

#include "Tensor.h"

#include <utility>
#include <cmath>
#include <set>

Tensor::Tensor(MatrixPtr data, vector<TensorPtr> children, std::string op)
    : data(data),
      grad(std::make_shared<Matrix>(vector<double>(data->rows * data->cols, 0.0), data->rows, data->cols)),
      back([](){}),
      op(std::move(op)),
      children(std::move(children)) {}


TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) {
    MatrixPtr newData = a->data + b->data;
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a,b}, "+");
    result->back = [a, b, result]() {
        a->grad = result->grad + a->grad;
        b->grad = result->grad + b->grad;
    };
    return result;
}

TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) {
    MatrixPtr newData = a->data * b->data;
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a,b}, "*");
    result->back = [a, b, result]() {
        a->grad = a->grad + result->grad * b->data->transpose();
        b->grad = b->grad + a->data->transpose() * result->grad;
    };
    return result;
}

TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) {
    MatrixPtr newData = a->data - b->data;
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a,b}, "-");
    result->back = [a, b, result]() {
        a->grad = result->grad + a->grad;
        b->grad = b->grad -result->grad;
    };
    return result;
}

TensorPtr operator/(const TensorPtr& a, double scalar) {
    MatrixPtr scaledData = a->data * (1.0 / scalar);
    auto result = std::make_shared<Tensor>(scaledData, vector<TensorPtr>{a}, "/");
    result->back = [a, scalar, result]() {
        a->grad = a->grad + result->grad * (1.0 / scalar);
    };
    return result;
}

TensorPtr pow(const TensorPtr& a, const double power) {
    MatrixPtr newData = matpow(a->data,power);
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a}, "pow");
    result->back = [a, result, power]() {
        a->grad = a->grad + elementwiseMul(power * ::matpow(a->data, power - 1), result->grad);
    };
    return result;
}

TensorPtr exp(const TensorPtr& a) {
    MatrixPtr newData = matexp(a->data);
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a}, "exp");
    result->back = [a, result]() {
        a->grad = a->grad + elementwiseMul(result->grad, result->data) ;
    };
    return result;
}

TensorPtr log(const TensorPtr& a) {
    MatrixPtr newData = matlog(a->data);
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a}, "log");
    result->back = [a, result]() {
        a->grad = a->grad + elementwiseMul(matpow(a->data, -1.0), result->grad);
    };
    return result;
}

TensorPtr relu(const TensorPtr& a) {
    MatrixPtr newData = matrelu(a->data);
    auto result = std::make_shared<Tensor>(newData, vector<TensorPtr>{a}, "relu");
    result->back = [a, result]() {
        // mask: 1.0 where result->data > 0, else 0.0
        vector<double> mask(result->data->rows * result->data->cols, 0.0);
        for (int i = 0; i < result->data->rows * result->data->cols; i++)
            mask[i] = result->data->data[i] > 0 ? 1.0 : 0.0;
        auto maskMatrix = std::make_shared<Matrix>(mask, result->data->rows, result->data->cols);
        a->grad = a->grad + elementwiseMul(maskMatrix, result->grad);
    };
    return result;

}

void Tensor::backward() {
    std::set<Tensor*> visited;
    vector<Tensor*> order;
    function<void(Tensor*)> visit = [&](Tensor* n) {
        if (visited.count(n)) return;
        visited.insert(n);
        for (auto& child : n->children) visit(child.get());
        order.push_back(n);
    };
    visit(this);

    this->grad = std::make_shared<Matrix>(
        vector<double>(this->data->rows * this->data->cols, 1.0),
        this->data->rows, this->data->cols);
    for (auto it= order.rbegin(); it != order.rend(); ++it) {
        (*it)->back();
    }
}