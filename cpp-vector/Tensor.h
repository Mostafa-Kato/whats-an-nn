//
// Created by mkato on 7/2/2026.
//

#ifndef WHATS_AN_NN_TENSOR_H
#define WHATS_AN_NN_TENSOR_H
#include "Matrix.h"


class Tensor {
public:
    MatrixPtr data;
    MatrixPtr grad;
    function<void()> back;
    std::string op;
    vector<shared_ptr<Tensor>> children;

    Tensor(MatrixPtr data, vector<shared_ptr<Tensor>> children = {}, std::string op = "");

    void backward();
};

using TensorPtr = std::shared_ptr<Tensor>;

TensorPtr operator+(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator*(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator-(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator/(const TensorPtr& a, double scalar);
TensorPtr pow(const TensorPtr& a, double power);
TensorPtr relu(const TensorPtr& a);
TensorPtr exp(const TensorPtr& a);
TensorPtr log(const TensorPtr& a);


#endif //WHATS_AN_NN_TENSOR_H
