//
// Created by mkato on 8/13/26.
//

#ifndef WHATS_AN_NN_TENSOR_CUH
#define WHATS_AN_NN_TENSOR_CUH
#include "Matrix.cuh"
#include "functional"

class Tensor {
public:
   MatrixPtr grad;
   MatrixPtr data;
   std::function<void()> back;
   vector<shared_ptr<Tensor>> children;

   Tensor(MatrixPtr data, vector<shared_ptr<Tensor>> children = {});

   void backward();

};

using TensorPtr = shared_ptr<Tensor>;

TensorPtr operator*(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator+(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator-(const TensorPtr& a, const TensorPtr& b);
TensorPtr operator*(const TensorPtr& a, double b);
TensorPtr operator*(double b, const TensorPtr& a);
TensorPtr tenAddBias(const TensorPtr& a, const TensorPtr& b);
TensorPtr tenPow(const TensorPtr& a, double power);
TensorPtr tenExp(const TensorPtr& a);
TensorPtr tenLog(const TensorPtr& a);
TensorPtr tenRELU(const TensorPtr& a);


#endif //WHATS_AN_NN_TENSOR_CUH
