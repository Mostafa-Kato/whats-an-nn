//
// Created by mkato on 8/13/26.
//

#include "Tensor.cuh"

#include <unordered_set>
#include <utility>


Tensor::Tensor(MatrixPtr data, vector<shared_ptr<Tensor>> children)
: data(std::move(data)), grad(std::make_shared<Matrix>(data->rows, data->cols)), children(std::move(children)) {}

TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
   auto res_data = matMul(a->data, b->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a,b});

   Tensor* res = result.get();
   result->back = [a, b, res]() {
      a->grad = matAdd(a->grad, matMul(res->grad, b->data->transpose()));
      b->grad = matAdd(b->grad, matMul(a->data->transpose(), res->grad));
   };
   return result;
}

TensorPtr operator*(const TensorPtr &a, double b) {
   auto res_data = a->data*b;
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a});

   Tensor* res = result.get();
   result->back = [a, b, res]() {
      a->grad =  matAdd(res->grad * b, a->grad);
   };

   return result;

}

TensorPtr operator*(double b, const TensorPtr &a) {
   return a*b;
}

TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
   auto res_data = matAdd(a->data, b->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a,b});

   Tensor* res = result.get();
   result->back = [a, b, res]() {
      a->grad = matAdd(res->grad, a->grad);
      b->grad = matAdd(res->grad, b->grad);
   };
   return result;
}

TensorPtr operator-(const TensorPtr &a, const TensorPtr &b) {
   auto res_data = matAdd(a->data, -1*b->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a,b});

   Tensor* res = result.get();
   result->back = [a, b, res]() {
      a->grad = matAdd(res->grad, a->grad);
      b->grad = matAdd(-1*res->grad, b->grad);
   };
   return result;
}

TensorPtr tenAddBias(const TensorPtr &a, const TensorPtr &b) {
   auto res_data = matAddBias(a->data, b->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a,b});

   Tensor* res = result.get();
   result->back = [a, b, res]() {
      a->grad = matAdd(res->grad, a->grad);
      b->grad = matAdd(matSumRows(res->grad), b->grad);
   };
   return result;
}

TensorPtr tenPow(const TensorPtr &a, double power) {
   auto res_data = matPow(a->data, power);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr>{a});

   Tensor* res = result.get();
   result->back = [a, power, res] () {
      auto drda = power * matPow(a->data, power - 1);
      a->grad = matAdd(a->grad, matMulElementWise(res->grad, drda));
   };
   return result;
}

TensorPtr tenExp(const TensorPtr &a) {
   auto res_data = matExp(a->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr> {a});

   Tensor* res = result.get();
   result->back = [a, res] () {
      a->grad = matAdd(a->grad, matMulElementWise(res->grad, res->data));
   };
   return result;
}

TensorPtr tenLog(const TensorPtr &a) {
   auto res_data = matLog(a->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr> {a});

   Tensor* res = result.get();
   result->back = [a, res] () {
      auto drda = matPow(a->data, -1);
      a->grad = matAdd(a->grad, matMulElementWise(res->grad, drda));
   };
   return result;
}

TensorPtr tenRELU(const TensorPtr &a) {
   auto res_data = matRELU(a->data);
   auto result = std::make_shared<Tensor>(res_data, vector<TensorPtr> {a});

   Tensor* res = result.get();
   result-> back = [a, res] () {
      auto drda = matRELUGrad(a->data);
      a->grad = matAdd(a->grad, matMulElementWise(res->grad, drda));
   };

   return result;
}

void Tensor::backward() {
   std::unordered_set<Tensor*> visited;
   vector<Tensor*> order;
   std::function<void(Tensor*)> visit = [&] (Tensor* n) {
      if (visited.count(n)) return; //has n been visited
      visited.insert(n);
      for (auto& child : n->children) visit(child.get());
      order.push_back(n);
   };
   visit(this);

   this->grad = std::make_shared<Matrix>(
       this->data->rows,
       this->data->cols,
       vector<double>(this->data->rows * this->data->cols, 1.0)
       );
   
   for (auto it= order.rbegin(); it != order.rend(); ++it) {
      if ((*it)->back) {
         (*it)->back();
      }
   }
}

