//
// Created by mkato on 8/14/26.
//

#ifndef WHATS_AN_NN_LAYER_CUH
#define WHATS_AN_NN_LAYER_CUH

#include "Tensor.cuh"

class Layer {
    TensorPtr weights;
    TensorPtr biases;

public:
    Layer(int numInputs, int numNeurons);
    TensorPtr operator()(const TensorPtr& input_vector, bool last = false) const;
    vector<TensorPtr> parameters() const;
};




#endif //WHATS_AN_NN_LAYER_CUH
