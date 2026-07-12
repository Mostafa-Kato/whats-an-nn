//
// Created by mkato on 7/1/2026.
//

#ifndef WHATS_AN_NN_LAYER_H
#define WHATS_AN_NN_LAYER_H
#include "Tensor.h"

class Layer {
    TensorPtr weights;
    TensorPtr biases;

public:
    Layer(int num_inputs, int num_neurons);
    TensorPtr operator()(const TensorPtr& input_vector, bool last = false) const;
    vector<TensorPtr> parameters() const;
};


#endif //WHATS_AN_NN_LAYER_H
