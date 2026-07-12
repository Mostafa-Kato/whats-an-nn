//
// Created by mkato on 7/1/2026.
//

#ifndef WHATS_AN_NN_LAYER_H
#define WHATS_AN_NN_LAYER_H
#include "Neuron.h"

using NeuronPtr = std::shared_ptr<Neuron>;

class Layer {
    vector<NeuronPtr> neurons;

public:
    Layer(int num_inputs, int num_neurons);
    vector<ValuePtr> operator()(const vector<ValuePtr>& input_vector, bool last = false) const;
    vector<ValuePtr> parameters() const;
};


#endif //WHATS_AN_NN_LAYER_H
