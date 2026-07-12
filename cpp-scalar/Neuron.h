//
// Created by mkato on 7/1/2026.
//

#ifndef WHATS_AN_NN_NEURON_H
#define WHATS_AN_NN_NEURON_H
#include "Value.h"



using std::shared_ptr;
using std::vector;
using std::function;


class Neuron {
    vector<ValuePtr> weights;
    ValuePtr bias;

public:
    Neuron(int num_inputs);
    ValuePtr operator()(const vector<ValuePtr>& inputs, bool activate = true) const;

    vector<ValuePtr> parameters() const;
};


#endif //WHATS_AN_NN_NEURON_H
