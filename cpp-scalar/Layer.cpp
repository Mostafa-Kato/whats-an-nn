//
// Created by mkato on 7/1/2026.
//

#include "Layer.h"

Layer::Layer(const int num_inputs, const int num_neurons) {
    for (int i = 0; i < num_neurons; i++) {
        this->neurons.push_back(std::make_shared<Neuron>(num_inputs));
    }
}

vector<ValuePtr> Layer::operator()(const vector<ValuePtr>& input_vector, const bool last) const {
    vector<ValuePtr> outputs;
    for (const auto& neuron : this->neurons) {
        if (last) {
            outputs.push_back(neuron->operator()(input_vector, false));
        }
        else {
            outputs.push_back(neuron->operator()(input_vector));
        }
    }
    return outputs;
}

vector<ValuePtr> Layer::parameters() const {
    vector<ValuePtr> parameters;
    for (const auto& neuron : this->neurons) {
        auto neuronParameters = neuron->parameters();
        parameters.insert(parameters.end(), neuronParameters.begin(), neuronParameters.end());
    }
    return parameters;
}

