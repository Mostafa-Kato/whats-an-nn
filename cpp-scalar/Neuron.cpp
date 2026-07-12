//
// Created by mkato on 7/1/2026.
//

#include "Neuron.h"
#include <random>

std::mt19937 gen(42);
std::normal_distribution<double> dist(0.0, 1.0);

Neuron::Neuron(const int num_inputs) {
    for (int i = 0; i < num_inputs; i++) {
        double w = dist(gen) * (1.0 / std::sqrt(static_cast<double>(num_inputs)));
        this->weights.push_back(std::make_shared<Value>(w));
    }
    this->bias = std::make_shared<Value>(0.0);
}

ValuePtr Neuron::operator()(const vector<ValuePtr>& inputs, const bool activate) const {
    auto output = std::make_shared<Value>(0);
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        output = output + inputs[i] * this->weights[i];
    }
    if (activate) {
        return relu(output + this->bias);
    }
    return output + this->bias;
}

vector<ValuePtr> Neuron::parameters() const {
    vector<ValuePtr> parameters;
    parameters.reserve(this->weights.size() + 1);
    parameters.insert(parameters.end(), this->weights.begin(), this->weights.end());
    parameters.push_back(this->bias);
    return parameters;
}