//
// Created by mkato on 8/14/2026.
//

#include "Layer.cuh"
#include <random>

using std::make_shared;

std::mt19937 gen(42);
std::normal_distribution<double> dist(0.0, 1.0);

Layer::Layer(const int num_inputs, const int num_neurons) {

    vector<double> weightVector(num_inputs * num_neurons, 0.0);
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_inputs; j++) {
            double w = dist(gen) * (1.0 / std::sqrt(static_cast<double>(num_inputs)));
            weightVector[i * num_inputs + j] = w;
        }
    }
    this->weights = std::make_shared<Tensor>(std::make_shared<Matrix>(num_neurons, num_inputs, weightVector));
    this->biases = std::make_shared<Tensor>(std::make_shared<Matrix>(num_neurons, 1, vector<double>(num_neurons, 0.0)));
}

TensorPtr Layer::operator()(const TensorPtr &input_vector, bool last) const {
    if (!last) {
        return tenRELU(tenAddBias(this->weights * input_vector ,biases));
    }
    return tenAddBias(this->weights*input_vector, biases);
}

vector<TensorPtr> Layer::parameters() const {
    return {this->weights, this->biases};
}
