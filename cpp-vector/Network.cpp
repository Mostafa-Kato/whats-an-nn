//
// Created by mkato on 7/1/2026.
//

#include "Network.h"


Network::Network(const vector<int>& layer_sizes) {
    for (int i = 0; i < static_cast<int>(layer_sizes.size()) - 1; i++) {
        this->layers.push_back( std::make_shared<Layer>(layer_sizes[i], layer_sizes[i+1]));
    }
}

TensorPtr Network::operator()(TensorPtr inputs) const {
    for (int i = 0; i < static_cast<int>(this->layers.size()); i++) {
        if (i == this->layers.size() - 1) {
            inputs = (*this->layers[i])(inputs, true);
        }
        else {
            inputs = (*this->layers[i])(inputs);
        }
    }
    return inputs;
}

vector<TensorPtr> Network::parameters() const {
    vector<TensorPtr> parameters;

    for (const auto& layer : layers) {
        auto layer_params = layer->parameters();
        parameters.insert(parameters.end(),
                          layer_params.begin(),
                          layer_params.end());
    }

    return parameters;
}

void Network::zero_grad() const {
    for (auto const& parameter : this->parameters()) {
        const int num_cols = parameter->grad->cols;
        const int num_rows = parameter->grad->rows;
        parameter->grad = std::make_shared<Matrix>(vector<double>(num_cols*num_rows, 0.0), num_rows, num_cols);
    }
}

void Network::update_parameter(const double learning_rate) const {
    for (auto const& parameter : this->parameters()) {
        parameter->data = parameter->data - parameter->grad * learning_rate;
    }
}
