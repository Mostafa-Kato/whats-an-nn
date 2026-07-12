//
// Created by mkato on 7/1/2026.
//

#include "Network.h"


Network::Network(const vector<int>& layer_sizes) {
    for (int i = 0; i < static_cast<int>(layer_sizes.size()) - 1; i++) {
        this->layers.push_back( std::make_shared<Layer>(layer_sizes[i], layer_sizes[i+1]));
    }
}

vector<ValuePtr> Network::operator()(vector<ValuePtr> inputs) const {
    for (int i = 0; i < static_cast<int>(this->layers.size()); i++) {
        if (i == this->layers.size() -1) {
            inputs = (*this->layers[i])(inputs, true);
        }
        else {
            inputs = (*this->layers[i])(inputs);
        }
    }
    return inputs;
}

vector<ValuePtr> Network::parameters() const {
    vector<ValuePtr> parameters;

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
        parameter->grad = 0;
    }
}

void Network::update_parameter(const double learning_rate) const {
    for (auto const& parameter : this->parameters()) {
        parameter->data -= parameter->grad * learning_rate;
    }
}
