//
// Created by mkato on 7/1/2026.
//

#ifndef WHATS_AN_NN_NETWORK_H
#define WHATS_AN_NN_NETWORK_H

#include "Layer.h"
using LayerPtr = std::shared_ptr<Layer>;

class Network {
    vector<LayerPtr> layers;

public:
    Network(const vector<int>& layer_sizes);
    TensorPtr operator()(TensorPtr inputs) const;
    vector<TensorPtr> parameters() const;
    void zero_grad() const;
    void update_parameter(double learning_rate) const;
};


#endif //WHATS_AN_NN_NETWORK_H
