//
// Created by mkato on 8/14/26.
//

#ifndef WHATS_AN_NN_NETWORK_CUH
#define WHATS_AN_NN_NETWORK_CUH
#include "Layer.cuh"
using LayerPtr = shared_ptr<Layer>;

class Network {
    vector<LayerPtr> layers;

public:
    Network(const vector<int>& layer_sizes);
    TensorPtr operator()(TensorPtr inputs) const;
    vector<TensorPtr> parameters() const;
    void zero_grad() const;
    void update_parameter(double learning_rate) const;
};


#endif //WHATS_AN_NN_NETWORK_CUH
