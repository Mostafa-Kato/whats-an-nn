//
// Created by mkato on 7/1/2026.
//

#ifndef WHATS_AN_NN_MNIST_LOADER_H
#define WHATS_AN_NN_MNIST_LOADER_H
#include <fstream>
#include <vector>
#include <stdexcept>

int reverseInt(int i);
std::vector<std::vector<double>> loadImages(const std::string& path);
std::vector<std::vector<double>> loadLabels(const std::string& path);

#endif //WHATS_AN_NN_MNIST_LOADER_H
