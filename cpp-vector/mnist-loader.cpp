//
// Created by mkato on 7/1/2026.
//

#include "mnist-loader.h"

// reverses byte order - MNIST files are big-endian, x86 is little-endian
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}

// returns flat vector of pixel values, normalized to [0,1]
// shape: [num_images][784]
std::vector<std::vector<double>> loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);

    int magic, num_images, rows, cols;
    file.read((char*)&magic, 4);      magic = reverseInt(magic);
    file.read((char*)&num_images, 4); num_images = reverseInt(num_images);
    file.read((char*)&rows, 4);       rows = reverseInt(rows);
    file.read((char*)&cols, 4);       cols = reverseInt(cols);

    std::vector<std::vector<double>> images(num_images, std::vector<double>(rows * cols));
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            images[i][j] = pixel / 255.0;
        }
    }
    return images;
}

// returns vector of one-hot encoded labels
// shape: [num_images][10]
std::vector<std::vector<double>> loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);

    int magic, num_labels;
    file.read((char*)&magic, 4);      magic = reverseInt(magic);
    file.read((char*)&num_labels, 4); num_labels = reverseInt(num_labels);

    std::vector<std::vector<double>> labels(num_labels, std::vector<double>(10, 0.0));
    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels[i][(int)label] = 1.0;
    }
    return labels;
}