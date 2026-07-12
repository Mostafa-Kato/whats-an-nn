//
// Created by mkato on 7/1/2026.
//
#include <algorithm>
#include <iostream>
#include <chrono>
#include<fstream>
#include "Network.h"
#include "mnist-loader.h"

vector<ValuePtr> toValuePtr(const vector<double>& inputs) {
    vector<ValuePtr> result;
    result.reserve(inputs.size());
    for (double val : inputs) {
        result.push_back(std::make_shared<Value>(val));
    }
    return result;
}

ValuePtr softmax_loss(const vector<ValuePtr>& predictions, const vector<double>& y_onehot) {
    auto maxval = *std::max_element(predictions.begin(), predictions.end(),
        [](const ValuePtr& a, const ValuePtr& b) { return a->data < b->data; });

    vector<ValuePtr> exps;
    exps.reserve(predictions.size());
    for (const auto& p : predictions) {
        exps.push_back(exp(p - maxval));
    }

    ValuePtr total = std::make_shared<Value>(0.0);
    for (const auto& e : exps) total = total + e;

    vector<ValuePtr> probs;
    probs.reserve(exps.size());
    for (const auto& e : exps) probs.push_back(e / total);

    int correct_idx = std::max_element(y_onehot.begin(), y_onehot.end()) - y_onehot.begin();
    auto p_correct = probs[correct_idx] + std::make_shared<Value>(1e-12);
    return log(p_correct) * std::make_shared<Value>(-1.0);
}

int main(int argc, char* argv[]) {
    const auto training_images = loadImages("../../training-data/train-images.idx3-ubyte");
    const auto training_labels = loadLabels("../../training-data/train-labels.idx1-ubyte");
    const auto x_test = loadImages("../../training-data/t10k-images.idx3-ubyte");
    const auto y_test = loadLabels("../../training-data/t10k-labels.idx1-ubyte");

    const auto net = Network({784, 64, 64, 10});
    const double learning_rate = 0.01;
    const auto t0 = std::chrono::high_resolution_clock::now();

    int batch_size = (argc  > 1) ? std::atoi(argv[1]) : 1000;

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < batch_size; j++) {
            net.zero_grad();

            auto pixels = toValuePtr(training_images.at(j));
            auto prediction = net(pixels);
            auto loss_value = softmax_loss(prediction, training_labels.at(j));
            loss_value->backward();
            net.update_parameter(learning_rate);
            if (j % 500 == 0) {
                std::cout << "epoch " << i << ", image " << j << ", loss = " << loss_value->data << std::endl;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();




    int correct = 0;

    for (size_t j = 0; j < x_test.size(); j++) {
        if (j % 500 == 0) {
            std::cout << "testing image " << j << "/" << x_test.size() << std::endl;
        }
        auto pixels = toValuePtr(x_test.at(j));
        auto prediction_raw = net(pixels);

        int predicted = 0;
        double max_val = prediction_raw[0]->data;
        for (int i = 1; i < 10; i++) {
            if (prediction_raw.at(i)->data > max_val) {
                max_val = prediction_raw.at(i)->data;
                predicted = i;
            }
        }

        int actual = std::max_element(y_test.at(j).begin(), y_test.at(j).end()) - y_test.at(j).begin();

        if (predicted == actual) {
            correct++;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::ofstream outFile("results.csv", std::ios::app);

    if (outFile.is_open()) {
        // (TrainingSize, TrainTime, TestTime, Accuracy)
        outFile << batch_size << ","
                <<  std::chrono::duration<double>(t1 - t0).count() << ","
                << std::chrono::duration<double>(t2 - t1).count() << ","
                << (100.0 * correct / x_test.size()) << std::endl;

        outFile.close();
    }
    std::cout << "Training time: " << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    std::cout << "Test accuracy: " << (100.0 * correct / x_test.size()) << "%" << std::endl;
    std::cout << "Testing time: " << std::chrono::duration<double>(t2 - t1).count() << "s" << std::endl;
    return 0;
}