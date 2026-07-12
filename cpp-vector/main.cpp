#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include "Network.h"
#include "mnist-loader.h"

TensorPtr toTensor(const vector<double>& inputs) {
    return std::make_shared<Tensor>(
        std::make_shared<Matrix>(inputs, static_cast<int>(inputs.size()), 1)
    );
}

TensorPtr softmax_loss(const TensorPtr& predictions, const vector<double>& y_onehot) {
    auto& pdata = predictions->data->data;
    double maxval = *std::max_element(pdata.begin(), pdata.end());

    vector<double> exps(pdata.size());
    double sum_exps = 0.0;
    for (size_t i = 0; i < pdata.size(); ++i) {
        exps[i] = std::exp(pdata[i] - maxval);
        sum_exps += exps[i];
    }

    vector<double> probs(pdata.size());
    for (size_t i = 0; i < pdata.size(); ++i) {
        probs[i] = exps[i] / sum_exps;
    }

    int correct_idx = std::max_element(y_onehot.begin(), y_onehot.end()) - y_onehot.begin();
    double loss_val = -std::log(probs[correct_idx] + 1e-12);

    auto loss_tensor = std::make_shared<Tensor>(
        std::make_shared<Matrix>(vector<double>{loss_val}, 1, 1),
        vector<TensorPtr>{predictions},
        "softmax_crossentropy"
    );

    Tensor* loss_ptr = loss_tensor.get(); // Raw pointer to avoid shared_ptr circular dependency leak
    loss_tensor->back = [predictions, probs, y_onehot, loss_ptr]() {
        vector<double> grad_data(probs.size());

        for (size_t i = 0; i < probs.size(); ++i) {
            grad_data[i] = probs[i] - y_onehot[i];
        }

        auto grad_matrix = std::make_shared<Matrix>(grad_data, predictions->data->rows, 1);

        double upstream_grad = loss_ptr->grad->data[0];
        predictions->grad = predictions->grad + grad_matrix * upstream_grad;
    };

    return loss_tensor;
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

            // Using .at() instead of [] forces C++ to check bounds and scream if it fails
            auto pixels = toTensor(training_images.at(j));
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
        auto pixels = toTensor(x_test.at(j));
        auto prediction_raw = net(pixels);

        int predicted = 0;
        double max_val = prediction_raw->data->data[0];
        for (int i = 0; i < 10; i++) {
            if (prediction_raw->data->data[i] > max_val) {
                max_val = prediction_raw->data->data[i];
                predicted = i;
            }
        }

        int actual = std::max_element(y_test.at(j).begin(), y_test.at(j).end()) - y_test.at(j).begin();

        if (predicted == actual) {
            correct++;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::ofstream outFile("results_tensor.csv", std::ios::app);

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