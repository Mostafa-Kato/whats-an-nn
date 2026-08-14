//
// Created by mkato on 8/12/26.
//
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include "Network.cuh"
#include "mnist-loader.h"

TensorPtr toTensor(const vector<double>& inputs) {
    return std::make_shared<Tensor>(
        std::make_shared<Matrix>(static_cast<int>(inputs.size()), 1, inputs)
    );
}

TensorPtr softmaxLoss(const TensorPtr& predictions, const vector<double>& y_onehot) {
    auto yMatrix = std::make_shared<Matrix>(predictions->data->rows, 1, y_onehot);
    auto data = predictions->data;
    // find the maximum and subtract it from each entry
    double max = matMaxValue(data);
    auto exps = matExp(scalMatAdd(data, max*-1));
    double sum = matSum(exps);

    //Compute softmax prob
    auto probs = exps/sum;

    //Compute loss from correct answer
    auto correctIndex = std::max_element(y_onehot.begin(), y_onehot.end()) - y_onehot.begin();
    double correctProb;
    cudaMemcpy(&correctProb, probs->data + correctIndex, sizeof(double), cudaMemcpyDeviceToHost);
    double lossVal = -std::log(correctProb + 1e-12);

    //Create loss tensor
    auto lossTensor = std::make_shared<Tensor>(
        std::make_shared<Matrix>(1, 1, vector{lossVal}),
        vector{predictions}
    );

    double one = 1.0;
    cudaMemcpy(lossTensor->grad->data, &one, sizeof(double), cudaMemcpyHostToDevice);

    Tensor* lossPtr = lossTensor.get();
    lossTensor->back = [predictions, probs, yMatrix, lossPtr] () {
        auto gradMatrix = std::make_shared<Matrix>(predictions->data->rows, 1);

        gradMatrix = matAdd(probs, yMatrix*-1);

        double upStreamGrad;
        cudaMemcpy(&upStreamGrad,lossPtr->grad->data, sizeof(double), cudaMemcpyDeviceToHost);
        predictions->grad = matAdd(predictions->grad ,gradMatrix * upStreamGrad);
    };

    return lossTensor;
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
            auto loss_value = softmaxLoss(prediction, training_labels.at(j));
            double lossDouble;
            cudaMemcpy(&lossDouble, loss_value->data->data, sizeof(double), cudaMemcpyDeviceToHost);
            loss_value->backward();
            net.update_parameter(learning_rate);
            if (j % 500 == 0) {
                std::cout << "epoch " << i << ", image " << j << ", loss = " << lossDouble << std::endl;
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

        std::vector<double> h_pred(10);
        cudaMemcpy(h_pred.data(), prediction_raw->data->data, 10 * sizeof(double), cudaMemcpyDeviceToHost);

        int predicted = std::max_element(h_pred.begin(), h_pred.end()) - h_pred.begin();

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