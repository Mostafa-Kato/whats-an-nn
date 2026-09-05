//
// Created by mkato on 8/12/26.
//
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include "Network.cuh"
#include "mnist-loader.h"

//super inefficient but i need something just to see if understanding works,
//should just rework the actual loader in the future
vector<double> loadBatch(const vector<vector<double>>& input, const int inputSize, const int batchSize, const int iteration) {
    vector<double> batch = vector<double>(inputSize * batchSize);
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < inputSize; j++) {
           batch[j*batchSize + i] = input[i + iteration*batchSize][j];
        }
    }
    return batch;
}

TensorPtr softmaxLoss(const TensorPtr& predictions, const vector<double>& y_onehot, const int batchSize) {
    auto yMatrix = std::make_shared<Matrix>(10, batchSize, y_onehot);
    auto data = predictions->data;
    // find the maximum and subtract it from each entry then exponentiate
    auto max = matMaxAxis(predictions->data, COLS);
    auto exps = matExp(matAddColBroadcast(data, max*-1));
    auto sum = matSumAxis(exps, COLS);
    //Compute softmax prob
    auto probs = matMulColBroadcast(exps, matPow(sum, -1));

    //Compute losses from correct answers
    auto correctProb = matSumAxis(matMulElementWise(probs, yMatrix), COLS); // Result is 1 * batchSize
    auto lossVals = -1*matLog(scalMatAdd(correctProb, 1e-12));
    double lossSum;
    cudaMemcpy(&lossSum , matSumAxis(lossVals, ROWS)->data, sizeof(double), cudaMemcpyDeviceToHost);
    double lossVal = lossSum / batchSize;

    //Create loss tensor
    auto lossTensor = std::make_shared<Tensor>(
        std::make_shared<Matrix>(1, 1, vector{lossVal}),
        vector{predictions}
    );

    double one = 1.0;
    cudaMemcpy(lossTensor->grad->data, &one, sizeof(double), cudaMemcpyHostToDevice);

    Tensor* lossPtr = lossTensor.get();
    lossTensor->back = [predictions, probs, yMatrix, lossPtr, batchSize] () {
        auto gradMatrix = matAdd(probs, yMatrix*-1);
        double upStreamGrad;
        cudaMemcpy(&upStreamGrad,lossPtr->grad->data, sizeof(double), cudaMemcpyDeviceToHost);
        predictions->grad = matAdd(predictions->grad ,gradMatrix * (upStreamGrad / batchSize));
    };

    return lossTensor;
}

int main(int argc, char* argv[]) {
    const auto trainingImages = loadImages("../../training-data/train-images.idx3-ubyte");
    const auto trainingLabels = loadLabels("../../training-data/train-labels.idx1-ubyte");
    const auto xTest = loadImages("../../training-data/t10k-images.idx3-ubyte");
    const auto yTest = loadLabels("../../training-data/t10k-labels.idx1-ubyte");

    auto numImages = trainingImages.size();
    const int batchSize = (argc  > 1) ? std::atoi(argv[1]) : 32;
    const int epochSize = 10;
    const int numIterations = numImages / batchSize;

    const auto net = Network({784, 64, 64, 10});
    const double learningRate = 0.01;
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < epochSize; i++) {
       for (int j = 0; j < numIterations; j++) {
           net.zero_grad();

           vector<double> batch = loadBatch(trainingImages, 784, batchSize, j);
           vector<double> batchLabels = loadBatch(trainingLabels, 10, batchSize, j);
           MatrixPtr batchMat = std::make_shared<Matrix>(784, batchSize, batch);
           TensorPtr batchTsr = std::make_shared<Tensor>(batchMat);
           auto prediction = net(batchTsr);
           auto lossValue = softmaxLoss(prediction, batchLabels, batchSize);
           double lossDouble;
           cudaMemcpy(&lossDouble, lossValue->data->data, sizeof(double), cudaMemcpyDeviceToHost);
           lossValue->backward();
           net.update_parameter(learningRate);

           if (j % 100 == 0) {
               std::cout << "epoch " << i << ", batch " << j << ", loss = " << lossDouble << std::endl;
           }
       }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int correct = 0;
    for (size_t j = 0; j < xTest.size(); j++) {
        if (j % 500 == 0) {
            std::cout << "testing image " << j << "/" << xTest.size() << std::endl;
        }
        auto pixels = std::make_shared<Tensor>(std::make_shared<Matrix>(784, 1, xTest.at(j)));
        auto prediction_raw = net(pixels);

        std::vector<double> h_pred(10);
        cudaMemcpy(h_pred.data(), prediction_raw->data->data, 10 * sizeof(double), cudaMemcpyDeviceToHost);

        int predicted = std::max_element(h_pred.begin(), h_pred.end()) - h_pred.begin();
        int actual = std::max_element(yTest.at(j).begin(), yTest.at(j).end()) - yTest.at(j).begin();

        if (predicted == actual) {
            correct++;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Training time: " << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    std::cout << "Test accuracy: " << (100.0 * correct / xTest.size()) << "%" << std::endl;
    std::cout << "Testing time: " << std::chrono::duration<double>(t2 - t1).count() << "s" << std::endl;

    return 0;
}