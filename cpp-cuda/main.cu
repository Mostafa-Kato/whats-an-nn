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
           batch[i*inputSize + j] = input[i + iteration*batchSize][j];
        }
    }
    return batch;
}

TensorPtr softmaxLoss(const TensorPtr& predictions, const vector<double>& y_onehot) {
    auto yMatrix = std::make_shared<Matrix>(predictions->data->rows, 1, y_onehot);
    auto data = predictions->data;
    // find the maximum and subtract it from each entry
    double max = matMaxValue(data);
    auto exps = matExp(scalMatAdd(data, max*-1));
    double sum = matSumAxis(exps);

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
    const auto trainingImages = loadImages("../../training-data/train-images.idx3-ubyte");
    const auto trainingLabels = loadLabels("../../training-data/train-labels.idx1-ubyte");
    const auto xTest = loadImages("../../training-data/t10k-images.idx3-ubyte");
    const auto yTest = loadLabels("../../training-data/t10k-labels.idx1-ubyte");

    auto numImages = trainingImages.size();
    const int batchSize = (argc  > 1) ? std::atoi(argv[1]) : 32;
    const int epochSize = 5;
    const int numIterations = numImages / batchSize;

    const auto net = Network({784 * batchSize, 64, 64, 10});
    const double learningRate = 0.01;
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < epochSize; i++) {
       for (int j = 0; j < numIterations; j++) {
           vector<double> batch = loadBatch(trainingImages, 784, batchSize, j);
           vector<double> batchLabels = loadBatch(trainingLabels, 10, batchSize, j);
           MatrixPtr batchMat = std::make_shared<Matrix>(batchSize, 784, batch);
           TensorPtr batchTsr = std::make_shared<Tensor>(batchMat);
           auto prediction = net(batchTsr);
           auto lossValue = softmaxLoss(prediction, batchLabels);
           lossValue->backward();
       }
    }


    return 0;
}