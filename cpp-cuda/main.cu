//
// Created by mkato on 8/12/26.
//
#include "Matrix.cuh"

int main() {
    int arows = 2;
    int acols = 3;
    int brows = 3;
    int bcols = 2;

    vector<double> a = {-1,2,-3,-4,-5,6};
    vector<double> b = {1,2,3,4,5,6};

    const auto matA = std::make_shared<Matrix>(arows, acols, a);
    const auto matB = std::make_shared<Matrix>(brows, bcols, b);

    auto matC = matPow(matA, 2);

    printMatrix(matC);

    return 0;
}