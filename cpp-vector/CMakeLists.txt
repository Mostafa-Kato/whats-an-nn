cmake_minimum_required(VERSION 3.20)
project(cpp_vectorized)
set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_vectorized
        main.cpp
        Matrix.cpp
        Tensor.cpp
        Layer.cpp
        Network.cpp
        mnist-loader.cpp
)