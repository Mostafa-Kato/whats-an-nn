cmake_minimum_required(VERSION 3.20)
project(cpp_scalar)
set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_scalar
        main.cpp
        Value.cpp
        Neuron.cpp
        Layer.cpp
        Network.cpp
        mnist-loader.cpp
)