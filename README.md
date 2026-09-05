# whats-an-nn: Building a Neural Network from Scratch

## Overview
At first, the goal of this project was to simply create a neural network in python without using any external libraries other than numpy to manually calculate gradients. My curiosity then led me to explore creating an autograd engine, which is inspired by Andrej Karpathy's micrograd tutorial. Afterwards, I decided to port to C++ to observe how much the interpreter overhead really affects performance. Finally, I discovered that libraries such as pytorch use vector based architecture rather than my naive scalar implementation, prompting me to further optimize in that respect. As a final optimization step, I introduced batching and parallelization through CUDA.

## Stages
- Python Scalar Autograd
- C++ Scalar Autograd
- C++ Matrix Autograd
- CUDA Autograd

## Neural Network Architecture
Each neural network consisted of 4 layers, with each layer containing [784,64,64,10] neurons respectively. Each layer was passed through a ReLU activation function and the final layer's loss was calculated through softmax cross-entropy. 

## Scalar vs Vector Benchmarks
The following benchmarks were performed on the following hardware:
AMD Ryzen 5 3600
16 GB Ram
RTX 4060 8GB

### Speed Test Results (Time to train on 1000 images)
| Stage | Training Time (s) |
|---|---|
| Python Scalar | 1990 |
| C++ Scalar | 75 |
| C++ Vectorized | 0.76 |           

![Speed Comparison](Images/Training%20Time%20Comparison.png)


### Accuracy Vs Training Size (Identical seed for both C++ stages ensured same accuracy, Python omitted due to performance)
| Number of Training Images | Testing Accuracy |
|---|---|
| 1000 | 65.96% |
| 2000 | 85.69% |
| 4000 | 88.35% |
| 8000 | 88.42% |
| 16000 | 93.56% |
| 32000 | 94.16% |
| 59000 | 94.82% |

![Accuracy vs Training Size](Images/Testing%20Accuracy%20vs.%20Training%20Size.png)

## CPU vs CUDA Benchmarks
The following benchmarks were performed on the following hardware:
Intel Core i7-14650HX (2.20 GHz)
32 GB RAM
RTX 4070 Laptop 8GB

### Speed Test Results
#### CPU
| # of Epochs | Training Time (s) |
|---|---|
| 1 | 28.84 |
| 3 | 85.93 | 
| 5 | 142.77 | 
| 10 | 283.36 | 

#### CUDA
| # of Epochs | Training Time (s) |
|---|---|
| 1 | 3.32 | 
| 3 | 7.04 |
| 5 | 10.16 | 
| 10 | 18.52 | 
| 100 | 171.86 | 

![Speed Comparison](Images/CPU%20vs.%20GPU%Total%20Time.png)

### Accuracy Test Results
#### CPU
| # of Epochs | Accuracy |
|---|---|
| 1 | 94.82 |
| 3 | 95.42 | 
| 5 | 96.31 | 
| 10 | 97.14 | 

#### CUDA
| # of Epochs | Accuracy |
|---|---|
| 1 | 89.94 | 
| 3 | 93.01 |
| 5 | 94.21 | 
| 10 | 95.8 | 
| 100 | 97.36 | 

## Key Takeaways
- **Scalar Autograd:** Every operation that takes place during activation can be decomposed into a directed acyclic graph. To perform backpropagation all that's required is a topological traversal of the graph.
- **C++ Speedup:** Implementing the naive scalar architecture into C++ resulted in a ~27x speedup without any algorithmic changes.
- **Scalar vs Vector:** Moving from scalar to matrix based operations resulted in a 100x speedup. I hadn't realized that the overhead to creating so many Value object would be _that_ performance intensive.
- **CUDA:** CUDA presented me with by far the largest challenge in terms of implementation. Wrapping my mind around parallelized operations certainly had a learning curve, but the results speak for themselves. CUDA provided a roughly 15x speedup at 10 epochs, but at a slightly worse performance. If we account for the time to train at a similar accuracy, it is still 1.6x faster to use CUDA.

