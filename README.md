# whats-an-nn: Building a Neural Network from Scratch

## Overview
At first, the goal of this project was to simply create a neural network in python without using any external libraries other than numpy to manually calculate gradients. My curiosity then led me to explore creating an autograd engine, which is inspired by Andrej Karpathy's micrograd tutorial. Afterwards, I decided to port to C++ to observe how much the interpreter overhead really affects performance. Finally, I discovered that libraries such as pytorch use vector based architecture rather than my naive scalar implementation, prompting me to further optimize in that respect. As a final optimization step, I plan to introduce CUDA in order to further increase performance.

## Stages
- Python Scalar Autograd
- C++ Scalar Autograd
- C++ Matrix Autograd
- CUDA (Not yet implemented)

## Neural Network Architecture
Each neural network consisted of 4 layers, with each layer containing [784,64,64,10] neurons respectively. Each layer was passed through a ReLU activation function and the final layer's loss was calculated through softmax cross-entropy. 

## Benchmarks
All benchmarks were performed on the following hardware:
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

## Key Takeaways
- **Scalar Autograd:** Every operation that takes place during activation can be decomposed into a directed acyclic graph. To perform backpropagation all that's required is a topological traversal of the graph.
- **C++ Speedup:** Implementing the naive scalar architecture into C++ resulted in a ~27x speedup without any algorithmic changes.
- **Scalar vs Vector:** Moving from scalar to matrix based operations resulted in a 100x speedup. I hadn't realized that the overhead to creating so many Value object would be _that_ performance intensive. 

## Remaining paths to explore
- **CUDA:** The biggest step that can be taken for further optimization is the introduction of parallelization. The **vast** majority of operations that this neural network performs are simply matrix multiplications and additions. Implementing parallel computation and more optimal matrix algorithms (not just the ones I took in my freshman linear algebra class) should theoretically introduce a large performance boost. 
