import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
x = mnist.data.to_numpy() / 255.0
labels = mnist.target.to_numpy().astype(int)

y = np.eye(10)[labels]

def softmax(input):
    shifted = input - np.max(input, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(input):
    return np.maximum(0,input)

def loss(y, predictions):
    return -np.mean(np.sum(y * np.log(predictions), axis=1))

h=64
W1 = np.random.randn(784,h) * np.sqrt(2/784)
b1 = np.random.randn(h)
W2 = np.random.randn(h,h) * np.sqrt(2/h)
b2 = np.random.randn(h)
W3 = np.random.randn(h,10) * np.sqrt(2/h)
b3 = 0

z1 = x @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
a2 = relu(z2)

z3 = a2 @ W3 + b3
a3 = softmax(z3)

learning_rate = 0.01
batch_size = 64

for i in range(100000):
    batch = np.random.choice(len(x), batch_size, replace=False)
    x_batch = x[batch]
    y_batch = y[batch]

    z1 = x_batch @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    a3 = softmax(z3)

    dz3 = a3 - y[batch]
    dW3 = a2.T @ dz3 / len(x_batch)
    db3 = np.mean(dz3,0)

    da2 = dz3 @ W3.T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2 / len(x_batch)
    db2 = np.mean(dz2,0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = x_batch.T @ dz1 / len(x_batch)
    db1 = np.sum(dz1, 0) / len(x_batch)

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    W3 -= learning_rate * dW3
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    b3 -= learning_rate * db3

    if i % 100 == 0:
        print(f"iteration {i}, loss {loss(y_batch,a3)}")

z1_test = x[:5000] @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = relu(z2_test)
z3_test = a2_test @ W3 + b3
a3_test = softmax(z3_test)

predictions = np.argmax(a3_test, axis=1)
true_labels = np.argmax(y[:5000], axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"accuracy: {accuracy}")