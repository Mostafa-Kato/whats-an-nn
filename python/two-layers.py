import numpy as np
import matplotlib.pyplot as plt

cluster0 = np.random.randn(10,2)
cluster1 = np.random.randn(10,2)+5

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0]).reshape(-1,1)

x1 = x[:,0]
x2 = x[:,1]

h = 3 #Hidden Size




W1 = np.random.randn(2,h)
b1 = np.random.randn(h)
W2 = np.random.randn(h, 1)
b2 = 0

def sigmoid(input):
    return 1 / (1+np.exp(-input))

def loss(input):
    return -np.mean(y * np.log((input)) + (1-y) * np.log((1-input)))

z1 = x @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

learning_rate = 1

y=y.reshape(-1,1)
for i in range(10000):
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    dz2 = a2 - y
    dW2 = a1.T @ dz2 / len(x)
    db2 = np.mean(dz2)

    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)
    dW1 = x.T @ dz1 / len(x)
    db1 = np.sum(dz1, 0) / len(x)

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    if i % 100 == 0:
        print(f"iteration {i}")

predictions = np.round(a2)
plt.scatter(x1, x2, c=predictions)
plt.show()

