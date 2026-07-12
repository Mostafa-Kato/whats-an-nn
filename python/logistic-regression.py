import matplotlib.pyplot as plt
import numpy as np

cluster0 = np.random.randn(10,2)
cluster1 = np.random.randn(10,2)+5

x = np.vstack([cluster0, cluster1])
y = np.array([0]*10 + [1] * 10)


w1 = 0
w2 = 0
b = 0

x1 = x[:,0]
x2 = x[:,1]

z = w1*x1 + w2*x2 + b

learning_rate = 0.01

def sigmoid(input):
    return 1 / (1+np.exp(-input))

def loss(Y, prediction):
    return -np.mean(Y * np.log(prediction) + (1-Y) * np.log(1-prediction))

for i in range(1000):
    z = w1*x1 + w2*x2 + b


    dldw1 = np.mean((sigmoid(z) - y) * x1)
    dldw2 = np.mean((sigmoid(z) - y) * x2)
    dldb = np.mean(sigmoid(z) - y)

    w1 -= dldw1 * learning_rate
    w2 -= dldw2 * learning_rate
    b -= dldb * learning_rate

    if i % 100 == 0:
        print(f"iteration {i}: loss= {loss(y, sigmoid(z))}")

predictions = np.round(sigmoid(z))
plt.scatter(x1, x2, c=predictions)
plt.show()