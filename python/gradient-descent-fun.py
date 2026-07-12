import matplotlib.pyplot as plt, numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2,4,6,8,10,12,14,16,18,20])


def loss(inputs, outputs):
    return np.mean(np.square(outputs - inputs))


m = 0
b = 0
for i in range(10000):
    learning_rate = 0.01
    prediction = x * m + b

    dldm = np.mean(2 * (y - prediction) * (-x) )
    dldb = np.mean(-2 * (y - prediction))
    m -= learning_rate * dldm
    b -= learning_rate * dldb

    if i % 100 == 0:
        print(f"iteration {i}: loss={loss(prediction, y)}, m={m}, b={b}")

plt.scatter(x,y)
plt.plot(x, m*x + b)
plt.show()