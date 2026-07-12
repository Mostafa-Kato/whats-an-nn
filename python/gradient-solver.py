import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.datasets import fetch_openml


class Value:
    def __init__(self, data, _child=(), _op=''):
        self.data = data
        self.grad = 0
        self._back = lambda : None
        self._prev = set(_child)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = self.num_to_val(other)
        result = Value(self.data + other.data, (self, other), '+')
        def add_grad():
            self.grad += result.grad
            other.grad += result.grad
        result._back = add_grad
        return result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self.num_to_val(other)
        result = Value(self.data - other.data, (self,other), '-')
        def sub_grad():
            self.grad += result.grad
            other.grad -= result.grad
        result._back = sub_grad
        return result

    def __rsub__(self, other):
        other = self.num_to_val(other)
        return other - self

    def __mul__(self, other):
        other = self.num_to_val(other)
        result = Value(self.data * other.data, (self, other), '*')
        def mul_grad():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._back = mul_grad
        return result

    def __rmul__(self,other):
        return self * other

    def __truediv__(self, other):
        other = self.num_to_val(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, power, modulo=None):
        result = Value(self.data ** power, (self,), '**')
        def pow_grad():
            self.grad += power * (self.data ** (power - 1)) * result.grad
        result._back = pow_grad
        return result

    def num_to_val(self, number):
        if isinstance(number, Value):
            return number
        else:
            return Value(number, (), '')

    def relu(self):
        result = Value(max(0, self.data), (self,), 'relu')
        def relu_grad():
            if result.data > 0:
                self.grad += 1 * result.grad
            else:
                self.grad += 0
        result._back = relu_grad
        return result

    def exp(self):
        out_data = math.exp(self.data)
        result = Value(out_data, (self,), 'exp')

        def exp_grad():
            self.grad += out_data * result.grad

        result._back = exp_grad
        return result

    def log(self):
        out_data = math.log(self.data)
        result = Value(out_data, (self,), 'log')

        def log_grad():
            self.grad += (1 / self.data) * result.grad

        result._back = log_grad
        return result

    def __neg__(self):
        return self * -1

    def backward(self):
        visited = set()
        order = []
        def visit(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    visit(child)
                order.append(n)
        visit(self)

        self.grad = 1
        for node in reversed(order):
            node._back()


class Neuron:
    def __init__(self, num_inputs):
        self._weights = [Value(np.random.randn() * (1 / num_inputs ** 0.5)) for i in range(num_inputs)]
        self._bias = Value(0)

    def __call__(self, inputs, activate=True):
        output = Value(0)
        for i, input in enumerate(inputs):
            output += input * self._weights[i]
        if activate:
            return (output + self._bias).relu()
        return output + self._bias

    def parameters(self):
        return self._weights + [self._bias]

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self._neurons = [Neuron(num_inputs) for i in range(num_neurons)]

    def __call__(self, input_vector, last=False):
        outputs = []
        for neuron in self._neurons:
            if last:
                outputs.append(neuron(input_vector, False))
            else:
                outputs.append(neuron(input_vector))
        return outputs

    def parameters(self):
        parameters = []
        for neuron in self._neurons:
            parameters.extend(neuron.parameters())
        return parameters

class Network:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def __call__(self, inputs):
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                inputs = self.layers[i](inputs, True)
            else:
                inputs = self.layers[i](inputs)
        return inputs

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0

    def update_parameter(self, learning_rate):
        for parameter in self.parameters():
            parameter.data -= parameter.grad * learning_rate




mnist = fetch_openml('mnist_784', version=1)
x = mnist.data.to_numpy() / 255.0
labels = mnist.target.to_numpy().astype(int)
y=np.eye(10)[labels]

net = Network([784,64,64,10])

def loss(y_onehot, prediction):
    correct_idx = np.argmax(y_onehot)
    p_correct = prediction[correct_idx]
    return -(p_correct + 1e-12).log()

def softmax(input):
    maxval = max(input, key=lambda v: v.data)
    shifted = [v - maxval for v in input]
    exp = [v.exp() for v in shifted]
    return [v/sum(exp) for v in exp]

learning_rate = 0.01
x_small = x[:1000]
y_small = y[:1000]
t0 = time.time()
for i in range(1):
    for j, pixels in enumerate(x_small):
        net.zero_grad()
        pixels = [Value(p) for p in pixels]
        prediction_raw = net(pixels)
        prediction = softmax(prediction_raw)
        loss_value = loss(y[j], prediction)
        loss_value.backward()

        net.update_parameter(learning_rate)
        if j % 100 == 0:
            print(f"Iteration: {j}, Loss: {loss_value}")

print(f"time: {time.time() - t0}s")


