import math
import numpy as np

e = math.e
#np.random.seed(420)

X = [[i] for i in range(100)] # 100 samples, 1 feature each
X = np.array(X)

y = [[i**2] for i in range(100)] # 100 samples, 1 target each
y = np.array(y)

class Layer:
    def __init__(self, input_dim, output_dim): # output_dim = number of neurons
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

class Activation:
    def forward(self, input):
        self.output = np.maximum(0, input)

layer1 = Layer(1, 3) # 1 input features, 3 neurons
layer2 = Layer(3, 2) # 3 input features, 2 neurons
layer3 = Layer(2, 1) # 2 input features, 1 neuron
activation1 = Activation()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
layer3.forward(activation1.output)
activation1.forward(layer3.output)

print("Output of the network:")
print(activation1.output)