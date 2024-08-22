import numpy as np

X = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])

class Layer:
    def __init__(self, input_dim, output_dim): # output_dim = number of neurons
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

class ActivationReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)

layer1 = Layer(2, 3) # 2 input features, 3 neurons
layer2 = Layer(3, 2) # 3 input features, 2 neurons

layer1.forward(X)
print("Layer 1 output:")
print(layer1.output)
print("_________________________")
layer2.forward(layer1.output)
print("Layer 2 output:")
print(layer2.output)
