import math
import random
import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")
target = dataset["label"]
data = dataset.drop("label", axis=1)

e = math.e
#np.random.seed(420)


X = np.array(data)

y = np.array(target)
y = np.eye(10)[y]


class Layer:
    def __init__(self, input_dim, output_dim): # output_dim = number of neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.array([[random.randint(1, 999)/1000 for _ in range(input_dim)]for _ in range(output_dim)])
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

class Activation:
    def forward(self, input):
        self.output = np.maximum(0, input)

class Network:
    def __init__(self, *layers: tuple):
        self.layers = []
        for layer in layers:
            self.layers.append(Layer(layer[0], layer[1]))
        self.activation_object = Activation()

    def add(self, layer):
        self.layers.append(layer)
    
    def print_network(self):
        for i, layer in enumerate(self.layers):
            print("Layer {i} takes {layer.input_dim} inputs and has {layer.output_dim} neurons")

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            X = self.activation_object.forward(X)
        return X
    
    def backward(self, X, y):
        pass
            
"""
layer1 = Layer(784, 3) # 784 input features, 3 neurons
layer2 = Layer(3, 2) # 3 input features, 2 neurons
layer3 = Layer(2, 10) # 2 input features, 1 neuron
activation1 = Activation()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
layer3.forward(activation1.output)
activation1.forward(layer3.output)

prediction = activation1.output

loss = np.sum((prediction - y)**2)
cross_entropy_loss = -np.sum(y * np.log(prediction + 1e-10))
def backpropagation(loss, prediction, y, activation, layer):
    dLoss_dPrediction = -2*(y - prediction)
    dPrediction_dActivation = 1
    dActivation_dLayer = np.where(layer.output > 0, 1, 0)
    dLayer_dWeights = activation

    dLoss_dWeights = np.dot(dLayer_dWeights.T, dLoss_dPrediction * dPrediction_dActivation * dActivation_dLayer)
    dLoss_dBias = dLoss_dPrediction * dPrediction_dActivation * dActivation_dLayer
    return dLoss_dWeights, dLoss_dBias

print("Loss of the network:")
print("MSE: ", loss)
print("Cross Entropy: ", cross_entropy_loss)"""

network = Network((784, 3), (3, 2), (2, 10))