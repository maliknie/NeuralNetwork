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
            
def backpropagation(loss, predictions, y, layers):
    # Step 1: Compute the gradient of the loss with respect to the final prediction
    dLoss_dPrediction = -2 * (y - predictions)
    
    # Start with the gradient of the loss with respect to the output of the final layer
    dLoss_dOutput = dLoss_dPrediction
    
    # Initialize an empty list to store gradients for all layers
    gradients = []
    
    # Step 2: Loop through each layer in reverse order (from output to input)
    for i in reversed(range(len(layers))):
        layer = layers[i]
        
        # Derivative of the activation function with respect to the input of this layer
        if i == len(layers) - 1:
            dPrediction_dActivation = 1  # Assuming the final layer's output is linear
        else:
            dActivation_dLayer = np.where(layer.output > 0, 1, 0)  # Assuming ReLU
            
        dLayer_dWeights = layers[i-1].output if i > 0 else X  # X is the input for the first layer
        
        # Chain rule: dLoss/dWeights = dLoss/dOutput * dOutput/dActivation * dActivation/dWeights
        dLoss_dWeights = np.dot(dLayer_dWeights.T, dLoss_dOutput * dActivation_dLayer)
        dLoss_dBias = np.sum(dLoss_dOutput * dActivation_dLayer, axis=0)
        
        # Store the gradients for this layer
        gradients.append((dLoss_dWeights, dLoss_dBias))
        
        # Compute the gradient for the input of the next layer (moving backwards)
        dLoss_dOutput = np.dot(dLoss_dOutput * dActivation_dLayer, layer.weights.T)
    
    # Step 3: Reverse the gradients list to match the order of layers
    gradients.reverse()
    
    # Step 4: Return the gradients for all layers
    return gradients

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

print("Loss of the network:")
print("MSE: ", loss)
print("Cross Entropy: ", cross_entropy_loss)"""

network = Network((784, 3), (3, 2), (2, 10))