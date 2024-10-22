import math
import numpy as np
import pandas as pd

# Dataset
dataset = pd.read_csv("NeuralNetwork/train.csv")
target = dataset["label"]
data = dataset.drop("label", axis=1)

X = np.array(data)
y = np.eye(10)[np.array(target)]  # One-hot encoding

# Activation Functions
class Activation:
    def relu(self, input):
        return np.maximum(0, input)
    
    def softmax(self, predictions):
        exp_values = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))  # For numerical stability
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

# Layer Class
class Layer:
    def __init__(self, input_dim, output_dim, activation):
        # He initialization for ReLU activations
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))  # Initialize bias as zeros
        self.activation = activation

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.activation(self.output)

# Network Class
class Network:
    def __init__(self):
        self.layers = []
        self.activation_object = Activation()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X
        return self.output

    def train(self, X, y, epochs, learning_rate):
        runningloss = 0
        for epoch in range(epochs):
            for i in range(len(X)):
                self.backward(X[i].reshape(1, -1), y[i].reshape(1, -1), learning_rate)
                runningloss += np.sum((y[i] - self.forward(X[i].reshape(1, -1))) ** 2) / len(X)
                if i % 1000 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {runningloss:.4f}')
                    runningloss = 0
            loss = np.sum((y - self.forward(X)) ** 2) / len(X)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def backward(self, X, y, learning_rate):
        backpropagation(self, X, y, learning_rate)

# Backpropagation
def backpropagation(network, X, y, learning_rate):
    network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers

    predictions = network.output
    loss_gradient = 2 * (predictions - y)

    for i in reversed(range(num_layers)):
        layer = network.layers[i]
        
        if i == num_layers - 1:  # Output layer with softmax
            delta = loss_gradient
        else:
            next_layer = network.layers[i + 1]
            delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * (layer.output > 0)  # ReLU derivative

        layer_deltas[i] = delta
        input_to_layer = X if i == 0 else network.layers[i - 1].output
        dLoss_dWeights = np.dot(input_to_layer.T, delta)
        dLoss_dBias = np.sum(delta, axis=0)

        # Update weights and biases with learning rate
        layer.weights -= learning_rate * dLoss_dWeights
        layer.bias -= learning_rate * dLoss_dBias

# Define network and layers
activation = Activation()
network = Network()

# Adjust number of neurons if needed
network.add(Layer(784, 128, activation.relu))  # Hidden layer 1
network.add(Layer(128, 64, activation.relu))  # Hidden layer 2
network.add(Layer(64, 10, activation.softmax))  # Output layer

# Train the network with a lower learning rate
network.train(X, y, epochs=100, learning_rate=0.001)
