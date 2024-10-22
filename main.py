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
    
    def relu_derivative(self, output):
        return (output > 0).astype(float)
    
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

    def train(self, X, y, epochs, learning_rate, batch_size):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)

            loss = self.calculate_loss(X, y)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def backward(self, X, y, learning_rate):
        backpropagation(self, X, y, learning_rate)

# Backpropagation
def backpropagation(network, X, y, learning_rate):
    predictions = network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers

    # Gradient of loss with respect to the output layer input
    delta = predictions - y  # Simplified gradient for softmax + cross-entropy

    for i in reversed(range(num_layers)):
        layer = network.layers[i]

        if i == num_layers - 1:
            # Output layer delta already computed
            layer_deltas[i] = delta
        else:
            next_layer = network.layers[i + 1]
            activation_derivative = network.activation_object.relu_derivative(layer.output)
            delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * activation_derivative
            layer_deltas[i] = delta

        input_to_layer = X if i == 0 else network.layers[i - 1].output
        dLoss_dWeights = np.dot(input_to_layer.T, layer_deltas[i]) / X.shape[0]
        dLoss_dBias = np.sum(layer_deltas[i], axis=0, keepdims=True) / X.shape[0]

        # Update weights and biases
        layer.weights -= learning_rate * dLoss_dWeights
        layer.bias -= learning_rate * dLoss_dBias

def calculate_loss(self, X, y):
    predictions = self.forward(X)
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-8
    log_preds = np.log(predictions + epsilon)
    loss = -np.mean(np.sum(y * log_preds, axis=1))
    return loss

# Define network and layers
activation = Activation()
network = Network()

# Adjust number of neurons if needed
network.add(Layer(784, 128, activation.relu))  # Hidden layer 1
network.add(Layer(128, 64, activation.relu))  # Hidden layer 2
network.add(Layer(64, 10, activation.softmax))  # Output layer

# Train the network with a lower learning rate
network.train(X, y, epochs=100, learning_rate=0.001, batch_size=32)
