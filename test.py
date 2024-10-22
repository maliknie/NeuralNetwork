import numpy as np
import pandas as pd
import os
import pickle

LOAD_PARAMS = False

# Load and preprocess the dataset
dataset = pd.read_csv("train.csv")
target = dataset["label"]
data = dataset.drop("label", axis=1)

# Normalize input data
X = np.array(data) / 255.0  # Scale pixel values to [0, 1]
y = np.eye(10)[np.array(target)]  # One-hot encoding of labels

# Activation Functions
class relu:
    def value(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)

class softmax:
    def value(self, x):
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Invalid values in softmax input.")
        
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def derivative(self, x):
        return x * (1 - x)

class sigmoid:
    def value(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
class tanh:
    def value(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - x ** 2

# Layer Class
class Layer:
    def __init__(self, input_dim, output_dim, activation_func):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)  # He initialization for ReLU
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func  # Activation function instance
    
    def forward(self, input_data):
        self.input = input_data  # Store input for use in backpropagation
        self.output_pre_activation = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func.value(self.output_pre_activation)
        return self.output

# Network Class
class Network:
    def __init__(self):
        self.layers = []

    def save_params(self, file_path):
        params = {
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.bias for layer in self.layers]
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = params['weights'][i]
            layer.bias = params['biases'][i]
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        epsilon = 1e-8  # Small constant to prevent log(0)
        log_preds = np.log(predictions + epsilon)
        loss = -np.mean(np.sum(y * log_preds, axis=1))
        return loss
    
    def backward(self, X, y, learning_rate):
        backpropagation(self, X, y, learning_rate)
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        if LOAD_PARAMS:
            try:
                self.load_params('params.bin')
                loss = self.calculate_loss(X, y)
                print(f'Initial Loss: {loss:.4f}')
            except:
                print('Error loading parameters. Training from scratch.')

        num_samples = X.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
    
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)
    
            loss = self.calculate_loss(X, y)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
            if epoch == epochs-1:
                self.save_params('params.bin')

    def test(self, X, y):
        predictions = self.forward(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return accuracy

# Backpropagation function with gradient clipping
def backpropagation(network, X, y, learning_rate):
    predictions = network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers
    tmp_learning_rate = learning_rate

    # Calculate delta for the output layer
    delta = predictions - y  # Derivative of cross-entropy loss with softmax
    
    for i in reversed(range(num_layers)):
        layer = network.layers[i]
        
        if i == num_layers - 1:
            layer_deltas[i] = delta
        else:
            next_layer = network.layers[i + 1]
            activation_derivative = layer.activation_func.derivative(layer.output_pre_activation)
            delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * activation_derivative
            layer_deltas[i] = delta
        
        input_to_layer = X if i == 0 else network.layers[i - 1].output
        dLoss_dWeights = np.dot(input_to_layer.T, layer_deltas[i]) / X.shape[0]
        dLoss_dBias = np.sum(layer_deltas[i], axis=0, keepdims=True) / X.shape[0]

        np.clip(dLoss_dWeights, -1, 1, out=dLoss_dWeights)
        np.clip(dLoss_dBias, -1, 1, out=dLoss_dBias)

        layer.weights -= tmp_learning_rate * dLoss_dWeights
        layer.bias -= tmp_learning_rate * dLoss_dBias

# Initialize the network and add layers
network = Network()

# Define activation functions
activation_functions = {
    'relu': relu(),
    'sigmoid': sigmoid(),
    'tanh': tanh(),
    'softmax': softmax()
}

# Choose activation functions for layers
network.add(Layer(784, 128, activation_functions['relu']))  # Hidden layer 1
network.add(Layer(128, 64, activation_functions['relu']))   # Hidden layer 2
network.add(Layer(64, 10, activation_functions['softmax']))  # Output layer

network.train(X, y, epochs=25, learning_rate=0.01, batch_size=32)

# Load the test dataset
test_dataset = pd.read_csv("test.csv")
test_target = test_dataset["label"]
test_data = test_dataset.drop("label", axis=1)
X_test = np.array(test_data) / 255.0
y_test = np.eye(10)[np.array(test_target)]

print("Test accuracy:", network.test(X_test, y_test))
