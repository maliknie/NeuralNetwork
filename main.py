import numpy as np
import pandas as pd

# Load and preprocess the dataset
dataset = pd.read_csv("train.csv")
target = dataset["label"]
data = dataset.drop("label", axis=1)

# Normalize input data
X = np.array(data) / 255.0  # Scale pixel values to [0, 1]
y = np.eye(10)[np.array(target)]  # One-hot encoding of labels

# Activation Functions
class Activation:
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

# Layer Class
class Layer:
    def __init__(self, input_dim, output_dim, activation_func):
        # He initialization for ReLU activations
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func  # Activation function (e.g., activation.relu)
    
    def forward(self, input_data):
        self.input = input_data  # Store input for use in backpropagation
        self.output_pre_activation = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func(self.output_pre_activation)
        return self.output

# Network Class
class Network:
    def __init__(self):
        self.layers = []
        self.activation = Activation()
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X  # Final output of the network
        return self.output
    
    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        # Cross-entropy loss
        epsilon = 1e-8  # Small constant to prevent log(0)
        log_preds = np.log(predictions + epsilon)
        loss = -np.mean(np.sum(y * log_preds, axis=1))
        return loss
    
    def backward(self, X, y, learning_rate):
        backpropagation(self, X, y, learning_rate)
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
    
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)
    
            loss = self.calculate_loss(X, y)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

# Backpropagation Function
def backpropagation(network, X, y, learning_rate):
    predictions = network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers
    
    # Calculate delta for the output layer
    delta = predictions - y  # Derivative of cross-entropy loss with softmax
    
    for i in reversed(range(num_layers)):
        layer = network.layers[i]
        
        if i == num_layers - 1:
            # Output layer delta
            layer_deltas[i] = delta
        else:
            next_layer = network.layers[i + 1]
            activation_derivative = network.activation.relu_derivative(layer.output_pre_activation)
            delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * activation_derivative
            layer_deltas[i] = delta
        
        input_to_layer = X if i == 0 else network.layers[i - 1].output
        dLoss_dWeights = np.dot(input_to_layer.T, layer_deltas[i]) / X.shape[0]
        dLoss_dBias = np.sum(layer_deltas[i], axis=0, keepdims=True) / X.shape[0]
        
        # Update weights and biases
        layer.weights -= learning_rate * dLoss_dWeights
        layer.bias -= learning_rate * dLoss_dBias

# Initialize the network and add layers
activation = Activation()
network = Network()

network.add(Layer(784, 128, activation.relu))     # Hidden layer 1
network.add(Layer(128, 64, activation.relu))      # Hidden layer 2
network.add(Layer(64, 10, activation.softmax))    # Output layer

# Train the network
network.train(X, y, epochs=10, learning_rate=0.001, batch_size=32)
