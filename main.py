import numpy as np
import pandas as pd
import os
import pickle

LOAD_PARAMS = True

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
       # Check for NaN or Inf in x
        if np.isnan(x).any() or np.isinf(x).any():
            print("NaN or Inf detected in input to softmax.")
            print(x)
            raise ValueError("Invalid values in softmax input.")
        
        # Safe softmax computation
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

# Layer Class
class Layer:
    def __init__(self, input_dim, output_dim, activation_func, load_params=False):
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

    def save_params(self, file_path):
        fd = os.open(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    
        # Create a dictionary to store weights and biases
        params = {
            'weights': [layer.weights for layer in network.layers],
            'biases': [layer.bias for layer in network.layers]
        }
        
        # Serialize the parameters using pickle
        serialized_params = pickle.dumps(params)
        
        # Write the serialized data to the file
        os.write(fd, serialized_params)
        
        # Close the file
        os.close(fd)

        print(f"Parameters saved to {file_path}")

    def load_params(self, file_path):
        # Open the file in binary read mode
        fd = os.open(file_path, os.O_RDONLY)
        
        # Read the entire file content
        serialized_params = os.read(fd, os.path.getsize(file_path))
        
        # Deserialize the parameters using pickle
        params = pickle.loads(serialized_params)
        
        # Assign the loaded weights and biases back to the network layers
        for i, layer in enumerate(network.layers):
            layer.weights = params['weights'][i]
            layer.bias = params['biases'][i]
        
        # Close the file
        os.close(fd)

        print(f"Parameters loaded from {file_path}")
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
        if LOAD_PARAMS:
            self.load_params('params.bin')
            loss = self.calculate_loss(X, y)
            print(f'Initial Loss: {loss:.4f}')
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
            if epoch == epochs-1:
                self.save_params('params.bin')
                print('Parameters saved')
    def test(self, X, y):
        for i in range(len(X)):
            #print(f"Predicted: {np.argmax(self.forward(X[i]))}, Actual: {np.argmax(y[i])}")
            pass
        predictions = self.forward(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return accuracy



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
network.train(X, y, epochs=20, learning_rate=0.01, batch_size=32)


# Load the test dataset
test_dataset = pd.read_csv("test.csv")
test_target = test_dataset["label"]
test_data = test_dataset.drop("label", axis=1)
X = np.array(test_data) / 255.0
y = np.eye(10)[np.array(test_target)]

print(network.test(X, y))