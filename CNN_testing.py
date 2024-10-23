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
class ReLU:
    def value(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)

class Softmax:
    def value(self, x):
        if np.isnan(x).any() or np.isinf(x).any():
            print("NaN or Inf detected in input to softmax.")
            print(x)
            raise ValueError("Invalid values in softmax input.")
        
        x = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    def derivative(self, x):
        return x * (1 - x)


class Sigmoid:
    def value(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)

# Layer Class
class Layer:
    def __init__(self, input_dim, output_dim, activation_func):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01  # Use smaller scale
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func  # Activation function instance
    
    def forward(self, input_data):
        self.input = input_data
        self.output_pre_activation = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func.value(self.output_pre_activation)

        # Check for NaN or Inf in output
        if np.isnan(self.output).any() or np.isinf(self.output).any():
            print("NaN or Inf detected in layer output.")
            print(self.output)
            raise ValueError("Invalid values in layer output.")

        return self.output

# Network Class
class Network:
    def __init__(self, kernel_size=[3], pooling_layers=[2]):
        self.kernel_size = kernel_size
        self.pooling_layers = pooling_layers
        self.layers = []

    def save_params(self, file_path):
        with open(file_path, 'wb') as f:
            # Create a dictionary to store weights and biases
            params = {
                'weights': [layer.weights for layer in self.layers],
                'biases': [layer.bias for layer in self.layers]
            }
            # Serialize the parameters using pickle
            pickle.dump(params, f)
            print(f"Parameters saved to {file_path}")

    def load_params(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                # Deserialize the parameters using pickle
                params = pickle.load(f)
                # Assign the loaded weights and biases back to the network layers
                for i, layer in enumerate(self.layers):
                    layer.weights = params['weights'][i]
                    layer.bias = params['biases'][i]
            print(f"Parameters loaded from {file_path}")
        except Exception as e:
            print(f"Error loading parameters: {e}")

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

        # Check for NaN or Inf in loss
        if np.isnan(loss) or np.isinf(loss):
            print("NaN or Inf detected in loss calculation.")
            raise ValueError("Invalid values in loss calculation.")

        return loss
    
    def backward(self, X, y, learning_rate):
        backpropagation(self, X, y, learning_rate)
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        if LOAD_PARAMS:
            try:
                self.load_params('params.bin')
                loss = self.calculate_loss(X, y)
                print(f'Initial Loss: {loss:.4f}')
            except Exception as e:
                print('Error loading parameters. Training from scratch.')

        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                X_processed = np.zeros((X_batch.shape[0], 169))  # Initialize processed output

                for j in range(len(X_batch)):
                    X_processed[j] = self.CNN_preprocessing(X_batch[j])

                self.backward(X_processed, y_batch, learning_rate)
                
                # Optionally calculate loss after each batch
                loss = self.calculate_loss(X_processed, y_batch)

            # Optional to save parameters after each epoch
            loss = self.calculate_loss(X_processed, y_batch)  # Calculate loss on processed batch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
            if epoch == epochs - 1:
                self.save_params('params.bin')
                print('Parameters saved')

    def convolution2d(self, X, kernel):
        kernel_size = kernel.shape[0]
        height, width = X.shape
        output_height = height - kernel_size + 1
        output_width = width - kernel_size + 1
        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(X[i:i + kernel_size, j:j + kernel_size] * kernel)

        return output

    def max_pooling2d(self, X, pool_size):
        height, width = X.shape
        output_height = height // pool_size
        output_width = width // pool_size
        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.max(X[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size])

        return output

    def CNN_preprocessing(self, X):
        kernel_size = self.kernel_size[0]
        pooling_size = self.pooling_layers[0]

        X = X.reshape(28, 28)  # Assuming input is a flattened 28x28 image

        # Apply convolution
        X = self.convolution2d(X, np.ones((kernel_size, kernel_size)) / (kernel_size ** 2))

        # Apply max pooling
        X = self.max_pooling2d(X, pooling_size)

        # Flatten to a vector
        X = X.flatten()

        # Check output shape
        expected_shape = 169  # Change as needed
        if X.shape[0] != expected_shape:
            raise ValueError(f"Unexpected output shape from CNN preprocessing: {X.shape}. Expected shape: ({expected_shape},)")

        return X


# Backpropagation function with gradient clipping
def backpropagation(network, X, y, learning_rate):
    predictions = network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers
    loss = network.calculate_loss(X, y)
    tmp_learning_rate = min(learning_rate, loss * 0.1)

    # Calculate delta for the output layer
    delta = predictions - y  # Derivative of cross-entropy loss with softmax
    
    for i in reversed(range(num_layers)):
        layer = network.layers[i]
        
        if i == num_layers - 1:
            # Backprop through output layer
            layer_deltas[i] = delta * layer.activation_func.derivative(layer.output_pre_activation)
        else:
            # Backprop through hidden layers
            next_layer = network.layers[i + 1]
            layer_deltas[i] = np.dot(layer_deltas[i + 1], next_layer.weights.T) * layer.activation_func.derivative(layer.output_pre_activation)

        # Update weights and biases
        layer.weights -= tmp_learning_rate * np.dot(layer.input.T, layer_deltas[i])
        layer.bias -= tmp_learning_rate * np.sum(layer_deltas[i], axis=0, keepdims=True)

# Instantiate and add layers to the network
network = Network(kernel_size=[3], pooling_layers=[2])
network.add(Layer(169, 64, ReLU()))      # Hidden layer 1
network.add(Layer(64, 64, ReLU()))       # Hidden layer 2
network.add(Layer(64, 10, Softmax()))    # Output layer

# Train the network
network.train(X, y, epochs=10, learning_rate=0.001, batch_size=32)

# Load the test dataset
test_dataset = pd.read_csv("test.csv")
test_data = np.array(test_dataset) / 255.0  # Normalize test data

# Assuming test data needs the same preprocessing
X_test_processed = np.zeros((test_data.shape[0], 784))
for j in range(len(test_data)):
    X_test_processed[j] = network.CNN_preprocessing(test_data[j])

# Predictions
predictions = network.forward(X_test_processed)
