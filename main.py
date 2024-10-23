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
    def relu(self, x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)
    
    def softmax(self, x, derivative=False):
        # nan und inf error catche
        if np.isnan(x).any() or np.isinf(x).any():
            print("NaN or Inf detected in input to softmax.")
            print(x)
            raise ValueError("Invalid values in softmax input.")
        
        # softmax usrechne
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # derivative = jacobian matrix
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        if derivative:
            jacobian_matrices = []
            for prob in probabilities:
                jacobian_matrix = np.diag(prob)
                for i in range(len(prob)):
                    for j in range(len(prob)):
                        if i == j:
                            jacobian_matrix[i][j] = prob[i] * (1 - prob[i])
                        else:
                            jacobian_matrix[i][j] = -prob[i] * prob[j]
                jacobian_matrices.append(jacobian_matrix)
            return np.array(jacobian_matrices)
        
        return probabilities

class Layer:
    def __init__(self, input_dim, output_dim, activation_func, load_params=False):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func 
    
    def forward(self, input_data):
        self.input = input_data 
        self.output_pre_activation = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func(self.output_pre_activation)
        return self.output

class Network:
    def __init__(self):
        self.layers = []
        self.activation = Activation()

    def save_params(self, file_path):
        fd = os.open(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    
        params = {
            'weights': [layer.weights for layer in network.layers],
            'biases': [layer.bias for layer in network.layers]
        }
        
        serialized_params = pickle.dumps(params)
        
        os.write(fd, serialized_params)

        os.close(fd)

        print(f"Parameters saved to {file_path}")

    def load_params(self, file_path):
        fd = os.open(file_path, os.O_RDONLY)
        serialized_params = os.read(fd, os.path.getsize(file_path))
        params = pickle.loads(serialized_params)
        
        for i, layer in enumerate(network.layers):
            layer.weights = params['weights'][i]
            layer.bias = params['biases'][i]
        
        os.close(fd)

        print(f"Parameters loaded from {file_path}")
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X  
        return self.output
    
    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        epsilon = 1e-8  # Kleine konstante damit es kein log(0) gibt
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
                print("Training complete")
            if epoch % 50 == 0 and epoch != 0:
                self.save_params(f'params_{epoch}.bin')
                print(f'Parameters saved at epoch {epoch}')
    def test(self, X, y):
        for i in range(len(X)):
            #print(f"Predicted: {np.argmax(self.forward(X[i]))}, Actual: {np.argmax(y[i])}")
            pass
        predictions = self.forward(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return accuracy



def backpropagation(network, X, y, learning_rate):
    predictions = network.forward(X)
    num_layers = len(network.layers)
    layer_deltas = [None] * num_layers
    loss = network.calculate_loss(X, y)
    tmp_learning_rate = min(learning_rate, loss * 0.1)
    
    # Calculate delta for the output layer
    delta = predictions - y 
    
    for i in reversed(range(num_layers)):
        layer = network.layers[i]
        
        if i == num_layers - 1:
            layer_deltas[i] = delta
        else:
            next_layer = network.layers[i + 1]
            activation_derivative = layer.activation_func(layer.output_pre_activation, derivative=True)
            delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * activation_derivative
            layer_deltas[i] = delta
        
        input_to_layer = X if i == 0 else network.layers[i - 1].output
        dLoss_dWeights = np.dot(input_to_layer.T, layer_deltas[i]) / X.shape[0]
        dLoss_dBias = np.sum(layer_deltas[i], axis=0, keepdims=True) / X.shape[0]
        
        layer.weights -= tmp_learning_rate * dLoss_dWeights
        layer.bias -= tmp_learning_rate * dLoss_dBias

# Initialize the network and add layers
activation = Activation()
network = Network()

network.add(Layer(784, 512, activation.relu))      # Hidden layer 1
network.add(Layer(512, 256, activation.relu))       # Hidden layer 2
network.add(Layer(256, 128, activation.relu))       # Hidden layer 3
network.add(Layer(128, 10, activation.softmax))    # Output layer

# Train the network
network.train(X, y, epochs=5, learning_rate=0.001, batch_size=32)


# Load the test dataset
test_dataset = pd.read_csv("test.csv")
test_target = test_dataset["label"]
test_data = test_dataset.drop("label", axis=1)
X = np.array(test_data) / 255.0
y = np.eye(10)[np.array(test_target)]

print(network.test(X, y))