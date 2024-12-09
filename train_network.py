import numpy as np
import pandas as pd
import os
import pickle
import tkinter as tk


def load_dataset(dataset_type):
    match dataset_type:
        case "test":
            fp = FILEPATH_TEST
        case "train":
            fp = FILEPATH_TRAIN 
        case _:
            raise Exception("Enter either 'train' or 'test'.")
        
    dataset = pd.read_csv(fp)
    target = dataset["label"]
    data = dataset.drop("label", axis=1)

    X = np.array(data) / 255.0
    y = np.eye(10)[np.array(target)]

    return X, y

def go(network, mode, epochs, learning_rate, batch_size):
    match mode:
        case "train":
            X, y = load_dataset("train")
            network.train(X, y, epochs, learning_rate, batch_size)
        case "test":
            X, y = load_dataset("test")
            accuracy = network.test(X, y)
            print("Accuracy: ", accuracy)


class Activation:
    def relu(self, x_relu, derivative=False):
        if derivative:
            return (x_relu > 0).astype(float)
        return np.maximum(0, x_relu)
    
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
        if __name__ == "__main__":
            self.output = self.activation_func(self.output_pre_activation)
        else:
            self.output = self.activation_func(self, self.output_pre_activation)
        return self.output

class Network:
    def __init__(self):
        self.layers = []
        self.activation = Activation()

    def predict(self, X_guess, y_guess=None):
        if type(X_guess) == type(None):
            print("No input data provided.", X_guess)
            return
        prediction = np.argmax(self.forward(X_guess))
        if not y_guess == None:
            label = np.argmax(y_guess)
            return prediction, label
        return prediction

    def save_params(self, file_path):
        file_path = f"params/{file_path}"
        params = {
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.bias for layer in self.layers]
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Parameters saved to {file_path}")


    def load_params(self, file_path, load):
        if load:
            try:
                with open(file_path, 'rb') as f:
                    params = pickle.load(f)
                for i, layer in enumerate(self.layers):
                    layer.weights = params['weights'][i]
                    layer.bias = params['biases'][i]
                print(f"Parameters loaded from {file_path}")
            except FileNotFoundError:
                print("No parameters found. Training network from scratch.")

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
        predictions = self.forward(X)
        num_layers = len(self.layers)
        layer_deltas = [None] * num_layers
        loss = self.calculate_loss(X, y)
        tmp_learning_rate = min(learning_rate, loss * 0.1)
        
        delta = predictions - y 
        
        for i in reversed(range(num_layers)):
            layer = self.layers[i]
            
            if i == num_layers - 1:
                layer_deltas[i] = delta
            else:
                next_layer = self.layers[i + 1]
                activation_derivative = layer.activation_func(layer.output_pre_activation, derivative=True)
                delta = np.dot(layer_deltas[i + 1], next_layer.weights.T) * activation_derivative
                layer_deltas[i] = delta
            
            input_to_layer = X if i == 0 else self.layers[i - 1].output
            dLoss_dWeights = np.dot(input_to_layer.T, layer_deltas[i]) / X.shape[0]
            dLoss_dBias = np.sum(layer_deltas[i], axis=0, keepdims=True) / X.shape[0]
            
            layer.weights -= tmp_learning_rate * dLoss_dWeights
            layer.bias -= tmp_learning_rate * dLoss_dBias
    
    def train(self, X, y, epochs, learning_rate, batch_size):
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
                print('Parameters saved')
                print("Training complete")
            if epoch % 50 == 0 and epoch != 0:
                self.save_params(f'params_{epoch}.bin')
                print(f'Parameters saved at epoch {epoch}')

    def test(self, X, y):
        misclassified_images = []

        for i in range(len(X)):
            output = self.forward(X[i])
            prediction = np.argmax(output)
            confidence = np.max(output)
            actual = np.argmax(y[i])
            
            if i % 500 == 0:
                print(f"Predicted: {prediction}, Actual: {actual}, Confidence: {confidence:.2f}")

            if prediction != actual:
                misclassified_images.append((X[i], actual, prediction, confidence))

        predictions = self.forward(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

        return accuracy


if __name__ == "__main__":
    # Filepath settings
    FILEPATH_TEST = "datasets/test.csv"
    FILEPATH_TRAIN = "datasets/train.csv"
    FILEPATH_PARAMS = "params/params97_38.bin"

    # General Settings
    MODE = "test"
    LOAD_PARAMS = True

    #Training Settings
    EPOCHS = 5
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32


    activation = Activation()
    network = Network()

    network.add(Layer(784, 512, activation.relu))
    network.add(Layer(512, 256, activation.relu))
    network.add(Layer(256, 128, activation.relu))
    network.add(Layer(128, 10, activation.softmax))

    network.load_params(FILEPATH_PARAMS, LOAD_PARAMS)

    go(network, MODE, EPOCHS, LEARNING_RATE, BATCH_SIZE)