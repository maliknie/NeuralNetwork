import numpy as np
import pandas as pd
import os
import pickle
import tkinter as tk


def show_errors():
    if input("Do you want to see the errors? (Y/n) ").lower() == "y":
        root = tk.Tk()
        app = ImageDisplay(root)
        app.load_misclassified_images(misclassified_images)
        root.mainloop()

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
            show_errors()

class ImageDisplay:
    def __init__(self, master):
        self.master = master
        self.master.title("Misclassified Image Viewer")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.label = tk.Label(master, text="", font=('Arial', 14))
        self.label.pack()

        self.confidence_label = tk.Label(master, text="", font=('Arial', 12))
        self.confidence_label.pack()

        self.next_button = tk.Button(master, text="Next", command=self.show_next_image)
        self.next_button.pack()

        self.misclassified_images = []
        self.current_index = 0

    def display_image(self, pixel_values, true_label, predicted_label, confidence):
        self.canvas.delete("all")
        pixel_values = [float(v) for v in pixel_values]

        if len(pixel_values) != 784:
            raise ValueError("The input must contain exactly 784 values.")

        for i in range(28):
            for j in range(28):
                value = pixel_values[i * 28 + j]
                grayscale_value = int(value * 255)
                hex_color = f'#{grayscale_value:02x}{grayscale_value:02x}{grayscale_value:02x}'
                self.canvas.create_rectangle(
                    j * 10, i * 10, (j + 1) * 10, (i + 1) * 10,
                    outline="", fill=hex_color
                )

        self.label.config(text=f"True Label: {true_label}, Predicted Label: {predicted_label}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")

    def load_misclassified_images(self, images):
        """Load the list of misclassified images."""
        self.misclassified_images = images
        self.current_index = 0 
        self.show_next_image()

    def show_next_image(self):
        """Display the next misclassified image in the list."""
        if self.current_index < len(self.misclassified_images):
            image_data, true_label, predicted_label, confidence = self.misclassified_images[self.current_index]
            self.display_image(image_data, true_label, predicted_label, confidence)
            self.current_index += 1
        else:
            self.label.config(text="No more misclassified images.")
            self.confidence_label.config(text="")

class Activation:
    def relu(self, x_relu, derivative=False):
        if derivative:
            return (x_relu > 0).astype(float)
        return np.maximum(0, x_relu)
    
    def softmax(self, x, derivative=False):
        # NaN and Inf error catch
        if np.isnan(x).any() or np.isinf(x).any():
            print("NaN or Inf detected in input to softmax.")
            print(x)
            raise ValueError("Invalid values in softmax input.")
        
        # Softmax calculation
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Derivative = Jacobian matrix
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

class ConvLayer:
    def __init__(self, num_filters, filter_size, padding=0, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def conv2d(self, x):
        if x.ndim == 3:  # Add batch dimension if missing
            x = x[np.newaxis, :, :, :]
        elif x.ndim == 2:  # Add both batch and channel dimensions
            x = x[np.newaxis, np.newaxis, :, :]
        
        batch_size, channels, height, width = x.shape
        padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        out_height = (height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.filter_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                region = padded_input[:, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size]
                output[:, :, i, j] = np.tensordot(region, self.filters, axes=((1, 2, 3), (1, 2, 3)))
        
        return output
    
    def forward(self, input):
        self.input = input
        self.output = self.conv2d(input)
        return self.output

    def backward(self, d_out, learning_rate):
        d_filters = np.zeros_like(self.filters)
        batch_size, channels, height, width = self.input.shape
        padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        padded_d_input = np.pad(np.zeros_like(self.input), ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(d_out.shape[2]):
            for j in range(d_out.shape[3]):
                region = padded_input[:, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size]
                for f in range(self.num_filters):
                    d_filters[f] += np.sum(region * d_out[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                
                for n in range(batch_size):
                    padded_d_input[n, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size] += \
                        np.sum(self.filters[:, np.newaxis, :, :] * d_out[n, :, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)

        d_input = padded_d_input[:, :, self.padding:height+self.padding, self.padding:width+self.padding]
        self.filters -= learning_rate * d_filters
        return d_input


class PoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size
        output = np.zeros((batch_size, num_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                region = x[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))

        return output

    def backward(self, d_out):
        batch_size, num_channels, height, width = d_out.shape
        d_input = np.zeros((batch_size, num_channels, height * self.pool_size, width * self.pool_size))

        for i in range(height):
            for j in range(width):
                region = d_input[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                region[np.arange(batch_size)[:, None, None], np.arange(num_channels)[:, None], np.unravel_index(np.argmax(region, axis=(2, 3)), (self.pool_size, self.pool_size))] = d_out[:, :, i, j]

        return d_input

class Network:
    def __init__(self):
        self.layers = [
            ConvLayer(num_filters=8, filter_size=3, padding=1),  # Convolutional layer
            PoolingLayer(pool_size=2, stride=2),  # Pooling layer
            ConvLayer(num_filters=16, filter_size=3, padding=1),  # Second convolutional layer
            PoolingLayer(pool_size=2, stride=2),  # Second pooling layer
            ConvLayer(num_filters=32, filter_size=3, padding=1),  # Third convolutional layer
            PoolingLayer(pool_size=2, stride=2),  # Third pooling layer
            # Fully connected layers can be added here as before...
        ]
        self.activation = Activation()
        self.misclassified_images = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out, learning_rate):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate)

    def train(self, X, y, epochs, learning_rate, batch_size):
        num_batches = X.shape[0] // batch_size
        for epoch in range(epochs):
            for i in range(num_batches):
                x_batch = X[i * batch_size: (i + 1) * batch_size]
                y_batch = y[i * batch_size: (i + 1) * batch_size]
                predictions = self.forward(x_batch)
                loss = predictions - y_batch
                d_out = self.activation.softmax(predictions, derivative=True)  # Use softmax derivative
                self.backward(d_out, learning_rate)

    def test(self, X, y):
        correct_predictions = 0
        for i in range(X.shape[0]):
            prediction = self.forward(X[i:i+1])
            if np.argmax(prediction) == np.argmax(y[i]):
                correct_predictions += 1
            else:
                self.misclassified_images.append((X[i], np.argmax(y[i]), np.argmax(prediction), np.max(prediction)))
        return correct_predictions / X.shape[0]

# Initialize and run the network
network = Network()
FILEPATH_TRAIN = "datasets/train.csv"
go(network, "train", epochs=10, learning_rate=0.01, batch_size=64)  # Adjust epochs and learning rate as needed
