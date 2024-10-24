import numpy as np
import pandas as pd
import os
import pickle
import tkinter as tk

if __name__ == "__main__":

    lines = 0
    with open("datasets/drawn_digit.csv", "r") as file:
        for line in file:
            lines += 1

    df = pd.read_csv("datasets/drawn_digit.csv", header=None)

    X_guess = df.iloc[:, :-1].values
    y_guess = df.iloc[:, -1].values 

    # Reshape the input data to 28x28 images
    X_guess = X_guess.reshape(lines, 784).astype('float32')  # Reshape to (num_samples, 28, 28, 1)
    X_guess /= 255  # Normalize to [0, 1]

    # Convert labels to integers if they are strings
    y_guess = y_guess.astype(int)

    print(f"Loaded {len(X_guess)} samples.")




    test_dataset = pd.read_csv("datasets/test.csv")
    test_target = test_dataset["label"]
    test_data = test_dataset.drop("label", axis=1)
    X_test = np.array(test_data) / 255.0
    y_test = np.eye(10)[np.array(test_target)]




    LOAD_PARAMS = True

    # Load and preprocess the dataset
    dataset = pd.read_csv("datasets/train.csv")
    target = dataset["label"]
    data = dataset.drop("label", axis=1)

    # Normalize input data
    X = np.array(data) / 255.0  # Scale pixel values to [0, 1]
    y = np.eye(10)[np.array(target)]  # One-hot encoding of labels

class ImageDisplay:
    def __init__(self, master):
        self.master = master
        self.master.title("Misclassified Image Viewer")

        # Canvas to display the image
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        # Label to show the true label, predicted label, and confidence
        self.label = tk.Label(master, text="", font=('Arial', 14))
        self.label.pack()

        # Label to show the confidence score
        self.confidence_label = tk.Label(master, text="", font=('Arial', 12))
        self.confidence_label.pack()

        # Button to skip to the next misclassified image
        self.next_button = tk.Button(master, text="Next", command=self.show_next_image)
        self.next_button.pack()

        # To store misclassified images and indices
        self.misclassified_images = []
        self.current_index = 0

    def display_image(self, pixel_values, true_label, predicted_label, confidence):
        self.canvas.delete("all")  # Clear previous image
        pixel_values = [float(v) for v in pixel_values]  # Ensure values are floats

        # Ensure pixel_values has exactly 784 values
        if len(pixel_values) != 784:
            raise ValueError("The input must contain exactly 784 values.")

        # Loop over the pixel values to display each one on the canvas
        for i in range(28):
            for j in range(28):
                value = pixel_values[i * 28 + j]  # Access each value
                # Multiply by 255 to get the grayscale intensity (0-255)
                grayscale_value = int(value * 255)
                # Create the hex color for the grayscale (RGB)
                hex_color = f'#{grayscale_value:02x}{grayscale_value:02x}{grayscale_value:02x}'
                # Draw a rectangle (10x10 pixels for each "pixel" on the canvas)
                self.canvas.create_rectangle(
                    j * 10, i * 10, (j + 1) * 10, (i + 1) * 10,
                    outline="", fill=hex_color
                )

        # Update the label with true and predicted values
        self.label.config(text=f"True Label: {true_label}, Predicted Label: {predicted_label}")

        # Update the confidence label
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")

    def load_misclassified_images(self, images):
        """Load the list of misclassified images."""
        self.misclassified_images = images
        self.current_index = 0  # Start from the first misclassified image
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


# Activation Functions
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

    def guess(self, X_guess, y_guess=None):
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


    def load_params(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = params['weights'][i]
            layer.bias = params['biases'][i]
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
        # Store misclassified images along with confidence scores
        misclassified_images = []

        for i in range(len(X)):
            output = self.forward(X[i])
            prediction = np.argmax(output)
            confidence = np.max(output)  # Confidence is the highest probability from the softmax output
            actual = np.argmax(y[i])
            
            if i % 500 == 0:
                print(f"Predicted: {prediction}, Actual: {actual}, Confidence: {confidence:.2f}")

            # Identify misclassified images
            if prediction != actual:
                misclassified_images.append((X[i], actual, prediction, confidence))

        # Calculate accuracy
        predictions = self.forward(X)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

        return accuracy, misclassified_images





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


if __name__ == "__main__":
    # Initialize the network and add layers

    activation = Activation()
    network = Network()

    network.add(Layer(784, 512, activation.relu))      # Hidden layer 1
    network.add(Layer(512, 256, activation.relu))       # Hidden layer 2
    network.add(Layer(256, 128, activation.relu))       # Hidden layer 3
    network.add(Layer(128, 10, activation.softmax))    # Output layer

    if LOAD_PARAMS:
        try:
            network.load_params("params/params97_38.bin")
        except FileNotFoundError:
            print("No parameters found. Training network from scratch.")

    # Train the network
    #network.train(X, y, epochs=5, learning_rate=0.0005, batch_size=32)


    # Load the test dataset and evaluate the network
    accuracy, misclassified_images = network.test(X_test, y_test)
    print(accuracy)

    # Ask the user if they want to see the errors
if input("Do you want to see the errors? (Y/n) ").lower() == "y":
    # Display misclassified images using the ImageDisplay class
    root = tk.Tk()
    app = ImageDisplay(root)  # Pass the display (ImageDisplay class)
    app.load_misclassified_images(misclassified_images)  # Load the misclassified images
    root.mainloop()  # Start the Tkinter event loop


#print(network.guess(X_guess, y_guess))
