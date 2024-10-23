import numpy as np
import pandas as pd
import os
import pickle
import math
import tkinter as tk

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

    def guess(self, X, y):
        print("Prediction:")
        print(np.argmax(self.forward(X)))
        print("Label: ")
        print(y)

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
                self.save_params('parameter.bin')
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

network.add(Layer(784, 128, activation.relu))      # Hidden layer 1
network.add(Layer(128, 32, activation.relu))       # Hidden layer 2
network.add(Layer(32, 10, activation.softmax))    # Output layer

# Train the network
network.train(X, y, epochs=25, learning_rate=0.01, batch_size=32)

def predict_image():
    class DrawingApp:
        def __init__(self, master):
            self.master = master
            self.master.title("Draw a Digit with Soft Gray Edges")
    
            self.canvas = tk.Canvas(master, width=280, height=280, bg='black')  # Background black for MNIST-style look
            self.canvas.pack()

            self.save_button = tk.Button(master, text='Save', command=self.return_image)
            self.save_button.pack()
            
            self.clear_button = tk.Button(master, text='Clear', command=self.clear_canvas)
            self.clear_button.pack()
    
            self.canvas.bind("<B1-Motion>", self.paint)
            self.canvas.bind("<ButtonRelease-1>", self.reset)
    
            self.image_data = np.zeros((28, 28), dtype=int)  # Initialize as black
            self.returned_image = None
    
        def paint(self, event):
            x, y = event.x // 10, event.y // 10
            if 0 <= x < 28 and 0 <= y < 28:
                # Draw the main white line (255)
                if self.image_data[y, x] < 255:  # Only update if it's not already white
                    self.image_data[y, x] = 255
                    self.update_canvas(x, y, 255)
    
                # Draw smooth gray surrounding pixels
                self.draw_surrounding_pixels(x, y)
    
        def draw_surrounding_pixels(self, x, y):
            for dx in [-2, -1, 0, 1, 2]:  # Include a larger radius for smoother transition
                for dy in [-2, -1, 0, 1, 2]:
                    if (dx != 0 or dy != 0) and 0 <= x + dx < 28 and 0 <= y + dy < 28:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= 2:  # Limit the distance
                            # Gradually fade to gray, with closer pixels being brighter
                            gray_value = int(255 * (1 - distance / 2))  # Full white to dark gray
                            # Update only if the new gray value is lighter (higher) than the current value
                            if self.image_data[y + dy, x + dx] < gray_value:
                                self.image_data[y + dy, x + dx] = gray_value
                                self.update_canvas(x + dx, y + dy, gray_value)
    
        def update_canvas(self, x, y, gray_value):
            hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
            self.canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill=hex_color, outline=hex_color)
    
        def reset(self, event):
            pass
        
        def clear_canvas(self):
            self.canvas.delete("all")
            self.image_data.fill(0)  # Reset to black
    
        def return_image(self):
            self.returned_image = self.image_data.reshape(1, 784) / 255.0
            self.master.quit()  # Close the window
    
    # Initialize the Tkinter application and start the main loop
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

    return app.returned_image  # Return the image after the window is closed

    
    

# Load the test dataset
test_dataset = pd.read_csv("test.csv")
test_target = test_dataset["label"]
test_data = test_dataset.drop("label", axis=1)
X = np.array(test_data) / 255.0
y = np.eye(10)[np.array(test_target)]

print(network.test(X, y))

while True:
    input_tensor = predict_image()  # input_tensor should now be a valid array, not None
    if input_tensor is not None:
        network.guess(input_tensor, "No Label given")
    else:
        print("No image drawn.")

    user_input = input("Draw another image? (Y/n): ")
    if user_input.lower() == 'n' or user_input.lower() == 'N':
        break
