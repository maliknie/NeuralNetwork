from MalikSchmiddiFramework import *
import numpy as np
import pickle
import pandas as pd

# Load MNIST data from CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    labels = data['label'].values
    data = data.drop('label', axis=1).values
    return data, labels

# Split data into training and validation sets
def train_val_split(data, labels, val_ratio=0.2):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    split_idx = int(data.shape[0] * (1 - val_ratio))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    x_train, x_val = data[train_indices], data[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]
    return x_train, y_train, x_val, y_val

# One-hot encoding for labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Train the model
def train_model(model, x_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(x_train)):
            x = x_train[i].reshape(1, 1, 28, 28)
            y = y_train[i].reshape(1, -1) 
            
            # Forward pass
            output = model.forward(x)
            
            # Compute loss (using the Loss class)
            loss = Loss.categorical_crossentropy(y, output)
            epoch_loss += loss
            
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {i}/{len(x_train)}, Loss: {loss}")

            # Backward pass
            grad_output = Loss.categorical_crossentropy_derivative(y, output)
            model.backward(grad_output, learning_rate)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(x_train)}")


def save_model(model, filename="cnn_model.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

filename = "datasets/train.csv"
data, labels = load_data(filename)
labels = one_hot_encode(labels)

# Split data into training and validation sets
x_train, y_train, x_val, y_val = train_val_split(data, labels, val_ratio=0.2)

# Define CNN model architecture
model = Model([
    ConvLayer(filter_size=3, num_filters=8, padding=1, stride=1),
    ActivationLayer(Activation.relu, Activation.relu_derivative),
    PoolingLayer(pool_size=2, stride=2),
    Flatten(),
    Layer(input_dim=1352, output_dim=64, activation_func=Activation.relu, derivative_func=Activation.relu_derivative),
    Layer(input_dim=64, output_dim=32, activation_func=Activation.relu, derivative_func=Activation.relu_derivative), 
    Layer(input_dim=32, output_dim=10, activation_func=Activation.softmax)
])

# Train the model
train_model(model, x_train, y_train, epochs=10, learning_rate=0.001)

# Save the trained model
save_model(model)
