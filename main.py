import math
import random
import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")
target = dataset["label"]
data = dataset.drop("label", axis=1)

e = math.e
#np.random.seed(420)


X = np.array(data)

y = np.array(target)
y = np.eye(10)[y]


class Layer:
    def __init__(self, input_dim, output_dim): # output_dim = number of neurons
        self.weights = np.array([[random.randint(1, 999)/1000 for _ in range(input_dim)]for _ in range(output_dim)])
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

class Activation:
    def forward(self, input):
        self.output = np.maximum(0, input)

layer1 = Layer(784, 3) # 784 input features, 3 neurons
layer2 = Layer(3, 2) # 3 input features, 2 neurons
layer3 = Layer(2, 10) # 2 input features, 1 neuron
activation1 = Activation()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
layer3.forward(activation1.output)
activation1.forward(layer3.output)

prediction = activation1.output

loss = np.sum((prediction - y)**2)
cross_entropy_loss = -np.sum(y * np.log(prediction + 1e-10))

# Here I implement back propagation:

# Loss = (y - prediction)^2
# dLoss/dPrediction = -2(y - prediction)
# dPrediction/dActivation = 1
# dActivation/dLayer = 1 if layer.output > 0 else 0
# dLayer/dWeights = input


print("Loss of the network:")
print("MSE: ", loss)
print("Cross Entropy: ", cross_entropy_loss)