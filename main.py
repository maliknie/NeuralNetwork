import numpy as np

def ReLU(x):
    return np.maximum(0, x)

class Network:
    def __init__(self, layers=[None, None, None]):
        self.input_layer = layers[0]
        self.hidden_layers = layers[1:-1]
        self.output_layer = layers[-1]

        self.operations = None
    
    def forward(self, x):
        previous_layer = None
        for layer in self.hidden_layers:
            for neuron in layer.neurons:
                neuron.compute(x, previous_layer)
            previous_layer = layer

        return x
    
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

class Neuron:
    def __init__(self, activation=None, weights=[None], bias=None):
        self.activation = activation
        self.weights = weights
        self.bias = bias

    def compute(self, previous_layer):
        if previous_layer != None:
            self.activation = ReLU(np.dot(previous_layer, self.weights) + self.bias)
            return self.activation

class Operations:
    def __init__(self):
        pass

    def compute(self):
        pass

class add(Operations):
    def __init__(self):
        pass

    def compute(self, x, y):
        return x + y
    
n = Network()
input_layer = Layer([Neuron([1, 2, 3], 1), Neuron([4, 5, 6], 1)])
hidden_layer = Layer([Neuron([1, 2, 3], 1), Neuron([4, 5, 6], 1)])
output_layer = Layer([Neuron([1, 2, 3], 1), Neuron([4, 5, 6], 1)])
n.input_layer = input_layer
n.hidden_layers = [hidden_layer]
n.output_layer = output_layer