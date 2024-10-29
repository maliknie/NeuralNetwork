import numpy as np
class ConvLayer:
    def __init__(self, num_filters, filter_size, padding=0, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.filters = np.random.randn(num_filters, 1, filter_size, filter_size) / (filter_size * filter_size)
        self.input = None
        self.d_input = None

    def conv2d(self, x):
        # Adjust input dimensions
        if x.ndim == 3:
            x = x[np.newaxis, :, :, :]
        elif x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]

        batch_size, channels, height, width = x.shape

        # Apply padding to the input
        padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Calculate output dimensions
        out_height = (height * self.padding - self.filter_size) // self.stride + 1
        out_width = (width * self.padding - self.filter_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        # Convolve filters with input
        for i in range(out_height):
            for j in range(out_width):
                try:
                    # Check for filter position validity
                    if (i * self.stride + self.filter_size > height) or (j * self.stride + self.filter_size > width):
                        raise ValueError(f"Filter position error at output index (i={i}, j={j}) - outside input bounds.")

                    region = padded_input[:, :, 
                                          i * self.stride:i * self.stride + self.filter_size, 
                                          j * self.stride:j * self.stride + self.filter_size]
                    output[:, :, i, j] = np.tensordot(region, self.filters, axes=((1, 2, 3), (1, 2, 3)))
                except ValueError as e:
                    print(e)
                    continue  # Skip invalid positions

        return output

    def forward(self, input):
        self.input = input
        self.output = self.conv2d(input)
        return self.output

    def backward(self, d_out, learning_rate):
        batch_size, _, height, width = self.input.shape
        num_filters, _, filter_height, filter_width = self.filters.shape

        d_input = np.zeros_like(self.input, dtype=np.float64)
        d_filters = np.zeros_like(self.filters, dtype=np.float64)

        # Only loop through valid positions where the filter fully fits in the input
        for k in range(num_filters):
            for i in range((height - filter_height) // self.stride + 1):
                for j in range((width - filter_width) // self.stride + 1):
                    try:
                        # Check for filter position validity
                        if (i * self.stride + filter_height > height) or (j * self.stride + filter_width > width):
                            raise ValueError(f"Backward filter position error for filter {k} at (i={i}, j={j}) - outside input bounds.")

                        # Extract the region from the input
                        region = self.input[:, :, 
                                            i * self.stride:i * self.stride + filter_height, 
                                            j * self.stride:j * self.stride + filter_width]

                        d_out_k = d_out[:, k, i, j].reshape(batch_size, 1, 1, 1)

                        # Print shapes for debugging
                        #print(f"Filter index: {k}, position (i={i}, j={j}), region shape: {region.shape}, d_out_k shape: {d_out_k.shape}, d_filters[{k}] shape before: {d_filters[k].shape}")

                        # Ensure d_filters[k] has the correct shape for addition
                        if d_filters[k].shape != (1, filter_height, filter_width):
                            #print(f"Adjusting shape of d_filters[{k}] from {d_filters[k].shape} to {(1, filter_height, filter_width)}")
                            d_filters[k] = np.zeros((1, filter_height, filter_width))

                        # Update d_filters and d_input
                        d_filters[k] += np.tensordot(region, d_out_k, axes=(0, 0)).reshape(1, filter_height, filter_width)
                        d_input[:, :, 
                                 i * self.stride:i * self.stride + filter_height, 
                                 j * self.stride:j * self.stride + filter_width] += (
                            d_out_k * self.filters[k]
                        )
                    except ValueError as e:
                       # print(f"filter index: {k}, i: {i}, j: {j}")
                        print(e)
                        continue  # Skip invalid positions
                    except Exception as e:
                        print(f"Unexpected error for filter {k} at (i={i}, j={j}): {e}")
                        continue  # Catch any other unexpected errors

        # Update filters with learning rate
        self.filters -= learning_rate * d_filters
        return d_input




# Activation Layer
class ActivationLayer:
    def __init__(self, activation_func, derivative_func=None):
        self.activation_func = activation_func
        self.derivative_func = derivative_func

    def forward(self, x):
        self.input = x
        return self.activation_func(x)

    def backward(self, d_out, learning_rate=None):
        if self.derivative_func is not None:
            return d_out * self.derivative_func(self.input)
        return d_out

# Pooling Layer with Backpropagation
class PoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size
        output = np.zeros((batch_size, num_channels, out_height, out_width))
        self.indices = np.zeros_like(x)

        for i in range(out_height):
            for j in range(out_width):
                region = x[:, :, 
                            i * self.pool_size:i * self.pool_size + self.pool_size, 
                            j * self.pool_size:j * self.pool_size + self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))
                self.indices[:, :, 
                              i * self.pool_size:i * self.pool_size + self.pool_size, 
                              j * self.pool_size:j * self.pool_size + self.pool_size] = (
                    region == output[:, :, i, j][:, :, np.newaxis, np.newaxis]
                )
        
        return output
    
    def backward(self, d_out, learning_rate=None):
        batch_size, num_channels, height, width = d_out.shape
        d_input = np.zeros_like(self.indices)

        for i in range(height):
            for j in range(width):
                d_input[:, :, 
                         i * self.pool_size:i * self.pool_size + self.pool_size, 
                         j * self.pool_size:j * self.pool_size + self.pool_size] += (
                    d_out[:, :, i, j][:, :, np.newaxis, np.newaxis] * 
                    self.indices[:, :, 
                                  i * self.pool_size:i * self.pool_size + self.pool_size, 
                                  j * self.pool_size:j * self.pool_size + self.pool_size]
                )
        
        return d_input

# Activation Functions
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        return Activation.softmax(x) * (1 - Activation.softmax(x))

# Dense Layer (Fully Connected Layer)
class Layer:
    def __init__(self, input_dim, output_dim, activation_func, derivative_func=None):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.activation_func = activation_func 
        self.derivative_func = derivative_func 
    
    def forward(self, input_data):
        self.input = input_data 
        self.output_pre_activation = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation_func(self.output_pre_activation)
        return self.output

    def backward(self, d_out, learning_rate):
        if self.derivative_func is not None:
            d_out *= self.derivative_func(self.output_pre_activation)

        d_input = np.dot(d_out, self.weights.T)  
        d_weights = np.dot(self.input.T, d_out)
        d_bias = np.sum(d_out, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return d_input

# Flatten Layer
class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d_out, learning_rate=None):
        return d_out.reshape(self.input_shape)

# Model
class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out, learning_rate):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate)

# Loss Functions
class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
