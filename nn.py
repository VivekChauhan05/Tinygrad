from tinygrad.Engine import Value
from tinygrad.graph import Draw
import matplotlib.pyplot as plt
import random

class TinyNeuron:
    def __init__(self, num_inputs, activation):
        # Initialize a tiny neuron with random weights and a bias.
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, inputs):
        # Compute the weighted sum and apply the activation function.
        weighted_sum = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        output = self.activation(weighted_sum)
        return output

    def parameters(self):
        # Return the parameters of the tiny neuron (weights and bias).
        return self.weights + [self.bias]

class TinyLayer:
    def __init__(self, num_inputs, num_neurons, activation):
        # Initialize a tiny layer with multiple tiny neurons.
        self.neurons = [TinyNeuron(num_inputs, activation) for _ in range(num_neurons)]

    def __call__(self, inputs):
        # Pass the input through all tiny neurons in the tiny layer.
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Return the parameters of all tiny neurons in the tiny layer.
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class TinyNeuralNetwork:
    def __init__(self, input_features, hidden_layers, activations):
        # Initialize a tiny neural network with tiny layers and activation functions.
        layer_sizes = [input_features] + hidden_layers
        activation_funcs = [Activation_function[name] for name in activations]
        self.layers = [TinyLayer(layer_sizes[i], layer_sizes[i + 1], activation_funcs[i]) for i in range(len(hidden_layers))]

    def __call__(self, inputs):
        # Pass the input through all tiny layers of the tiny neural network.
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        # Return the parameters of all tiny layers in the tiny neural network.
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self, X, Y, iterations, loss_function, learning_rate=0.01):
        history = []

        for i in range(iterations):
            # Forward pass
            predictions = [self(x) for x in X]
            error = loss_function(Y, predictions)

            # Backward pass
            for p in self.parameters():
                p.grad = 0.0
            error.backward()

            # Gradient descent
            for p in self.parameters():
                p.data -= learning_rate * p.grad

            history.append([i + 1, error.data])

        graph = Draw(error)

        return history, graph

    def predict(self, X):
        predictions = [self(x).data for x in X]
        return predictions

    def plot(self, history):
        plt.xlabel("Iterations")
        plt.ylabel("Cost Function")
        plt.title("Learning Curve")
        plt.plot([row[0] for row in history], [row[1] for row in history], color='red', label='loss')
        plt.legend()

Activation_function = {
    'Tanh': Value.tanh,
    'Relu': Value.relu,
    'Sigmoid': Value.sigmoid,
    'Linear': Value.linear
}
