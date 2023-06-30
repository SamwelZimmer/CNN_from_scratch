import numpy as np
from layer import Layer


class Dense(Layer): 
    def __init__(self, input_size: int, output_size: int):
        # randomly initialise the weights (no.cols = input_size, no.rows = output_size) and biases (column vector)
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)


    def forward(self, input) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    

    def backward(self, output_gradient, learning_rate) -> np.ndarray:

        # derivative of error w.r.t. the weights
        weight_gradient = np.dot(output_gradient, self.input.T)

        # update parameters using gradient descent
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient

        # return derivative of error w.r.t. the inputs
        return np.dot(self.weights.T, output_gradient)


    

