from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):

        # the activation function and its derivative
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input) -> np.ndarray:
        self.input = input

        # return the inputs modifed by the activation function
        return self.activation(self.input) 
    
    def backward(self, output_gradient, learning_rate) -> np.ndarray:

        # return the derivative of the error w.r.t the input
        return np.multiply(output_gradient, self.activation_prime(self.input))
