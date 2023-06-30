import numpy as np
from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input) -> np.ndarray:
        # return the input in the shape of the output
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate) -> np.ndarray:
        # return the output in the shape of the input
        return np.reshape(output_gradient, self.input_shape)
