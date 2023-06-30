import numpy as np
from typing import Tuple
from scipy import signal 
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: Tuple[int], kernal_size: int, depth: int) -> None:

        # assign each of the input's dimensions to a variable
        input_depth, input_height, input_width = input_shape

        self.depth = depth
        self.input_shape = input_shape

        self.input_depth = input_depth

        # the height and width of the output is the difference between the size of the input and kernel plus 1
        self.output_shape = (depth, input_height - kernal_size + 1, input_width - kernal_size + 1)
        
        # the shape of the kernels is 4D (input_depth = depth of each kernal) as each kernel is 3D and the fourth dimension is the number of them (depth)
        self.kernel_shape = (depth, input_depth, kernal_size, kernal_size)

        # the kernel and bias values are randomly chosen
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
        

    def forward(self, input) -> np.ndarray:
        self.input = input

        # start by coping the current biases
        self.output = np.copy(self.biases)

        # loop through the the output and input depths
        for i in range(self.depth):
            for j in range(self.input_depth):

                # calculate the output for each position of the kernel by summing the cross-correlations
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return self.output
    

    def backward(self, output_gradient, learning_rate) -> np.ndarray:

        # initialise empty matricies for the gradients
        kernel_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        # loop through the the output and input depths
        for i in range(self.depth):
            for j in range(self.input_depth):

                # calculate the derivative (gradient) of the Error w.r.t K --> dE / dK
                kernel_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")

                # calculate the gradient of the Input --> dE / dX
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full") 

            
        # update the kernels and biases (gradient descent)
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient     # -->  do not need to calculate the gradient of the bias as it's the same as the output

        return input_gradient
                




