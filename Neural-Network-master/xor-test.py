import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

class Layer:
    """
    The base layer class for all layers

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer
    
    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name):
        """
        The constructor for the layer class

        Parameters:
            name: The name of the layer
        """

        self.name = name
        self.input_shape = None
        self.output_shape = None

    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            x: The input to the layer

        Returns:
            The output of the layer
        """

        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        # we can use an optimizer instead of learning rate
        """
        The backward pass of the layer

        Parameters:
            grad: The gradient of the loss w.r.t. the output of the layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """

        raise NotImplementedError


class DenseLayer(Layer):
    """
    The dense layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer
        weights: The weights of the layer
        bias: The bias of the layer
    
    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name, input_shape, output_shape):
        """
        The constructor for the dense layer class

        Parameters:
            name: The name of the layer
            input_shape: The shape of the input to the layer
            output_shape: The shape of the output of the layer
        """

        super().__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape
     

        # Xavier initialization
        # sampling weights from a standard normal distribution
        np.random.seed(182)
        
        self.weights = np.random.randn(output_shape, input_shape)
        self.bias = np.random.randn(output_shape, 1)
        # randn returns a sample from the "standard normal" distribution

        print ("in Xaviar init: ")
        print ("weights: \n", self.weights)
        print ("bias: \n", self.bias)

    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            input: The input to the layer

        Returns:
            The output of the layer
        """

        self.input = input

        # print ("input shape: ", self.input.shape)
        # print ("weights shape: ", self.weights.shape)
        # print ("bias shape: ", self.bias.shape)

        return np.dot(self.weights, self.input) + self.bias
    

    def backward(self, output_grad, learning_rate):
        """
        The backward pass of the layer

        Parameters:
            output_grad: The gradient of the loss w.r.t. the output of the layer
     
        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """
        # output_grad: dL/dy (L: loss, y: output of the layer)
        # output_grad: The gradient of the loss w.r.t. the output of the layer

        weights_grad = np.dot(output_grad, self.input.T)
        # dL/dB = dL/dy * dy/dB = dL/dy * 1
        bias_grad = output_grad

        # input_grad: dL/dx
        input_grad = np.dot(self.weights.T, output_grad)

    
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * bias_grad

        return input_grad

network = [
    DenseLayer("dense1", X.shape[1], 3),
    Tanh(),
    DenseLayer("dense2", 3, 1),
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1)

# decision boundary plot

# points = []
# for x in np.linspace(0, 1, 20):
#     for y in np.linspace(0, 1, 20):
#         z = predict(network, [[x], [y]])
#         points.append([x, y, z[0,0]])

# points = np.array(points)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
# plt.show()
