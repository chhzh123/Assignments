import torch
import numpy as np

class Activation(object): # Base class

    def __init__(self, name):
        self.name = name
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs.clone().detach() # for backprop
        return self.func(inputs)

    def backward(self, grad):
        return self.d_func(self.inputs) * grad # element-wise

    def func(self, x):
        raise NotImplementedError

    def d_func(self, x):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)

class ReLU(Activation):

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        """
        x: (N, out_feat)
        """
        # Since torch has no maximum function,
        # use numpy to implement here
        return np.maximum(x, 0.0) # element-wise

    def d_func(self, x):
        """
        x: (N, out_feat)
        dx = 1 if x > 0
             0 if x <= 0
        """
        return x > 0.0

class Sigmoid(Activation):

    def __init__(self):
        super().__init__("Sigmoid")

    def func(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def d_func(self, x):
        return self.func(x) * (1.0 - self.func(x))

class Tanh(Activation):

    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def d_func(self, x):
        return 1.0 - self.func(x) ** 2