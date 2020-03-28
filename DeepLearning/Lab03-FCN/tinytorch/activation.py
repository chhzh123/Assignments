import torch
import numpy as np

class Activation(): # Base class
    def __init__(self, name):
        self.name = name
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs.clone().detach() # for backprop
        return self.func(inputs)

    def backward(self, grad):
        return self.d_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def d_func(self, x):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)

class ReLU(Activation):
    """docstring for ReLU"""
    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        """
        x: (N, out_feat)
        """
        return np.maximum(x, 0.0) # element-wise

    def d_func(self, x):
        return x > 0.0 # 1 if x > 0 else 0