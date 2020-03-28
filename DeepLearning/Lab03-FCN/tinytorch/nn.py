import torch
import torch.nn as nn

class Layer: # Base class
    def __init__(self, name):
        self.name = name
        self.inputs = None
        # self.grads = None
        self.params = None

    def forward(self, inputs):
        self.inputs = inputs.clone().detach() # for backprop
        return self._forward(inputs)

    def backward(self, delta):
        return self._backward(delta)

    def _forward(self, inputs):
        raise NotImplementedError

    def _backward(self, delta):
        raise NotImplementedError

    def parameters(self):
        return self.params

    def __call__(self, inputs):
        return self.forward(inputs)

class Linear(Layer):
    def __init__(self, in_feat, out_feat,
                w_init=nn.init.kaiming_uniform_,
                b_init=nn.init.kaiming_uniform_):
        """
        https://pytorch.org/docs/stable/nn.html#linear
        w: (in_feat, out_feat)
        b: (1, out_feat)
        """
        super().__init__("Linear")

        self.params = {
            "w": w_init(torch.empty([in_feat, out_feat])),
            "b": b_init(torch.empty([1, out_feat])),
            "d_w": torch.zeros([in_feat, out_feat]),
            "d_b": torch.zeros([1, out_feat])
        }
        self.inputs = None

    def _forward(self, inputs):
        """
        inputs: (N, in_feat)
        inputs * w: (N, out_feat)
        b: (1, out_feat) # broadcasting
        """
        return self.inputs @ self.params["w"] + self.params["b"]

    def _backward(self, delta):
        """
        delta: (N, out_feat)
        inputs: (N, in_feat)
        delta_this: (N, in_feat)
        """
        self.params["d_w"] = self.inputs.T @ delta
        self.params["d_b"] = torch.sum(delta, axis=0) # need to sum over batch
        return delta @ self.params["w"].T

class Module(object):
    """NN base class"""
    def __init__(self, name):
        super(Module, self).__init__()
        self.name = name
        
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)