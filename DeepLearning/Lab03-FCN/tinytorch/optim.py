import torch

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param["d_w"] = torch.zeros(param["w"].shape)
            param["d_b"] = torch.zeros(param["b"].shape)

    def step(self):
        return self._step()

    def _step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum

    def step(self):
        for param in reversed(self.params):
            param["w"] -= self.lr * param["d_w"]
            param["b"] -= self.lr * param["d_b"]