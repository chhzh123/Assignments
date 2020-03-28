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
        """
        https://d2l.ai/chapter_optimization/momentum.html
        """
        for param in reversed(self.params):
            for item in ["w", "b"]:
                if param.get("v_{}".format(item),None) == None:
                    param["v_{}".format(item)] = torch.zeros(param[item].shape)
                param["v_{}".format(item)] = self.momentum * param["v_{}".format(item)] - self.lr * param["d_{}".format(item)]
                param[item] += param["v_{}".format(item)]