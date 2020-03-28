import torch

class Optimizer(object): # Base class
    def __init__(self, params, lr):
        """
        params: All network layer parameters (in a list),
                needed to be added before training,
                stored by references/objects, thus can be modified later
                For linear layer, there are w, b, d_w, d_b four params.
        lr: learning rate
        """
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """
        Needed to be call before each epoch
        """
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
        SGD with momentum, need to store v for each param

        v_{t+1}=\gamma v_t - \eta\nabla L(\theta_t)
        \theta_{t+1} = \theta_t + v_{t+1}

        Ref: https://d2l.ai/chapter_optimization/momentum.html
        """
        for param in reversed(self.params):
            for item in ["w", "b"]:
                if param.get("v_{}".format(item),None) == None:
                    param["v_{}".format(item)] = torch.zeros(param[item].shape)
                param["v_{}".format(item)] = self.momentum * param["v_{}".format(item)] - self.lr * param["d_{}".format(item)]
                param[item] += param["v_{}".format(item)]