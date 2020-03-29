import torch

class Loss(object): # Base class

    def loss(self, pred, target):
        raise NotImplementedError

    def grad(self, pred, target):
        raise NotImplementedError

    def __call__(self, pred, target): # overload
        return self.loss(pred, target)

class CrossEntropyLoss(Loss):
    """
    Softmax + NLLLoss
    Ref: https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
    """
    def __init__(self):
        self.probs = None # used for backprop

    def loss(self, pred, target):
        """
        pred: (N, class)
        target: (N, )

        loss(x,class) = -x[class] + log(\sum_j exp(x[j]))

        Ref: https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
        """
        n = pred.shape[0] # batch_size
        shifted_pred = pred - torch.max(pred, axis=1, keepdims=True).values # avoid explosion
        exp = torch.exp(shifted_pred)
        log_probs = shifted_pred - torch.log(torch.sum(exp, axis=1, keepdims=True)) # avoid log 0
        self.probs = torch.exp(log_probs) # stored for backprop
        return -torch.sum(log_probs[torch.arange(n),target]) / n

    def grad(self, pred, target):
        """
        pred: (N, class)
        target: (N, )
            delta = pred - target
        """
        n = pred.shape[0]
        self.probs[torch.arange(n), target] -= 1 # one-hot encoding
        return self.probs / n