import torch

class Loss:

    def loss(self, pred, target):
        raise NotImplementedError

    def grad(self, pred, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, pred, target): # overload
        return self.loss(pred, target)

class CrossEntropyLoss(Loss):
    """
    Softmax + NLLLoss
    https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
    """
    def __init__(self):
        self.probs = None

    def loss(self, pred, target):
        """
        pred: (N, class)
        target: (N, )
            \sum target * log(pred)
        """
        n = pred.shape[0] # batch_size
        shifted_pred = pred - torch.max(pred, axis=1, keepdims=True).values # avoid explosion
        exp = torch.exp(shifted_pred)
        log_probs = shifted_pred - torch.log(torch.sum(exp, axis=1, keepdims=True))
        self.probs = torch.exp(log_probs)
        # https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
        return -torch.sum(log_probs[torch.arange(n),target]) / n # avoid log 0
        # p = exp / torch.sum(exp, axis=1, keepdims=True) # softmax
        # return -torch.sum(torch.log(p[torch.arange(n),target])) / n

    def grad(self, pred, target):
        """
        pred: (N, class)
        target: (N, )
            delta = pred - target
        """
        n = pred.shape[0]
        self.probs[torch.arange(n), target] -= 1
        return self.probs / n
        # one_hot = torch.zeros(probs.shape)
        # one_hot[torch.arange(n),target] = 1
        # return (pred - one_hot) / n