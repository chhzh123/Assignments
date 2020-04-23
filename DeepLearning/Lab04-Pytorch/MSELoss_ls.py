import numpy as np

def MSELoss_ls(input,target,smooth_factor):
    """
    MSE Loss with label smoothing
    input: (n * C)
    target: (n,)
    return: scalar

    Loss(x,smooth_factor)=(x_i-smooth_factor)^2+sum_{j in C,j ne i}(x_j - (1-smooth_factor)/(C-1))^2
    Loss(X,smooth_factor)=sum_{x in X}Loss(x,smooth_factor)
    """
    n, C = input.shape
    one_hot = np.zeros(input.shape)
    one_hot[np.arange(n),target] = 1
    correct = (input[np.arange(n),target] - smooth_factor) ** 2
    wrong = ((input - (1 - smooth_factor) / (C - 1)) ** 2 * (1 - one_hot)).sum(axis=1)
    return np.sum(correct + wrong)