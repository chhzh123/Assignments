"""
Question #1
    You should finish the code here.
"""

import math

class Solution(object):
    """
    TODO: Finish the activition functions here.
    """
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def relu(self, x):
        return x if x >= 0 else 0

    def leaky_relu(self, alpha, x):
        return x if x >= 0 else (alpha * x)

    def elu(self, alpha, x):
        return x if x >= 0 else (alpha * (math.exp(x) - 1))