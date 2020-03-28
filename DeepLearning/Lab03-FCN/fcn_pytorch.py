import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.fcn1 = nn.Linear(28*28,256)
        self.fcn2 = nn.Linear(256,128)
        self.fcn3 = nn.Linear(128,10)

    def forward(self, x):
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.fcn2(x)
        x = F.relu(x)
        x = self.fcn3(x)
        return x

    def reset_parameters(self, params):
        self.fcn1.weight.data = params[0]["w"].T
        self.fcn1.bias.data = params[0]["b"]
        self.fcn2.weight.data = params[1]["w"].T
        self.fcn2.bias.data = params[1]["b"]
        self.fcn3.weight.data = params[2]["w"].T
        self.fcn3.bias.data = params[2]["b"]