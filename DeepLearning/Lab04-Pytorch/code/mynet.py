import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    TODO: Implementation of a simple Convolutional neural network.
    HINT: You can refer to the baby model in `01.toy.py`, and
          the document of PyTorch from the official website.
    """
    """YOUR CODE HERE"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential( # 3*32*32
            nn.Conv2d(3, 6, 5), # 6*28*28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 6*14*14
            nn.Conv2d(6, 16, 5), # 16*10*10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 16*5*5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x
    """END OF YOUR CODE"""