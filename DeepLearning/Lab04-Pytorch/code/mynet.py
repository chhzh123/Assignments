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

vgg_config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
    Ref: Karen Simonyan, Andrew Zisserman
         Very Deep Convolutional Networks for Large-Scale Image Recognition
         ICLR, 2015
    """
    def __init__(self, name):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_config[name])
        self.classifier = nn.Sequential( # three fcns
            nn.Dropout(), # avoid overfitting
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for out_channels in cfg:
            if out_channels == "M": # max pooling
                layers += [nn.MaxPool2d(2)]
            else:
                # preserve image resolution
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels), # avoid overfitting
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)