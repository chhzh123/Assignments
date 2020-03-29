"""
A SIMPLE AI
    You will implement a full PyTorch project here. You will train
    a classifier on CIFAR-10. You will achieve at least 60% accuracy
    on CIFAR-10 test dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

from tqdm import tqdm
import sys
sys.path.append("../")
from utils import check

"""
Some 'hyperparameters'.
"""
LEARNING_RATE = 1e-1
NUM_EPOCHS = 100
BATCH_SIZE = 20 # Mini-batch size
DEVICE = torch.device('cpu')

"""
Define a model here.
"""
class Net(nn.Module):
    """
    TODO: Implementation of a simple Convolutional neural network.
    HINT: You can refer to the baby model in `01.toy.py`, and
          the document of PyTorch from the official website.
    """
    """YOUR CODE HERE"""
    pass
    """END OF YOUR CODE"""

model = Net().to(device=DEVICE)

"""
Dataset here.
TODO: Implement the dataset and dataloader of CIFAR-10.
HINT: You can refer to dataloader of MNIST in `02.learn-to-count.py`
      and the document in PyTorch official website. Please be attention
      to the channels of images in CIFAR-10 is 3, while in MNIST is 1.
"""
train_dataloader = None
test_dataloader = None
"""YOUR CODE HERE"""
pass
"""END OF YOUR CODE"""


"""
Define a criterion here.(Loss function)
CAN BE MODIFIED.
"""

criterion = F.cross_entropy

"""
Define an optimizer here.
CAN BE MODIFIED.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

"""
Evaluation function here.
"""
def evaluate(model_eval, loader_eval, criterion_eval):
    """
    TODO: Implement the evaluate loop.
    """
    """YOUR CODE HERE"""
    pass
    """END OF YOUR CODE"""

"""
Training loop here.
"""
model.train()
for epoch_idx in range(NUM_EPOCHS):
    """
    TODO: Implement the training loop
    """
    """YOUR CODE HERE"""
    pass
    """END OF YOUR CODE"""

    train_resp = evaluate(model, train_dataloader, criterion)
    eval_resp = evaluate(model, test_dataloader, criterion)

    print ('-*-*-*-*-*- Epoch {} -*-*-*-*-*-'.format(epoch_idx))
    print ('Train Loss: {:.6f}\t'.format(train_resp['loss']))
    print ('Train Acc: {:.6f}\t'.format(train_resp['acc']))
    print ('Eval Loss: {:.6f}\t'.format(eval_resp['loss']))
    print ('Eval Acc: {:.6f}\t'.format(eval_resp['acc']))
    print ('\n')

    torch.save(model, 'simple-ai.pth')