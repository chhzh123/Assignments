"""
A SMALLEST TOY MODEL.
    You will learn how to create an app from scrach with PyTorch.
    In this app, you will train a model that can simulate XOR operation.
"""
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("../")
from utils import check

"""
Some 'hyperparameters'.
"""
LEARNING_RATE = 1e-1
NUM_EPOCHS = 1000
DEVICE = torch.device('cpu')

"""
Define a model here.
"""
class baby(nn.Module):
    """
    A really simple baby network with only two fully connected layers
        and one nonlinear layer. But it's adequate to do the job here,
        `Learning the XOR operation`, which is hard to solve using 
        linear regression and logistic regression as you learned before.
    Structure:
        (input) - FC1 - ReLU - FC2 - (output)
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(baby, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output

model = baby(2, 10, 2).to(device=DEVICE)

"""
Dataset here.
"""
dataset = ([[0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]],
           [0, 1, 1, 0])

"""
Define a criterion here.(Loss function)
"""

def mse_loss(input, target):
    """
    20 pts.
    TODO: Implementation of mean squared error loss.
    HINT: Please pay attention to the dimensions of input and output.
    """

    """YOUR CODE HERE"""
    pass
    """END OF YOUR CODE"""

criterion = mse_loss

# You should find that the error between your implementation and official
# implementation is very small.
check.check_mse_loss(criterion)

"""
Define an optimizer here.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

"""
Training loop here.
"""
for epoch_idx in range(NUM_EPOCHS):
    """
    The full routine of one epoch in training: 
        1. Get data and label, check format.
        2. Forward propagation, get output score.
        3. Compute loss with your criterion.
        4. Run back propagation, update parameters.
        5. Other affairs.
    """
    """
    1. Get data and label. Check format.
        You should pay attention to the format of data and label here.
        (a) dtype: generally, the `data` should be of dtype torch.float32,
                   and the `label` should be of dtype torch.long
        (b) device: `data` and `label` should be in the same device.
        (c) shape: `data` should be a tensor of shape (N, M), where N means
                   the number of data (number of images in this dataset e.g.),
                   and M means the dimension of each data (pixel number of each
                   image e.g.). In this toy app, N is 4 and M is 2.
                   
                   `label` should be a tensor of shape (N,), (but not (N,1) or (1,N)).
    """
    data, label = dataset
    data = torch.tensor(data).to(DEVICE)
    label = torch.tensor(label).to(DEVICE)

    """
    2. Forward propagation, get output score.
    """
    output = model(data)

    """
    3. Compute loss with your criterion.
    """
    loss = criterion(output, label)


    """
    4. Run back propagation, update parameters.
    """
    optimizer.zero_grad() # Clear the gradients stored in the optimizer.
    loss.backward()       # Run backprop with loss. Thanks to PyTorch, we can ignore the detailed grad computation here.
    optimizer.step()      # Update the parameters in the model.

    """
    5. Other affairs.
        Print a log, write some result to tensorboard, update learning rate with scheduler, save the model checkpoint etc.
    """
    if epoch_idx % 100 == 0:
        print('-*-*-*-*-*- Epoch {} -*-*-*-*-*-\n'.format(epoch_idx))
        print('Output:')
        print(output.detach().numpy())
        print()
        print('Pred:')
        print(output.argmax(dim=1, keepdim=True).detach().numpy())
        print()
        print('loss: {}'.format(loss))
        print('\n')


torch.save(model, 'toy.pth')