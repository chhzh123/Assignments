"""
LEARN TO COUNT
    You will learn how to use dataset and dataloader to
    deal with data in PyTorch. In this app, you will train
    a classifier that can recognize hand-writen numbers.
    You will train a simple model with MNIST dataset.
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
NUM_EPOCHS = 20
BATCH_SIZE = 20 # Mini-batch size
DEVICE = torch.device('cuda')

"""
Define a model here.
"""
class Net(nn.Module):
    """
    30 pts.
    TODO: Implementation of a simple Convolutional neural network.
    HINT: You can imitate the baby model in `01.toy.py`, and refer to
          the document of PyTorch from the official website.
    """
    """YOUR CODE HERE"""
    def __init__(self):
        super(Net, self).__init__()
        # in\_chan, out\_chan, kernel\_size
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(26 * 26 * 8, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = output.view(output.shape[0],-1)
        output = self.fc(output)
        return output
    """END OF YOUR CODE"""

model = Net().to(device=DEVICE)

"""
Dataset here.
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""
Define a criterion here.(Loss function)
"""

criterion = F.cross_entropy

"""
Define an optimizer here.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

"""
Evaluation function here.
"""
def evaluate(model_eval, loader_eval, criterion_eval):
    model_eval.eval()
    loss_eval = 0
    correct = 0.
    pbar = tqdm(total = len(loader_eval), desc='Evaluation', ncols=100)
    with torch.no_grad():
        for data, target in loader_eval:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss_eval += criterion_eval(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update(1)
    pbar.close()

    loss_eval = loss_eval / loader_eval.dataset.__len__()
    accuracy = correct / loader_eval.dataset.__len__()
    response = {'loss': loss_eval, 'acc': accuracy}
    return response

"""
Training loop here.
"""
model.train()
for epoch_idx in range(NUM_EPOCHS):
    pbar = tqdm(total = len(train_dataloader), desc='Train - Epoch {}'.format(epoch_idx), ncols=100)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(1)
    pbar.close()

    train_resp = evaluate(model, train_dataloader, criterion)
    eval_resp = evaluate(model, test_dataloader, criterion)

    print ('-*-*-*-*-*- Epoch {} -*-*-*-*-*-'.format(epoch_idx))
    print ('Train Loss: {:.6f}\t'.format(train_resp['loss']))
    print ('Train Acc: {:.6f}\t'.format(train_resp['acc']))
    print ('Eval Loss: {:.6f}\t'.format(eval_resp['loss']))
    print ('Eval Acc: {:.6f}\t'.format(eval_resp['acc']))
    print ('\n')
    torch.save(model, 'count.pth')