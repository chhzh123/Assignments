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
from mynet import LeNet, VGG
from myutils import EarlyStopping

from tqdm import tqdm
import sys
sys.path.append("../")
from utils import check

"""
Some 'hyperparameters'.
"""
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 32 # Mini-batch size
DEVICE = torch.device('cuda')
CONFIG = {"network": "VGG16",
          "lr": LEARNING_RATE,
          "optim": "Adam",
          "num_epochs": NUM_EPOCHS,
          "batch_size": BATCH_SIZE}

"""
Define a model here.
"""

# model definitions are in mynet.py, check for that!
# model = LeNet().to(device=DEVICE)
model = VGG("VGG16").to(device=DEVICE)
print(model)

"""
Dataset here.
TODO: Implement the dataset and dataloader of CIFAR-10.
HINT: You can refer to dataloader of MNIST in `02.learn-to-count.py`
      and the document in PyTorch official website. Please be attention
      to the channels of images in CIFAR-10 is 3, while in MNIST is 1.
"""
"""YOUR CODE HERE"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # magic numbers
                         std=[0.229, 0.224, 0.225]) # 3 channels
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# if num_workers are used, need to be in __main__
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(patience=5)

"""
Evaluation function here.
"""
def evaluate(model_eval, loader_eval, criterion_eval):
    """
    TODO: Implement the evaluate loop.
    """
    """YOUR CODE HERE"""
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
    """END OF YOUR CODE"""

"""
Training loop here.
"""
# if __name__ == '__main__':
train_acc = np.zeros(NUM_EPOCHS)
eval_acc = np.zeros(NUM_EPOCHS)
train_loss = np.zeros(NUM_EPOCHS)
eval_loss = np.zeros(NUM_EPOCHS)

model.train()
for epoch_idx in range(NUM_EPOCHS):
    """
    TODO: Implement the training loop
    """
    """YOUR CODE HERE"""
    pbar = tqdm(total = len(train_dataloader), desc='Train - Epoch {}'.format(epoch_idx), ncols=100)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()
    """END OF YOUR CODE"""

    train_resp = evaluate(model, train_dataloader, criterion)
    eval_resp = evaluate(model, test_dataloader, criterion)

    print ('-*-*-*-*-*- Epoch {} -*-*-*-*-*-'.format(epoch_idx))
    print ('Train Loss: {:.6f}\t'.format(train_resp['loss']))
    print ('Train Acc: {:.6f}\t'.format(train_resp['acc']))
    print ('Eval Loss: {:.6f}\t'.format(eval_resp['loss']))
    print ('Eval Acc: {:.6f}\t'.format(eval_resp['acc']))
    print ('\n')

    train_acc[epoch_idx] = train_resp['acc']
    eval_acc[epoch_idx] = eval_resp['acc']
    train_loss[epoch_idx] = train_resp['loss']
    eval_loss[epoch_idx] = eval_resp['loss']

    torch.save(model, 'simple-ai-{}.pth'.format(CONFIG["network"]))
    np.savez('results/simple-ai-{}'.format(CONFIG["network"]),
             config=CONFIG, train_acc=train_acc, eval_acc=eval_acc,
             train_loss=train_loss, eval_loss=eval_loss)
    # load by np.load(..., allow_pickle=True)

    if early_stopping(eval_loss[epoch_idx]): # early stopping
        print ("Early stopping at Epoch {}!".format(epoch_idx))
        break