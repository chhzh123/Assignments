import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

n = 28 * 28
C = 10
DOWNLOAD_MNIST = True
BATCH_SIZE = 100
EPOCH = 100
alpha = 0.01
PATH = "./mnist/"

train_data = torchvision.datasets.MNIST(
    root=PATH,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(root=PATH, train=False)
# shape from (60000, 28, 28) to (60000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255.
test_y = test_data.targets

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class Logistic(nn.Module):
	def __init__(self):
		super(Logistic, self).__init__()
		self.linear = nn.Linear(n,C)
		self.softmax = nn.Softmax(dim=1)

	def forward(self,x):
		x = self.linear(x)
		x = self.softmax(x)
		return x

logits = Logistic()
cross_entropy = nn.CrossEntropyLoss()

Accurate = []
Astore = []
bstore = []
A, b = [i for i in logits.parameters()]
for e in range(EPOCH):
    for step, (x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = x.view(-1, 28 * 28)  # reshape x to (batch, time_step, input_size)

        output = logits(b_x) # logits output
        loss = cross_entropy(output, b_y) # cross entropy loss
        if A.grad is not None:
            A.grad.zero_()
            b.grad.zero_()
        loss.backward() # backpropagation, compute gradients

        A.data = A.data - alpha * A.grad.data
        b.data = b.data - alpha * b.grad.data
        if step % 1000 == 0:
            test_output = logits(test_x.view(-1, 28 * 28))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            Accurate.append(sum(test_y.numpy() == pred_y.cpu().numpy()) / (1.0 * len(test_y.cpu().numpy())))
            print(Accurate[-1])
            Astore.append(A.detach())
            bstore.append(b.detach())
test_output = logits(test_x.view(-1, 28 * 28))
pred_y = torch.max(test_output, 1)[1].data.squeeze()

print(pred_y, 'prediction number')
print(test_y, 'real number')
Accurate.append(sum(test_y.numpy() == pred_y.numpy()) / (1.0 * len(test_y.numpy())))
print(Accurate[-1])

for i in range(len(Astore)):
    Astore[i] = (Astore[i] - Astore[-1]).norm()
    bstore[i] = (bstore[i] - bstore[-1]).norm()

plt.plot(Astore, label='A')
plt.plot(bstore, label='b')
plt.legend()
plt.show()
plt.cla()
plt.plot(Accurate)
plt.show()