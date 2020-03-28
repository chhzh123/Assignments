# 代码实现了搭建了网络训练的基本框架
# 在此代码的基础上，需要实现下面的内容(请留意TODO):
# 1、实现全连接神经网络的前向传播和反向传播(不使用torch.nn搭建网络，不使用backward方法进行反传)
# 2、实现交叉熵损失函数(不使用torch.nn.CrossEntropyLoss)
# 3、实现带动量的SGD优化器(不使用torch.optim.SGD)
# 代码可根据自己需要修改，实现上述内容即可
# 提示:
# 在实现过程中，可使用xxx.shape观察网络和数据的维度
# 可以将自己实现的输出与pytorch函数的输出进行比较(如损失函数与优化器)，观察自己的模块是否正常工作

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import sys
import tinytorch.nn as nn
from tinytorch.activation import ReLU
from tinytorch.loss import CrossEntropyLoss
from tinytorch.optim import SGD

from fcn_pytorch import TorchNet
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss
from torch.optim import SGD as TorchSGD

# TODO:在这里你需要实现一些类来实现上述三个内容
# 类的设计并无具体要求，能实现所需功能即可
# 比如，可以考虑先构建单层全连接层Layer类，再组成整体网络Net类
# 可单独设置Loss类与SGD类，也可以将这些功能的实现放到Net类中

# 一种可能的类的设计为
# TODO:在这里实现全连接神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__("FCN")
        self.fcn1 = nn.Linear(28*28,256)
        self.fcn2 = nn.Linear(256,128)
        self.fcn3 = nn.Linear(128,10)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.add_layers([self.fcn1,
                         self.fcn2,
                         self.fcn3])

    def forward(self,x):
        x = self.fcn1(x)
        x = self.relu1(x)
        x = self.fcn2(x)
        x = self.relu2(x)
        x = self.fcn3(x)
        return x

    def backward(self,delta):
        grad = self.fcn3.backward(delta)
        delta = self.relu2.backward(grad)
        grad = self.fcn2.backward(delta)
        delta = self.relu1.backward(grad)
        grad = self.fcn1.backward(delta)

# class Loss: # TODO:在这里实现交叉熵损失函数
# 已在tinytorch.loss.CrossEntropyLoss中实现

# class SGD: # TODO:在这里实现SGD优化器
# 已在tinytorch.optim.SGD中实现

# 对训练过程的准确率和损失画图
def training_process(train_loss, train_acc, test_acc,
                     train_loss_torch, train_acc_torch, test_acc_torch):
    shape = train_loss.shape[0]
    epoch = np.arange(1, shape+1)

    plt.plot(epoch, train_acc, label="trainAcc (TinyTorch)")
    plt.plot(epoch, test_acc, label="testAcc (TinyTorch)")
    plt.plot(epoch, train_acc_torch, label="trainAcc (PyTorch)")
    plt.plot(epoch, test_acc_torch, label="testAcc (PyTorch)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on train set and test set")
    plt.legend()
    plt.show()

    plt.plot(epoch, train_loss, label="TinyTorch")
    plt.plot(epoch, train_loss_torch, label="PyTorch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss on train set")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # 可直接使用这组超参数进行训练，也可以自己尝试调整
    lr = 0.02  # 学习率
    epoch = 20  # 迭代次数
    batch_size = 128  # 每一批次的大小

    # 训练数据的记录
    train_acc = np.zeros(epoch)
    test_acc = np.zeros(epoch)
    train_loss = np.zeros(epoch)

    train_acc_torch = np.zeros(epoch)
    test_acc_torch = np.zeros(epoch)
    train_loss_torch = np.zeros(epoch)

    # 对数据集图片做标准化并转为tensor
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])  # 对训练集的transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])  # 对测试集的transform

    print("Begin loading images...")
    # 借助torchvision中的函数读取MNIST，请将参数root换为自己数据存放的路径，或者设置download=True下载数据集
    # 读MNIST训练集
    trainSet = MNIST(root=".", train=True, transform=transform_train, download=False)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4)
    # 读MNIST测试集
    testSet = MNIST(root=".", train=False, transform=transform_test, download=False)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Finish loading images")

    # TODO:在这里对你实现的类进行实例化，之后开始对模型进行训练
    net = Net()  # 具体的实例化根据你的实现而定，此处只做示意(包括下面两行)
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=lr, momentum=0.9)

    # pytorch version
    torch_net = TorchNet()
    torch_criterion = TorchCrossEntropyLoss()
    torch_optimizer = TorchSGD(torch_net.parameters(), lr=lr, momentum=0.9)

    # 重复训练epoch次
    print("Begin training...")
    print("Batch size: {}\t # of batches: {}".format(batch_size,len(trainLoader)))
    for epo in range(epoch):
        epoch_loss = 0  # 当前epoch的损失
        correct1 = 0  # 当前epoch的训练集准确率
        correct2 = 0  # 当前epoch的测试集准确率

        epoch_loss_torch = 0  # 当前epoch的损失
        correct1_torch = 0  # 当前epoch的训练集准确率
        correct2_torch = 0  # 当前epoch的测试集准确率

        # 训练阶段
        # 利用每个mini-batch对网络进行更新
        for index, (data, label) in enumerate(trainLoader):  # 从trainLoader读取一个mini-batch
            # index是当前mini-batch的序号，data是图像，label是标签，data和label都有batch_size个
            data = data.view(data.size(0), -1)  # 展开，将输入的维度从[batch_size, 1, 28, 28]变成[batch_size, 784]

            def train(net, criterion, optimizer, data, label, name="TinyTorch"):
                optimizer.zero_grad() # 添加！
                output = net(data)  # TODO:完成前向传播，其中net是你实现的三层全连接神经网络，具体调用形式根据你的实现而定(包括下面三个)

                # 计算训练集准确率，output是网络的输出，维度应为[batch_size, 10]
                _, prediction = torch.max(output.data, 1)
                correct = (prediction == label).sum()

                loss = criterion(output, label)  # TODO:计算损失
                # torch_loss = torch_criterion(torch_output,label)
                if name == "TinyTorch":
                    net.backward(criterion.grad(output,label))
                else:
                    loss.backward()  # TODO:完成反向传播（计算梯度）
                optimizer.step()  # TODO:实现网络参数的更新

                return correct, loss.item()

            correct1_epoch, loss = train(net,criterion,optimizer,
                                   data.clone().detach(),label,"TinyTorch")
            correct1_epoch_torch, loss_torch = train(torch_net,
                                   torch_criterion,torch_optimizer,
                                   data.clone().detach(),label,"PyTorch")

            correct1 += correct1_epoch
            correct1_torch += correct1_epoch_torch
            epoch_loss += loss # 加上当前batch的损失
            epoch_loss_torch += loss_torch

        # 测试阶段
        # 测试时不需要tensor的梯度，可调用no_grad关掉梯度
        with torch.no_grad():
            for index, (data, label) in enumerate(testLoader):# 从testLoader读取一个mini-batch
                data = data.view(data.size(0), -1)
                output = net(data)  # 与上面对前向传播的实现保持一致
                output_torch = torch_net(data)

                # 计算测试集准确率
                _, prediction = torch.max(output.data, 1)
                correct2 += (prediction == label).sum()

                # 计算测试集准确率
                _, prediction_torch = torch.max(output_torch.data, 1)
                correct2_torch += (prediction_torch == label).sum()

        # 计算训练集和测试集准确率
        epoch_train_acc = (int(correct1) * 100 / 60000)
        epoch_test_acc = (int(correct2) * 100 / 10000)

        epoch_train_acc_torch = (int(correct1_torch) * 100 / 60000)
        epoch_test_acc_torch = (int(correct2_torch) * 100 / 10000)

        # 输出当前epoch的信息
        print("-------%2d-------" % epo)
        print("            TinyTorch   PyTorch")
        print("Epoch loss: %4.2f\t%4.2f" % (epoch_loss,epoch_loss_torch))
        print("Train acc:  %3.2f%%\t%3.2f%%" % (epoch_train_acc,epoch_train_acc_torch))
        print("Test acc:   %3.2f%%\t%3.2f%%" % (epoch_test_acc,epoch_test_acc_torch))
        print()

        # 记录loss和accuracy
        train_acc[epo] = epoch_train_acc
        test_acc[epo] = epoch_test_acc
        train_loss[epo] = epoch_loss

        train_acc_torch[epo] = epoch_train_acc_torch
        test_acc_torch[epo] = epoch_test_acc_torch
        train_loss_torch[epo] = epoch_loss_torch

        # 至此当前epoch结束

    # 当所有epoch结束后，对训练过程中的损失和准确率进行画图
    training_process(train_loss, train_acc, test_acc, train_loss_torch, train_acc_torch, test_acc_torch)

    # 如果需要，在训练结束时对模型和数据进行保存
    # 由于本次的模型是自定义的小模型，可考虑使用torch.save对整个模型进行保存(可保存为tar格式)
    # 训练过程的数据可使用numpy的save(savez)进行保存
    # 比如:
    # torch.save(net, model_path)
    # np.savez(data_path, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss)
