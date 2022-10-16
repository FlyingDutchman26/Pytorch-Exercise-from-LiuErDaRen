# 利用MNIST 手写数字识别数据集 实现一个完善的 多分类任务的 全连接神经网络
# 不设置 validation_set ，直接在整个集合上测试分类准确度
# 我的实验结果准确率达到了了97%，与老师的实验结果相符合

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 初始化并固定随机种子


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1012)

# prepare dataset

batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
# 处理图像数据的一个转换类 将pillow类转化为tensor, 并将值归一化： 0.1307 和 0.3081 为该数据集的均值和标准差

train_dataset = datasets.MNIST(
    root='./dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(
    root='./dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False,
                         batch_size=len(test_dataset))  # 测试肯定是

# design model


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)
        self.layers = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.ReLU(),
            self.l4,
            nn.ReLU(),
            self.l5
            # 最后一层softmax之后会自动接在损失函数之上
        )

    def forward(self, x):
        x = x.view(-1, 784)  # 吧minibatch reshape to [N, 784], 784是每个图像的特征维度
        return self.layers(x)


model = Net()

# construct loss and optimiter

# 包含了softmax层，并且会根据标签类别（即使是多类）,自动构建one-hot计算交叉熵，需要LongTensor类标签
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# training and test

loss_list = []
accuracy_list = []


def train(epoch):
    '''某一轮epoch上的训练'''
    epoch_loss = []  # 记录该轮epoch上每个batch的loss
    for batch_idx, batch_data in enumerate(train_loader, 1):
        X, y_label = batch_data
        y_pred = model(X)
        loss = criterion(y_pred, y_label)

        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = sum(epoch_loss)/len(epoch_loss)
    print(f'[epoch]:{epoch},  [loss]: {average_loss}')


def test():
    '''在全集合上测试一次准确率'''
    correct_num = 0
    num = len(test_dataset)
    with torch.no_grad():
        for batch_data in test_loader:
            X, y = batch_data
            y_pred = model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_num += torch.sum(y_pred == y).item()
    accuracy = correct_num/num
    accuracy_list.append(accuracy)
    print(f'Current accuracy on the whole set is {accuracy}')

# start training now!


num_epochs = 10


for epoch in range(1, num_epochs+1):
    train(epoch)
    test()
