# 修改任务9 在MNIST 数据集上 利用 卷积神经网络 完成多分类任务
# 不需要划分 validation_set ，直接有一个test_dataset 可以测试准确率
# 我的测试准确率达到了98%以上，比上一个全连接网络有所提高，与老师的结果相符
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

# 设置GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The current computing device is {device.type} ")

# prepare dataset

batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
# 处理图像数据的一个转换类 将pillow类转化为tensor, 并将值归一化： 0.1307 和 0.3081 为该数据集的均值和标准差
# 每一个数据为[28,28]的tensor

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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.linear = nn.Linear(320, 10)
        self.layers = nn.Sequential(
            self.conv1,
            self.pooling,
            nn.ReLU(),
            self.conv2,
            self.pooling,
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layers(x)
        # 在转换到全连接层之前需要进行flatten操作,变为[batch_size,320]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


model = Net()
model.to(device)

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
        X, y_label = X.to(device), y_label.to(device)
        # print("debug here: X shape:", X.shape)
        y_pred = model(X)
        loss = criterion(y_pred, y_label)

        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = sum(epoch_loss)/len(epoch_loss)
    loss_list.append(average_loss)
    print(f'[epoch]:{epoch},  [average_loss]: {average_loss}')


def test():
    '''在全集合上测试一次准确率'''
    correct_num = 0
    num = len(test_dataset)
    with torch.no_grad():
        for batch_data in test_loader:
            X, y = batch_data
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_num += torch.sum(y_pred == y).item()
    accuracy = correct_num/num
    accuracy_list.append(accuracy)
    print(f'Current accuracy on the test set is {accuracy}')

# start training now!


num_epochs = 10


for epoch in range(1, num_epochs+1):
    train(epoch)
    test()
