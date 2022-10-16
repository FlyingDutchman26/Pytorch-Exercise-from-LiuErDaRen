import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# 用Dataset DataLoader 重写ex_07的二分类任务， 使用mini-batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1012)


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = np.loadtxt(filepath, delimiter=',',
                          dtype=np.float32)  # 大部分显卡仅支持float32加速
        self.x_data = torch.tensor(data[:, :-1])
        self.y_data = torch.tensor(data[:, [-1]])    # 此步为保持矩阵维度，否则会变为一维
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


filepath = 'diabetes.csv'
dataset = DiabetesDataset(filepath)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 2)
        self.linear4 = nn.Linear(2, 1)
        self.layers = nn.Sequential(
            self.linear1,
            nn.Sigmoid(),
            self.linear2,
            nn.Sigmoid(),
            self.linear3,
            nn.Sigmoid(),
            self.linear4,
            nn.Sigmoid()
        )

    def forward(self, x):
        y_pred = self.layers(x)
        return y_pred


def accuracy(pred, labels):
    pred = pred.data
    pred = torch.where(pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
    pred = pred.squeeze(1)
    labels = labels.data.squeeze(1)
    logits = 1/pred.shape[0] * sum(pred == labels).item()
    return logits


model = Model()
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01)  # Adam优化器比SGD有更好的效果！差异明显

# SGD会使得loss大约在0.6附近不再下降，可能是遇到鞍点等原因

num_epochs = 200
epoch_list = []
loss_list = []
accuracy_list = []

for epoch in range(1, num_epochs+1):
    for i, batch_data in enumerate(train_loader, 0):
        X, y = batch_data
        y_pred = model(X)
        loss = criterion(y_pred, y)
        print(f'[epoch]:{epoch}, [iterate]:{i}, [loss]:{loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
