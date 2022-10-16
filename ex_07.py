import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 这是一个二分类任务 本质其实和回归差不多

data = np.loadtxt('diabetes.csv', delimiter=',',
                  dtype=np.float32)  # 大部分显卡仅支持float32， 且够用

x_data = torch.tensor(data[:, :-1])
y_data = torch.tensor(data[:, [-1]])    # 此步为保持矩阵维度，否则会变为一维

print(x_data.shape,y_data.shape)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.linear1 = nn.Linear(8,6)
        self.linear2 = nn.Linear(6,4)
        self.linear3 = nn.Linear(4,2)
        self.linear4 = nn.Linear(2,1)
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
    
def accuracy(pred,labels):
    pred = pred.data
    pred = torch.where(pred >= 0.5,torch.tensor([1.0]),torch.tensor([0.0]))
    pred = pred.squeeze(1)
    labels = labels.data.squeeze(1)
    logits = 1/pred.shape[0] * sum(pred == labels).item()
    return logits

model = Model()
criterion = nn.BCELoss(reduction= 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1) # Adam优化器比SGD有更好的效果！差异明显

# SGD会使得loss大约在0.6附近不再下降，可能是遇到鞍点等原因

num_epochs = 1000
epoch_list = []
loss_list = []
accuracy_list = []

for epoch in range(1, num_epochs+1):
    # 前向计算
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    metric = accuracy(y_pred,y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    accuracy_list.append(metric)
    
    if epoch % 50 == 0:
        print(f'[epoch]:{epoch}, [loss]:{loss.item()}, [accuracy]:{metric}')
    
    #反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.subplot(1,2,2)
plt.plot(epoch_list,accuracy_list)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
    
