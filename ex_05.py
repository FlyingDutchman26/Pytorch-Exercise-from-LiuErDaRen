from re import L
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearModel, self).__init__()
        # torch.nn.Linear 这也是一个类，同样继承于 torch.nn.Module，可以自动生成计算图
        self.Linear = torch.nn.Linear(1, 1)  # 实例化这个线性层 从 1维 到 1维 ，会生成对应参数（自动计算梯度）

    def forward(self, x):
        y_pred = self.Linear(x)     # 继承于torch.nn.Module的类都具有__call__()方法
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction= 'sum')  # 意思是不取平均，这是新版写法， size_average已被取代
# SGD优化器实现梯度下降， 梯度计算已经储存于model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000

for epoch in range(1,num_epochs+1):
    # 前向计算
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    # 清空上一次计算的梯度
    optimizer.zero_grad()
    
    #反向传播
    loss.backward()
    
    #参数更新
    optimizer.step()
    
    #打印状态
    print(f'[epoch]:{epoch} loss = {loss.item()}')
    
    
print('w = ', model.Linear.weight.item())
print('b = ', model.Linear.bias.item())
 
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

    