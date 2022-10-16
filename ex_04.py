import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return w * x


def loss(x, y):
    y_pred = forward(x)
    loss = (y_pred - y)**2
    return loss


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()                           # backward()会把计算图中所有需要梯度的地方都会求出来，最终计算图被释放。
        w.data = w.data - 0.01 * w.grad.data   # .data 是为了取出值进行tensor运算，否则会建立计算图！
        w.grad.data.zero_()                    # backward()产生梯度后，不会每次覆盖，会默认累加（因为某些原因），因此这里需要清零

    # .item() 是将其转化为python标量（返回值是tensor）
    print('[epoch]:', epoch, 'loss:', l.item())

print("predict (after training)", 4, forward(4).item())
