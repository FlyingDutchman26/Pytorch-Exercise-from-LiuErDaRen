# RNN exercise

import torch.nn as nn
import torch
batch_size = 5
seq_len = 6
input_size = 4
hidden_size = 3
num_layer = 2

cell = nn.RNN(input_size=input_size,
              hidden_size=hidden_size, num_layers=num_layer)
# 定义一个RNN模块类，输入的参数是为了确定RNN中参数的数量

# dataloader 的每一次迭代输入满足此格式
inputs = torch.rand(seq_len, batch_size, input_size)

hidden = torch.zeros(num_layer, batch_size,
                     hidden_size)    # 初始化一个hidden vector

out, hidden = cell(inputs)
# 每一个 output 的 size 满足 [seq_len,,batch_size,hidden_size]

print('Output Size:', out.shape)
print('Output:', out)
print('Hidden size:', hidden.shape)
print('Hidden:', hidden)
