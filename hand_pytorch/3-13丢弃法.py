# 当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。
# 丢弃：与这两个隐藏单元相关的权重的梯度均为0。

import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("../..")
import d2lzh_pytorch as d2l

X = torch.arange(16).view(2, 8)


def droupout(X, drop_prob):
    X = X.float()
    if drop_prob == 0:
        return torch.zeros(X.shape)
    mask = (torch.rand(X.shape) < (1 - drop_prob)).float()

    return X * mask / (1 - drop_prob)


print(droupout(X, 0.2))

# 模型参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens1)), dtype=torch.float32, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_hiddens2)), dtype=torch.float32, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens2, num_outputs)), dtype=torch.float32, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b1, W3, b3]

drop_prob1 = 0.2
drop_prob2 = 0.5


def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    X = (torch.matmul(X, W1) + b1).relu()
    if is_training:
        X = droupout(X, drop_prob1)
    X = (torch.matmul(X, W2) + b2).relu()
    if is_training:
        X = droupout(X, drop_prob2)
    X = (torch.matmul(X, W3) + b3)
