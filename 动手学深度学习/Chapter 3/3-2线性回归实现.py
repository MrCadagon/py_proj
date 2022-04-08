import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

num_examples = 1000
num_inputs = 2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
true_w = [2, -3, 4]
true_b = 4.2
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
print(features[0], labels[0])

for X, y in d2l.data_iter(10, features, labels):
    print(X, y)
    break

# 训练模型
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
lr = 0.03
num_epochs = 3
batch_size = 10
net = d2l.linreg
loss = d2l.squared_loss
opt = d2l.sgd
for epoch in range(num_epochs):
    for X, y in d2l.data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        opt([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l_vec = loss(net(features, w, b), labels)
    print(train_l_vec.mean())
