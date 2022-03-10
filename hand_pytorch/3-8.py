# 引入非线性变换 然后再作为下一个全连接层的输入


import torch
import numpy as np
import matplotlib.pylab as plt
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()


# Relu
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'Relu')
y.sum().backward()
xyplot(x, x.grad, 'Relu2')

# sigmoid
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'sigmoid2')

# tanh
y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'tanh2')
