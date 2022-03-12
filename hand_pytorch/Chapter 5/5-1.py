import torch
import sys
import torch.nn as nn

sys.path.append('.')
import d2lzh_pytorch_5 as d2l

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])

output = d2l.corr2d(X, K)
print(output)

# 边缘检测
X = torch.ones(6, 8)
X[:, 2: 6] = 0
output = d2l.corr2d(X, torch.tensor([[1, -1]]))
print(X)
print(output)


# 自定义二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernal_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernal_size))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias


# 它使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K
conv2D = Conv2D(kernal_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2D(X)
    l = ((Y_hat - output) ** 2).sum()
    l.backward()

    conv2D.weight.data -= lr * conv2D.weight.grad
    conv2D.bias.data -= lr * conv2D.bias.grad

    conv2D.weight.grad.fill_(0)
    conv2D.bias.grad.fill_(0)

    print('Step %d , loss %f' % (i, l.item()))
print("weight: ", conv2D.weight.data)
print("bias: ", conv2D.bias.data)


# 卷积神经网络经常使用奇数高宽的卷积核，如1、3、5和7，所以两端上的填充个数相等。
# 填充
def comp_conv2d(conv2d, X):
    print((1, 1) + X.shape)
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=3, padding=1)
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(5, 3), padding=(2, 1))
X = torch.rand(8, 8)
output = comp_conv2d(conv2d, X)
print(output.shape)

# stride  (a-b+2*c)/w+1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
output = comp_conv2d(conv2d, X).shape
print(output)

