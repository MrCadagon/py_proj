import torch
from torch import nn
import sys

sys.path.append("../")
import d2lzh_pytorch as d2l


# 输入多通道
def corr2d_multi_in(X, K):
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

output = corr2d_multi_in(X, K)
print(output)


# 多输出通道
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
print(K.shape)  # torch.Size([3, 2, 2, 2])

output = corr2d_multi_in_out(X, K)
print(output)


# 1×1卷积的主要计算发生在通道维上
def corr2d_multi_in_out_1X1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, -1)
    K = K.view(c_o, -1)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1X1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)

# 池化层：为了缓解卷积层对位置的过度敏感性。
import torch
from torch import nn


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

# stride padding
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

# ！赤化层不改变通道
X = torch.cat((X, X + 1), dim=1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

