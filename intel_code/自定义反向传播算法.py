import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 1.初始化网络层参数
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        # 2.定义网络层结构
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        # 3.定义所有网络层
        super(Perception, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Linear(hid_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 4.定义模型
        y = self.layer(x)
        return y

perception = Perception(100,1000,100)
print(perception)

