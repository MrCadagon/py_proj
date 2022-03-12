import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

for name, param in net.named_parameters():
    print(name, param.size())
for name, param in net[0].named_parameters():
    print(name, param.size())


# 关于网络的参数
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)  # 不是参数
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name)

# 查看上述网络的参数
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)
