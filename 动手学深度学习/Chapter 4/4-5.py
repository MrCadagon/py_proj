import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

# 存储Tensor
y = torch.ones(3) * 2
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)


# state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


net = MLP()
print('state_dict是一个从参数名称隐射到参数Tesnor的字典对象\n')
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

# 保存加载模型
torch.save(net.state_dict(), 'net.pt')
net2 = MLP()
net2.load_state_dict(torch.load('net.pt'))
