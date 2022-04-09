import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

num_examples = 1000
num_inputs = 2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
true_w = [2, -3, 4]
true_b = 4.2
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# 读取数据
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
for x, y in data_iter:
    print(x)
    print(y)
    break


# 定义模型1
class LinearNet(nn.Module):
    def __init__(self, in_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


net = LinearNet(10)
print(net)

# 定义模型 2 3
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print(net)

# 查看参数
for param in net.parameters():
    print(param)

# 初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 定义损失函数 优化算法
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 不同网络设定不用学习率
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)
# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))




