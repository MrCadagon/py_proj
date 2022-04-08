# 数据操作
import torch

# 随机数据
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 300])
print(x)

# 获取维度相同的tensor
x = x.new_ones(5, 3, dtype=torch.float64)
x = torch.rand_like(x, dtype=torch.float)
print(x)

# 获取形状——函数，变量
print(x.shape)
print(x.shape[0])
print(x.shape[1])
print(x.size())
print(x.size(0))
print(x.size(1))
