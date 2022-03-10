import torch
import numpy as np

# 广播机制
# 向x y轴做广播
x = torch.arange(1, 3).view(1, -1)
y = torch.arange(1, 4).view(-1, 1)
print(x + y)

# 运算开销
# waste other new space
y = y + x
# move into source space save memory
y[:] = y + x
torch.add(x, y, out=y)
# y += x(y.add_(x))

# 相互的转换
# 共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！
a = torch.ones(5)
b = a.numpy()
c = torch.from_numpy(b)
print(a, b, c)
