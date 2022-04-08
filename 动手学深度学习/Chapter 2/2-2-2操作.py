import torch

y = torch.rand(5, 3)
x = torch.rand(5, 3)
out = x + y
print(x + y)
# print(x.add_(y))
# print(torch.add(x, y))
output = torch.empty(5, 3)
# print(torch.add(x, y, out=output))

# #索引--引用
# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y = x[0, :]
y += 1
print(y, x[0, :])

# #改变大小--引用
# 注意view()返回的新Tensor与源Tensor虽然可能有不同的size，
# 但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。
x = torch.rand(5, 3)
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())

# 不共享data内存
y = x.clone().view(15)
y += 1
print(y, '\n', x)

# item(), 它可以将一个标量Tensor转换成一个Python number：
x = torch.rand(1)
y = x.item()
print(x, y)

# !!常用函数
# trace	对角线元素之和(矩阵的迹)
# diag	对角线元素
# triu/tril	矩阵的上三角/下三角，可指定偏移量
# mm/bmm	矩阵乘法，batch的矩阵乘法
# addmm/addbmm/addmv/addr/baddbmm..	矩阵运算
# t	转置
# dot/cross	内积/外积
# inverse	求逆矩阵
# svd	奇异值分解
