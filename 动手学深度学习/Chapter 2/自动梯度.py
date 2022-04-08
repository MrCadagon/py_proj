import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# 梯度
out.backward()  # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

# grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，！！
# 梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

# 针对问题1：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter02_prerequisite/2.3_autograd
# 我们不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量。
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print('z', z)

v = torch.tensor([[1, 1], [1, 1]], dtype=torch.float)
z.backward(v, retain_graph=True)
print(x.grad)

x.grad.data.zero_()
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
# 相当于zv相乘变为标量后再反向传播
z.backward(v)
print(x.grad)

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)  # True
print(y2, y2.requires_grad)  # False
print(y3, y3.requires_grad)  # True

# y2的梯度没有被回传 只有y1的2进行了梯度回传
y3.backward()
print(x.grad)

print('不会记录在计算图，所以不会影响梯度传播')
x = torch.ones(1, requires_grad=True)

print(x.data)  # 还是一个tensor
print(x.data.requires_grad)  # 但是已经是独立于计算图之外

y = 2 * x
# #######################################################   data()    #########################################################
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x)  # 更改data的值也会影响tensor的值
print(x.grad)
