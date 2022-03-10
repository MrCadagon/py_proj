# 实现线性回归-mini.batch-自动求导
# 1.数据集 2.model 3.loss oprimizer 4 train
# numpy ：广播（高维+低维=）高维+高维
import torch

# 1
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 2
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # in_feature out_feature 返回对象
        self.linear = torch.nn.Linear(1, 1)

    # __call__默认调用
    def forward(self, x):
        #  __call__默认调用
        y_pred = self.linear(x)
        return y_pred


# 实例化 可call的
model = LinearModel()

# 3
# 除N,降为
criterion = torch.nn.MSELoss(size_average=False)
# 对哪些weights进行优化  不构建计算图
# model.parameters()：检查module的所有成员
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred',y_test.data)