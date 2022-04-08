import torch
import numpy as np
import sys

sys.path.append(".")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=1, keepdim=True))
print(X.sum(dim=0, keepdim=True))


# 定义softmax
# 每一行的输出进行平均化处理
def softmax(X):
    x_exp = X.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
print('定义损失函数')
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
uu = y_hat.gather(1, y.view(-1, 1))
print('y_hat ， y', y_hat, y, uu)
print(y_hat.argmax(dim=1))

print(d2l.accuracy(y_hat, y))

print(d2l.evaluate_accuracy(test_iter, net))

num_epochs, lr = 100, 0.1
d2l.train_ch3(net, train_iter, test_iter, d2l.cross_entropy, num_epochs, batch_size, [W, b], lr)

# predict
print('predict\n')
X, y = iter(test_iter).next()

# print(X, y)
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
