# 模型输出可以是一个像图像类别这样的离散值。
# softmax回归的输出单元从一个变成了多个 softmax回归的输出值个数等于标签里的类别数。
# 公式：  y_hat=exp(o1)/sigma(exp(oi))


# 衡量两个概率分布差异的测量函数:交叉熵（定量的表示预测好坏的损失函数）


# 总结
# softmax回归适用于分类问题。它使用softmax运算输出类别的概率分布。
# softmax回归是一个单层神经网络，输出个数等于分类问题中的类别个数。
# 交叉熵适合衡量两个概率分布的差异。


# 3.5
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# 下载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

# 验证
# feature的shape是 (C x H x W)
feature, labels = mnist_train[0]
print(feature.shape, labels)

# 输出10图像
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

# 读取小批量数据
batch_size=256
num_workers=4
train_iter=torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter=torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
