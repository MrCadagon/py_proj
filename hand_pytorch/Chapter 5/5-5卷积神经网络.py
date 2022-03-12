# 卷积层的作用：
# 一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；
# 另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

import time
import torch
from torch import nn, optim

import sys
sys.path.append(".")
sys.path.append("../Chapter 3/")
import d2lzh_pytorch_5 as d2l
import d2lzh_pytorch as d2l_3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实现LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet()
print(net)


batch_size = 256
train_iter, test_iter = d2l_3.load_data_fashion_mnist(batch_size=batch_size)


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

