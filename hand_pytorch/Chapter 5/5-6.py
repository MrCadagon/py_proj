# 学习到的特征可以超越手工设计的特征
# 有5层卷积和2层全连接隐藏层，以及1个全连接输出层
# AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数
# AlexNet通过丢弃法（参见3.13节）来控制全连接层的模型复杂度。
# AlexNet引入了大量的图像增广，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。

import time
import torch
from torch import nn, optim
import torchvision

import sys
sys.path.append("..")
import d2lzh_pytorch_5 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

