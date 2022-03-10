# 1.关于损失函数的选择
# 线性回归用到的平方损失函数和softmax回归用到的交叉熵损失函数
# 2.验证集
# 数据来进行模型选择
# 欠拟合和过拟合：1.模型复杂度

# %matplotlib inline
import torch
import numpy as np
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat(features, torch.pow(features, 2), torch.pow(features, 2))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
