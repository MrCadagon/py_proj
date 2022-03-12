# 如果没有安装pandas，则反注释下面一行
# !pip install pandas

# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.shape)

train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# ##预处理数据
# aaaa = all_features.dtypes != 'object'  # 返回index
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# numeric_featres2 = all_features['MoSold']
# print(numeric_featres2)
# print(numeric_featres.index)
# print(aaaa)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转成指示特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)  # 288 331
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32)


# #训练模型
def get_net(features_num):
    net = nn.Linear(features_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, 0, 0.01)
    return net


# 对数误差函数
loss = torch.nn.MSELoss()


def log_rmse(net, features, labels):
    with torch.no_grad():
        labels_hat = torch.max(torch.tensor(1.0), net(features))
        log_rmse = torch.sqrt(loss(labels_hat.log(), labels.log()))
    return log_rmse


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
