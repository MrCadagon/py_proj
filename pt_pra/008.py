import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 基础模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 一个样本 性能好 时间长
# batch：时间短 性能差
# 小批量：shuffle


# xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = torch.from_numpy(xy[:, :-1])
# y_data = torch.from_numpy(xy[:, [-1]])

# 表明继承自Dataset
class DiabetesDateset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        pass

    # item函数
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        pass

    # len函数
    def __len__(self):
        return self.len
        pass


# 针对小数据集
dataset = DiabetesDateset('diabetes.csv')
# num_workers 表示并行n线程
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

# 进行训练
# linux fork win:spawn WARNINGS
# if __name__=='_main_'

epoch_list = []
loss_list = []
for epoch in range(100):
    # data-> x y
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        epoch_list.append(epoch)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
