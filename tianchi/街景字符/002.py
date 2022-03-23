# *******************************************************************************************************读取数据task1
import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


# Dataset：对数据集的封装，提供索引方式的对数据样本进行读取
# DataLoder：对Dataset进行封装，提供批量读取的迭代读取

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)


train_path = glob.glob('./mchar_train/*.png')
train_path.sort()
train_json = json.load(open('./file_josn/111.json'))
train_label = [train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       # 缩放到固定尺寸
                       transforms.Resize((64, 128)),

                       # 随机颜色变换
                       transforms.ColorJitter(0.2, 0.2, 0.2),

                       # 加入随机旋转
                       transforms.RandomRotation(5),

                       # 将图片转换为pytorch 的tesntor
                       transforms.ToTensor(),

                       # 对图像像素进行归一化
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ]))
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=10,  # 每批样本个数
    shuffle=False,  # 是否打乱顺序
    num_workers=10,  # 读取的线程个数
)

for data in train_loader:
    data1 = data[0]
    data2 = data[1]
    print('aa')
    break

# *******************************************************************************************************task3训练数据
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(32 * 3 * 7, 11)
        self.fc2 = nn.Linear(32 * 3 * 7, 11)
        self.fc3 = nn.Linear(32 * 3 * 7, 11)
        self.fc4 = nn.Linear(32 * 3 * 7, 11)
        self.fc5 = nn.Linear(32 * 3 * 7, 11)
        self.fc6 = nn.Linear(32 * 3 * 7, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6


model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.005)
loss_plot, c0_plot = [], []

# 迭代10个Epoch
epoch1 = 3

for epoch in range(epoch1):
    for data in train_loader:
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
               criterion(c1, data[1][:, 1]) + \
               criterion(c2, data[1][:, 2]) + \
               criterion(c3, data[1][:, 3]) + \
               criterion(c4, data[1][:, 4]) + \
               criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_plot.append(loss.item())
        c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item() * 1.0 / c0.shape[0])
    print(epoch)

x = [i for i in range(epoch1 * 3000)]
plt.figure(figsize=(10, 8))
plt.plot(x, loss_plot, color='blue', linewidth=1.0, linestyle='--')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(x, c0_plot, color='blue', linewidth=1.0, linestyle='--')
plt.show()
