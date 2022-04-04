import math

import random
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets

import os

current_dir = os.getcwd()
import sys

sys.path.append(current_dir + '/models')
import sequential, linear, activation, convolution, flatten
import sgd, criterion

cnn = sequential.Sequential()
cnn.addmodule('conv1', convolution.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
cnn.addmodule('relu1', activation.ReLU())
cnn.addmodule('conv2', convolution.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
cnn.addmodule('relu2', activation.ReLU())
cnn.addmodule('conv3', convolution.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
cnn.addmodule('flatten', flatten.Flatten())
cnn.addmodule('fc', linear.Linear(32 * 8 * 8, 10))
cnn.addmodule('softmax', activation.Softmax())
linear.Linear(64, 128)

# 定义损失函数
criterion = criterion.CrossEntropy()
# 定义优化器
optimzer = sgd.SGD(cnn, criterion=criterion, learning_rate=0.001, momentum=0.9)


# 导入数据集
# --手写数字数据集
data = datasets.load_digits()
inputs, targets = data.data.reshape(-1, 1, 8, 8), data.target

# --targets转成one-hot格式
num_classes = np.amax(targets) + 1
one_hot = np.zeros((targets.shape[0], num_classes))
one_hot[np.arange(targets.shape[0]), targets] = 1
targets = one_hot
# --随机打乱划分训练集和验证集
inds = list(range(inputs.shape[0]))
random.shuffle(inds)
trainset = inputs[inds[:int(inputs.shape[0] * 0.9)]], targets[inds[:int(inputs.shape[0] * 0.9)]]
valset = inputs[inds[int(inputs.shape[0] * 0.9):]], targets[inds[int(inputs.shape[0] * 0.9):]]
# 开始训练
losses_log, batch_size, losses_batch, vis_infos = [], 16, [], []
epochs = 20
for epoch in range(epochs):
    for idx in range(math.ceil(trainset[0].shape[0] / batch_size)):
        x_b = trainset[0][idx * batch_size: (idx + 1) * batch_size] / 16
        t_b = trainset[1][idx * batch_size: (idx + 1) * batch_size]
        output = cnn(x_b)
        loss = criterion(output, t_b)
        optimzer.step()
        losses_log.append(loss)
        losses_batch.append(loss)
        if len(losses_log) == 20:
            print(f'Epoch: {epoch+1}/10, Batch: {idx+1}/{math.ceil(trainset[0].shape[0] / 32)}, Loss: {sum(losses_log) / len(losses_log)}')
            losses_log = []
    # 每个epoch结束之后测试一下模型准确性
    predictions = cnn(valset[0]).argmax(1)
    acc = np.equal(predictions, valset[1].argmax(1)).sum() / predictions.shape[0]
    print(f'Accuracy of Epoch {epoch+1} is {acc}')
    vis_infos.append([acc, sum(losses_batch) / len(losses_batch)])
    losses_batch = []
# 画下loss曲线和acc曲线
plt.plot(range(epochs), [i[0] for i in vis_infos], marker='o', label='Accuracy')
plt.plot(range(epochs), [i[1] for i in vis_infos], marker='*', label='Loss')
plt.legend()
plt.show()