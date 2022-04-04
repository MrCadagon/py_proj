import numpy as np
import matplotlib.pyplot as plt

#1111111111111111111111111111111111111111111111111111111111111测试的数据
# 训练集特征2维
X = np.array([[2, 1], [-1, 1], [-1, -1], [1, -1]])
# 类别标签
y = np.array([1, 2, 3, 4])

# 初始化权重矩阵
np.random.seed(42)
w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 4)
print(w1)

# 学习率(先不加正则化处理)
γ = 0.01

#1111111111111111111111111111111111111111111111111111111111111神经网络
# 损失函数使用均方误差 E=1/2*(y-yhat)**2(单个样本)
# 对output进入端梯度E(w2*net_output)求导(使用标准BP算法)
# 对w2求梯度
def grad2_func(y, yhat, net_output):
    res = []
    for j in range(len(y)):
        res.append((y[j] - yhat[j]) * (-net_output))
    res = np.array(res).T
    return res


# 对w1求梯度
def grad1_func(y, yhat, sample, net_output):
    grad1 = (y - yhat).sum()  # 定值
    grad3 = sample[0], sample[1]
    res = []
    for i in range(w1.shape[0]):
        r1 = []
        for j in range(w1.shape[1]):
            grad2 = -(net_output[j] * (1 - net_output[j]) * w2[j, 0] +
                      net_output[j] * (1 - net_output[j]) * w2[j, 1] +
                      net_output[j] * (1 - net_output[j]) * w2[j, 2] +
                      net_output[j] * (1 - net_output[j]) * w2[j, 3])
            r1.append(grad1 * grad2 * grad3[i])
        res.append(r1)
    return np.array(res)


# 正向传播 (激活函数使用sigmoid,方便求导)
def sigmoid(net):
    res = []
    for x in net:
        res.append(1 / (1 + np.exp(-x)))

    return np.array(res)


# 向前传播
def forward(X):
    net_input = []
    net_output = []
    output = []
    for i in range(len(X)):
        net_input.append(np.dot(X[i], w1))
    for i in net_input:
        net_output.append(sigmoid(i))
    for i in net_output:
        output.append(np.dot(i, w2))
    return output, net_output


# 反向传播
def back_propagation(w1, w2, epoch=1000):
    error = []
    for epoch in range(epoch):
        # 计算向前传播，输出结果及中间变量(求梯度时,用的到)
        yhat, net_output = forward(X)
        loss = 0
        # 算出所有样本的均方误差
        for i in yhat:
            for j in range(len(i)):
                loss += 1 / 2 * (y[j] - i[j]) ** 2
        # print(f'loss:{loss}')
        error.append(loss)

        for index, i in enumerate(yhat):
            # 计算所有权重的梯度
            # 真实标签编码,独热编码，类别位置1，其他为0
            encode_y = np.array([0, 0, 0, 0])
            encode_y[index] = 1
            # print(encode_y)
            grad2 = grad2_func(encode_y, i, net_output[index])
            grad1 = grad1_func(encode_y, i, X[index], net_output[index])
            # 梯度更新，未加正则化
            w1 -= γ * grad1
            w2 -= γ * grad2

    return error, w1, w2


# 预测函数
def predict(test_x):
    output, _ = forward(test_x)
    # 使用numpy 进行softmax处理输出
    pred_y = []
    for i in output:
        sum_exp = 0
        res = []
        for j in i:
            sum_exp += np.exp(j)
        for j in i:
            res.append(np.exp(j) / sum_exp)
        pred_y.append(res.index(max(res)) + 1)
    return pred_y



#1111111111111111111111111111111111111111111111111111111111111主函数
if __name__ == '__main__':
    epoch = 1000
    loss, w1, w2 = back_propagation(w1, w2, epoch)
    plt.figure(figsize=(12, 5))
    plt.plot(range(epoch), loss, label='累计误差')
    plt.xlabel('迭代次数')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()

    test = np.array([[1, 1], [-2, 1], [-1, -2], [1, -1]])
    print(predict(test))

