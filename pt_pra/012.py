# 循环神经网络
# 针对序列问题：全连接层的参数很多
# RNN_cell:x--wx+wh+b--tanh--ht
# RNN:依次算出上述隐层的过程
# ！！！！！！！！只用
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)  # (batch,input_size) (batch,hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('inpiut size', input.shape)

    hidden = cell(input, hidden)

    print('output size', hidden)
    print(hidden)

# =======================================================now train
batch_size = 1
seq_len = 3  # 横向长度
input_size = 4
hidden_size = 2  # 每个hi中变量的维度
num_layers = 1
# num_layers 向上step
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
# 四个参数的维度
# seqsize,batch,input_size
# numLayers,batch,hidden_size
# seqsize,batch,hidden_size
# numLayers,batch,hidden_size
out, hidden = cell(inputs, hidden)  # (h0-hn-1)(hn)
print(out.shape)
print(hidden.shape)

# =======================================================now hello --》 ohlol  转换为分类问题 交叉熵损失
# trans to one-hot
input_size = 4  # one-hot 为四维
hidden_size = 4  # one-hot 为四维
num_layers = 1  # 随意
batch_size = 1  #
seq_len = 5  # 横向有五个样本
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # input size
y_data = [3, 1, 2, 3, 2]  # hidden size
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


# 使用Rnncell  讲一个序列转换为另一个序列
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    # 构造h0
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)
# torch.set_default_tensor_type(torch.DoubleTensor)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')  # 进行预测
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))

# 简洁*************************************************************************************************88
print('简洁实现\n\n\n')
# trans to one-hot
input_size = 4  # one-hot 为四维
hidden_size = 4  # one-hot 为四维
num_layers = 1  # 随意
batch_size = 1  #
seq_len = 5  # 横向有五个样本
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # input size
y_data = [3, 1, 2, 3, 2]  # hidden size
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)  #
    # input_size = 4
    # hidden_size = 4
    # num_layers = 1
    # batch_size = 1
    # seq_len = 5


net = Model(input_size, hidden_size, batch_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))

# one-hot缺点：纬度高，稀疏，硬编码
# 解决：数据的降维 使用embedding（嵌入层）


