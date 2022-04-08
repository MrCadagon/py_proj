import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

x = torch.tensor([0, 2])
d2l.one_hot(x, vocab_size)
X = torch.arange(10).view(2, 5)
inputs = d2l.to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)

# 2初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


state = d2l.init_rnn_state(X.shape[0], num_hiddens, device)
inputs = d2l.to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = d2l.rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

output = d2l.predict_rnn('分开', 10, d2l.rnn, params, d2l.init_rnn_state, num_hiddens, vocab_size,
                         device, idx_to_char, char_to_idx)
print(output)

# 困惑度是对交叉熵损失函数做指数运算后得到的值
# 任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小vocab_size


# 训练模型
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
d2l.train_and_predict_rnn(d2l.rnn, get_params, d2l.init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
d2l.train_and_predict_rnn(d2l.rnn, get_params, d2l.init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
