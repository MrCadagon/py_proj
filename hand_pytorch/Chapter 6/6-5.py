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

# 1定义模型
num_hiddens = 256
# rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)

model = d2l.RNNModel(rnn_layer, vocab_size).to(device)
output = d2l.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
print(output)