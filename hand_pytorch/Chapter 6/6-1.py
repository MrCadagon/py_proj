# p(w1,w2,w3)=p(w1)*p(w2|w1)*p(w3|w2,w1)

# 这里的马尔可夫假设是指一个词的出现只与前面n个词相关
# 引出1，2，3元语法
# 一元语法：P(1,2,3)=P(1)*P(2)*P(3)
# 二元语法：P(1,2,3)=P(1)*P(2|1)*P(3|2)

# 循环神经：非刚性地记忆所有固定长度的序列，而是通过隐藏状态来存储之前时间步的信息。

# 时间步tt的隐藏变量的计算由当前时间步的输入和上一时间步的隐藏变量共同决定：
# Ht=fai(Xt*Wxh+Ht-1*Whh+bn)    计算是循环的。使用循环计算的网络即循环神经网络
# Ht-1 上时刻隐藏变量
#  循环神经网络模型参数的数量不随时间步的增加而增长。

# 模拟循环神经网络
import torch

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
print(torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0)))