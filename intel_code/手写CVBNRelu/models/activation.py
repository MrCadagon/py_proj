'''
Function:
    定义激活函数
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import numpy as np
import module


'''ReLU'''
class ReLU(module.Module):
    def __init__(self):
        super(ReLU, self).__init__()
    '''定义前向传播'''
    def forward(self, x):
        x = np.where(x >= 0, x, 0)
        return x
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        gradient = np.where(self.storage['x'] >= 0, 1, 0)
        return accumulated_gradient * gradient

'''Softmax'''
class Softmax(module.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__(dim=dim)
    '''定义前向传播'''
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)
    '''定义反向传播'''
    def backward(self, accumulated_gradient):
        p = self.forward(self.storage['x'])
        gradient = p * (1 - p)
        return accumulated_gradient * gradient