'''
Function:
    实现全连接层
Author:
    Zhenchao Jin
微信公众号:
    Charles的皮卡丘
'''
import math
import numpy as np
# from .module import Module
import module

'''全连接层'''


class Linear(module.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(
            in_features=in_features, out_features=out_features, bias=True
        )
        # 初始化权重
        thresh = 1 / math.sqrt(in_features)
        self.weight = np.random.uniform(
            -thresh, thresh, (in_features, out_features)
        )
        self.bias = np.zeros((1, out_features))
        # 初始化storage
        self.storage.update({
            'direction': {
                'weight': np.zeros(np.shape(self.weight)),
                'bias': np.zeros(np.shape(self.bias))
            }
        })

    '''定义前向传播'''

    def forward(self, x):
        feats = x.dot(self.weight) + self.bias
        return feats

    '''定义反向传播'''
    # accumulated_gradient 为累计梯度
    def backward(self, accumulated_gradient):
        weight = self.weight
        if self.training:
            # 计算梯度
            grad_w = self.storage['x'].T.dot(accumulated_gradient)
            grad_b = np.sum(accumulated_gradient, axis=0, keepdims=True)

            # 根据梯度更新weight
            # ***获得sgd的梯度direction和结果params
            results = self.update(self.weight, grad_w, self.storage['direction']['weight'])
            # ***利用sgd的结果 更新自身的direction和结果params
            self.weight, self.storage['direction']['weight'] = results['params'], results['direction']

            # 根据梯度更新bias
            # ***获得sgd的梯度direction和结果params
            results = self.update(self.bias, grad_b, self.storage['direction']['bias'])
            # ***利用sgd的结果 更新自身的direction和结果params
            self.bias, self.storage['direction']['bias'] = results['params'], results['direction']
        return accumulated_gradient.dot(weight.T)

    '''返回参数数量'''

    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)