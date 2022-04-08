import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def conv2d_backward(grad_out, X, weight):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input


class Conv2D(torch.autograd.Function):
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)
        return F.conv2d(X, weight)

    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return conv2d_backward(grad_out, X, weight)


def unsqueeze_all(t):
    return t[None, :, None, None]


def bn_backward(grad_out, X, sum, sqrt_var, N, eps):
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps) ** 2

    d_var = d_denom / (2 * sqrt_var)

    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)

    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)

    grad_input += d_mean_dx
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)
    return grad_input


class BatchNorm(torch.autograd.Function):
    def forward(ctx, X, eps=1e-3):
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(X)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return bn_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)


def relu_backward(grad_out, X, is_X_turple):
    if (is_X_turple == 1):
        grad_input = np.where(X[0] > 0, grad_out, 0)
    else:
        grad_input = np.where(X > 0, grad_out, 0)
    return torch.tensor(grad_input)


class Relu(torch.autograd.Function):

    def forward(ctx, X, eps=1e-3):
        ctx.save_for_backward(X)
        return np.maximum(X, 0)

    def backward(ctx, grad_out):
        X = ctx.saved_tensors
        return relu_backward(grad_out, X, 1)


class FusedConv_BN_Relu_2D_Function(torch.autograd.Function):
    def forward(ctx, X, conv_weight, eps=1e-3):
        # N, C, H, W
        assert X.ndim == 4
        ctx.save_for_backward(X, conv_weight)
        # conv
        X = F.conv2d(X, conv_weight)
        # bn
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        # relu
        out = np.maximum(out, 0)
        return out

    def backward(ctx, grad_out):
        X, conv_weight, = ctx.saved_tensors
        # conv
        X_conv_out = F.conv2d(X, conv_weight)
        # bn
        eps = 1e-3
        sum = X_conv_out.sum(dim=(0, 2, 3))
        var = X_conv_out.var(unbiased=True, dim=(0, 2, 3))
        N = X_conv_out.numel() / X_conv_out.size(1)
        sqrt_var = torch.sqrt(var)
        mean = sum / N
        denom = sqrt_var + eps
        bn_out = X_conv_out - unsqueeze_all(mean)
        bn_out /= unsqueeze_all(denom)
        # relu_back
        grad_out = relu_backward(grad_out, bn_out, 0)
        # bn_back
        grad_out = bn_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                               ctx.N, ctx.eps)
        # conv_back
        grad_X, grad_input = conv2d_backward(grad_out, X, conv_weight)

        return grad_X, grad_input, None, None, None, None, None


import math

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        return FusedConv_BN_Relu_2D_Function.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

# 卷积验证
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
test = torch.autograd.gradcheck(Conv2D.apply, (X, weight))
print(test)

# BN
a = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
x = torch.rand(5, 2, 3, 4, requires_grad=True, dtype=torch.double)
b = unsqueeze_all(x)
print(torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False))

# relu验证
a = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
test2 = torch.autograd.gradcheck(Relu.apply, (a,), fast_mode=False)
print(test2)

# conv+bn+relu验证
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(2, 3, 4, 4, requires_grad=True, dtype=torch.double)
print(torch.autograd.gradcheck(FusedConv_BN_Relu_2D_Function.apply, (X, weight), eps=1e-3))

# # FLOPS测试
from torchstat import stat







# FLOPS
class LeNet(nn.Module):
    def __init__(self, in_channels):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1)  # 20x24x24
        self.bn1 = nn.BatchNorm2d(20)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        return out



class Net1(nn.Module):
    def __init__(self,in_channels):
        super(Net1, self).__init__()
        self.convbn1 = FusedConvBN(in_channels, 20, 3)

    def forward(self, input):
        out = self.convbn1(input)
        return out


model = LeNet(10)
model2 = Net1(10)

stat(model, (10, 224, 224))
stat(model2, (10, 224, 224))

# http://t.zoukankan.com/xuanyuyt-p-12653041.html

# # FLOPS测试
# import torch
# from torchvision.models import resnet18
# from thop import profile
# model = resnet18()
# # model = FusedConv_BN_Relu_2D_Function()
# input = torch.randn(10, 10, 128, 128)
# flops, params = profile(model2, inputs=(input, ))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))



inp = torch.rand(1,2,3,4)
active_elements_count=1
for s in inp.size()[1:]:
    print(s)
    active_elements_count *= s