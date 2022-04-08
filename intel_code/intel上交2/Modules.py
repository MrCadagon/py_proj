import torch
import torch.nn.functional as F
import numpy as np


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


class FusedConv_BN_Relu_2D(torch.autograd.Function):
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

