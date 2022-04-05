import math
import numpy as np
import data, image
import module


class BN_2D(module.Module):
    def __init__(self, out_channel):
        super(BN_2D, self).__init__(
            out_channel=out_channel
        )
        self.gamma
    def forward(self,x):
        if self.training:
            sample_mean = np.mean(x, axis=0)  # np.mean([[1,2],[3,4]])->[2,3]
            sample_var = np.var(x, axis=0)
            out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            out = gamma * out_ + beta
            cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)
        else:


    def batchnorm_forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance
          - running_mean: Array of shape(D,) giving running mean of features
          - running_var Array of shape(D,) giving running variance of features
        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        out, cache = None, None

        if mode == 'train':
            sample_mean = np.mean(x, axis=0)  # np.mean([[1,2],[3,4]])->[2,3]
            sample_var = np.var(x, axis=0)
            out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            out = gamma * out_ + beta
            cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)
        elif mode == 'test':
            # scale = gamma / np.sqrt(running_var + eps)
            # out = x * scale + (beta - running_mean * scale)
            x_hat = (x - running_mean) / (np.sqrt(running_var + eps))
            out = gamma * x_hat + beta
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out, cache

    def batchnorm_backward(dout, cache):

        # Backward pass for batch normalization.Inputs:- dout: Upstream derivatives, of shape (N, D)-
        # cache: Variable of intermediates from batchnorm_forward.Returns a tuple of:- dx: Gradient with respect to inputs x, of shape (N, D)- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)- dbeta: Gradient with respect to shift parameter beta, of shape (D,)"""

        dx, dgamma, dbeta = None, None, None

        out_, x, sample_var, sample_mean, eps, gamma, beta = cache
        N = x.shape[0]
        dout_ = gamma * dout
        dvar = np.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, axis=0)
        dx_ = 1 / np.sqrt(sample_var + eps)
        dvar_ = 2 * (x - sample_mean) / N
        # intermediate for convenient calculation
        di = dout_ * dx_ + dvar * dvar_
        dmean = -1 * np.sum(di, axis=0)
        dmean_ = np.ones_like(x) / N
        dx = di + dmean * dmean_
        dgamma = np.sum(dout * out_, axis=0)
        dbeta = np.sum(dout, axis=0)
        return dx, dgamma, dbeta