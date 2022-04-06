import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    N_1, C_1, H_1, W_1 = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C_1)

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + eps)
        xn = xc / std
        out = gamma * xn + beta

        cache = (mode, x, gamma, xc, std, xn, out)

        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        running_var *= momentum
        running_var += (1 - momentum) * var
    elif mode == 'test':
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
        cache = (mode, x, xn, gamma, beta, std)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var


    out = out.reshape(N_1, H_1, W_1, C_1).transpose(0, 3, 1, 2)
    return out, cache

def batchnorm_backward(dout, cache):
    N_1, C_1, H_1, W_1 = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, C_1)

    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    else:
        raise ValueError(mode)

    dx = dx.reshape(N_1, H_1, W_1, C_1).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta