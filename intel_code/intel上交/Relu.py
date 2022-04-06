import numpy as np

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx