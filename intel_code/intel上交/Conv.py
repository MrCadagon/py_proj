import numpy as np


def conv_forward(x, w, b, conv_padding, conv_stride):
    pad = conv_padding
    stride = conv_stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    n, c, height, width = x.shape
    f, c, hh, ww = w.shape
    h_out = int(1 + (height + 2 * pad - hh) / stride)
    w_out = int(1 + (width + 2 * pad - ww) / stride)
    out = np.zeros((n, f, h_out, w_out))

    for j in range(h_out):
        for k in range(w_out):
            x_pad_t = x_pad[:, :, j * stride:j * stride + hh, k * stride:k * stride + ww]
            for i in range(f):
                out[:, i, j, k] = np.sum(x_pad_t * w[i], axis=(1, 2, 3))
    out += b[None, :, None, None]

    cache = (x, w, b, conv_padding, conv_stride)
    return out, cache


def conv_backward(dout, cache):
    x, w, b, pad, stride= cache
    _, _, hh, ww = w.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    db = dout.sum((0, 2, 3))
    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    n, f, h_out, w_out = dout.shape

    for i in range(h_out):
        for j in range(w_out):
            x_pad_t = x_pad[:, :, i * stride:i * stride + hh, j * stride:j * stride + ww]
            for k in range(n):
                dx[k, :, i * stride:i * stride + hh, j * stride:j * stride + ww] += np.sum(
                    dout[k, :, i, j][:, None, None, None] * w, axis=0)
            for o in range(f):
                dw[o] += np.sum(dout[:, o, i, j][:, None, None, None] * x_pad_t, axis=0)

    dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db
