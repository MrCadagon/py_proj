import numpy as np
from normal import rel_error, eval_numerical_gradient_array
from Conv import conv_forward, conv_backward
from BN import batchnorm_forward, batchnorm_backward
from Relu import relu_forward, relu_backward


def test_Conv():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv_stride = 2
    conv_pad = 1
    out, _ = conv_forward(x, w, b, conv_pad, conv_stride)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097],
                              [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974],
                              [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383],
                              [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306],
                              [2.38090835, 2.38247847]]]])

    print('测试卷积层的正向传播：')
    print('difference: ', rel_error(out, correct_out))

    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2, )
    dout = np.random.randn(4, 2, 5, 5)
    conv_stride = 1
    conv_pad = 1

    dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_pad, conv_stride)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_pad, conv_stride)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_pad, conv_stride)[0], b, dout)

    out, cache = conv_forward(x, w, b, conv_pad, conv_stride)
    dx, dw, db = conv_backward(dout, cache)

    print('测试卷积层的反向传播：')
    print('dx error: ', rel_error(dx, dx_num))
    print('dw error: ', rel_error(dw, dw_num))
    print('db error: ', rel_error(db, db_num))


def test_BN():
    np.random.seed(231)
    N, C, H, W = 10, 4, 11, 12
    bn_param = {'mode': 'train'}
    gamma = np.ones(C)
    beta = np.zeros(C)
    for t in range(50):
        x = 2.3 * np.random.randn(N, C, H, W) + 13
        batchnorm_forward(x, gamma, beta, bn_param)
    bn_param['mode'] = 'test'
    x = 2.3 * np.random.randn(N, C, H, W) + 13
    a_norm, _ = batchnorm_forward(x, gamma, beta, bn_param)

    print('测试BN层的正向传播：')
    print('  means: ', a_norm.mean(axis=(0, 2, 3)))
    print('  stds: ', a_norm.std(axis=(0, 2, 3)))

    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(C)
    beta = np.random.randn(C)
    dout = np.random.randn(N, C, H, W)

    bn_param = {'mode': 'train'}
    fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
    fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
    fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma, dout)
    db_num = eval_numerical_gradient_array(fb, beta, dout)

    _, cache = batchnorm_forward(x, gamma, beta, bn_param)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    print('测试BN层的反向传播：')
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))


def test_Relu():
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = relu_forward(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    print('测试Relu函数的正向传播：')
    print('difference: ', rel_error(out, correct_out))

    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    print('测试Relu函数的反向传播：')
    print('dx error: ', rel_error(dx_num, dx))
