import torch
from Modules import Conv2D, BatchNorm, Relu, FusedConv_BN_Relu_2D
from Flops_compute import compute_Conv2d_flops, compute_BatchNorm2d_flops,compute_ReLU_flops

def mian():
    # 卷积验证
    X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
    weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
    test = torch.autograd.gradcheck(Conv2D.apply, (X, weight))
    print('卷积类[Conv2D]梯度验证结果：', test)

    # BN
    x = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
    test = torch.autograd.gradcheck(BatchNorm.apply, (x,), fast_mode=False)
    print('BN类[BatchNorm]梯度验证结果：', test)

    # relu验证
    x = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
    test = torch.autograd.gradcheck(Relu.apply, (x,), fast_mode=False)
    print('Relu类[Relu]梯度验证结果：', test)

    # conv+bn+relu验证
    X = torch.rand(2, 3, 4, 4, requires_grad=True, dtype=torch.double)
    weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
    test = torch.autograd.gradcheck(FusedConv_BN_Relu_2D.apply, (X, weight), eps=1e-3)
    print('conv+bn+relu类[FusedConv_BN_Relu_2D_Function]梯度验证结果：', test)

    # 分别计算conv bn relu的FLOPs值
    input = torch.rand(1, 10, 224, 224)
    output = torch.rand(1, 10, 222, 222)
    print('卷积部分的Flops为：',compute_Conv2d_flops(3, input, output))

    input = torch.rand(1, 10, 222, 222)
    output = torch.rand(1, 10, 222, 222)
    print('BN部分的Flops为：',compute_BatchNorm2d_flops(input, output))

    input = torch.rand(1, 10, 222, 222)
    output = torch.rand(1, 10, 222, 222)
    print('Relu部分的Flops为：',compute_ReLU_flops(input, output))

if __name__=="__main__":
    mian()
