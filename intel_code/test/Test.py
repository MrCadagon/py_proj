import torch
from Modules import Conv2D, BatchNorm, Relu, FusedConv_BN_Relu_2D
from Flops_compute import compute_Conv2d_flops, compute_BatchNorm2d_flops, compute_ReLU_flops, Net_by_torch
from torchstat_new import statistics


def conv_bn_relu_verify():
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
    print('conv+bn+relu类[FusedConv_BN_Relu_2D]梯度验证结果：', test)



def Flops_verify():
    # 分别计算conv bn relu的FLOPs值
    input = torch.rand(1, 10, 224, 224)
    output = torch.rand(1, 20, 222, 222)
    Flops_conv = compute_Conv2d_flops(3, input, output)
    print('卷积部分的Flops为：', Flops_conv)

    input = torch.rand(1, 20, 222, 222)
    output = torch.rand(1, 20, 222, 222)
    Flops_bn = compute_BatchNorm2d_flops(input, output)
    print('BN部分的Flops为：', Flops_bn)

    input = torch.rand(1, 20, 222, 222)
    output = torch.rand(1, 20, 222, 222)
    Flops_Relu = compute_ReLU_flops(input, output)
    print('Relu部分的Flops为：', Flops_Relu)

    # 利用nn.Module模块中的conv bn relu计算FLOPs
    model = Net_by_torch(10)
    node_list = statistics.stat(model, (10, 224, 224))

    # 输出实际值和理论值的比值
    print('卷积部分的Flops实际值与理论值的比值为',Flops_conv / node_list[0])
    print('BN部分的Flops实际值与理论值的比值为',Flops_bn / node_list[1])
    print('Relu部分的Flops实际值与理论值的比值为',Flops_Relu / node_list[2])
    print('卷积+BN+Relu部分的Flops实际值与理论值的比值为',(Flops_conv + Flops_bn + Flops_Relu) / (node_list[0] + node_list[1] + node_list[2]))


if __name__ == "__main__":
    conv_bn_relu_verify()
    Flops_verify()
