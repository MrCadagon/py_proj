from Flops_compute import compute_Conv2d_flops, compute_BatchNorm2d_flops,compute_ReLU_flops
import torch

input = torch.rand(1, 10, 224, 224)
output = torch.rand(1, 10, 222, 222)
print(compute_Conv2d_flops(3, input, output))

input = torch.rand(1, 10, 222, 222)
output = torch.rand(1, 10, 222, 222)
print(compute_BatchNorm2d_flops(input, output))

input = torch.rand(1, 10, 222, 222)
output = torch.rand(1, 10, 222, 222)
print(compute_ReLU_flops(input, output))
