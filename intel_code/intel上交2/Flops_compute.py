import torch.nn as nn
import numpy as np

def compute_Conv2d_flops(kernal_size, inp, out):
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = kernal_size, kernal_size
    out_c, out_h, out_w = out.size()[1:]
    groups = 1

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    total_flops = total_conv_flops
    return total_flops

def compute_BatchNorm2d_flops(inp, out):
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_flops = np.prod(inp.shape)
    # affine 需要乘2
    batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(inp, out):
    active_elements_count = inp.size()[0]
    for s in inp.size()[1:]:
        active_elements_count *= s
    return active_elements_count
