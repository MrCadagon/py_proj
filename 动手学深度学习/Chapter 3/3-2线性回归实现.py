import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_examples = 1000
num_inputs = 2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
true_w = [2, -3, 4]
true_b = 4.2
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
print(features[0], labels[0])


j = torch.LongTensor(indices[1: min(1 + 20, 100)])