import torch
from torch import nn
import numpy as np



rnn = nn.RNN(10, 20, 1)
input = torch.randn(5, 3, 10)
h0 = torch.randn(1, 3, 20)
output, hn = rnn(input, h0)
print(output)
print(hn)