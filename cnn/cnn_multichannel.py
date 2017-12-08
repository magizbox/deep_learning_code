import torch
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.Tensor(
    [
        [
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
        ]
    ]
))
input
kernel = Variable(torch.Tensor(
    [
        [
            [[1, 1], [0, 1]],
            [[1, 0], [-1, 0]]
        ]
    ]
))
kernel

output = F.conv2d(input, kernel)
output
