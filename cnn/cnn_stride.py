import torch
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.Tensor(
    [
        [
            [[1, 2, 3, 4, 5]]
        ]
    ]
))
input
kernel = Variable(torch.Tensor(
    [
        [
            [[1, 1, 1]],
        ]
    ]
))
kernel

output = F.conv2d(input, kernel, stride=1)
print(output)

output = F.conv2d(input, kernel, stride=2)
print(output)
