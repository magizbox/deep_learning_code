import torch
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.Tensor(
    [
        [
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        ]
    ]
))
input
kernel = Variable(torch.Tensor(
    [
        [
            [[1, 1, 1, 1, 1, 1]],
        ]
    ]
))
kernel

###########################################
# without padding                         #
###########################################
x = input
for i in range(3):
    x = F.conv2d(x, kernel)
    print(x)

###########################################
# default padding                         #
###########################################
x = input
for i in range(3):
    x = F.conv2d(x, kernel, padding=(0,3))
    size = x.size()[3]
    x = x.index_select(3, torch.LongTensor(range(0, size -1 )))
    print(x)

###########################################
# custom padding                          #
###########################################
x = input
for i in range(3):
    x = F.conv2d(x, kernel, padding=(0,3))
    size = x.size()[3]
    x = x.index_select(3, torch.LongTensor(range(0, size -1 )))
    print(x)

