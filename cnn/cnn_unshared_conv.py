import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

input = Variable(torch.Tensor([[[
    [1, 2, 3, 4]
]]]))
filters = Variable(torch.Tensor([[[[1, 1]]]]))

output = F.conv2d(input, filters)
print(output)


def unshared_conv1d(input, kernels):
    r = range(len(kernels))
    partitions = [input.index_select(3, torch.LongTensor([i, i + 1])) for i in
                  r]
    output = [F.conv2d(partitions[i], kernels[i]) for i in r]
    output = [_.squeeze().data.numpy()[0] for _ in output]
    output = torch.from_numpy(np.array(output)).view(1, 1, 1, 3)
    return output


kernels = [
    Variable(torch.Tensor([[[[1, 1]]]])),
    Variable(torch.Tensor([[[[2, 2]]]])),
    Variable(torch.Tensor([[[[3, 3]]]])),
]

output = unshared_conv1d(input, kernels)
print(output)
