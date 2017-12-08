import torch
import torch.nn.functional as F
from torch.autograd import Variable

inputs = Variable(torch.Tensor([
    [
        [
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]]))
filters = Variable(torch.Tensor([[[[0, 1], [4, 2]]]]))

output = F.conv2d(inputs, filters)
print(output)
