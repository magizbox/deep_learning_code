""" Implement ideas in good answer for question
Difference between “equivariant to translation” and “invariant to translation”
Link  : https://datascience.stackexchange.com/q/16060
Author: Vu Anh
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def conv(input):
    """ apply conv with simple filter to input
    :type input: 1D ndarray
    """
    x = torch.from_numpy(input)
    x = x.view(1, 1, len(x)).float()
    output = F.conv1d(
        Variable(x),
        Variable(torch.Tensor([[[1, 1]]]))
    )
    output = output.data.numpy()[0][0]
    return output


def shift(input):
    """ shift input
    :type input: 1D ndarray
    """
    output = np.roll(input, 1)
    return output

################################################
# Equivariant to translation
################################################

input = np.array([0, 3, 2, 0, 0])

conv(input)
shift(input)

shift(conv(input))
conv(shift(input))

conv(shift(input)) == shift(conv(input))