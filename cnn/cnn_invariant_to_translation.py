""" Implement ideas in good answer for question
Difference between “equivariant to translation” and “invariant to translation”
Link  : https://datascience.stackexchange.com/q/16060
Author: Vu Anh
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def pooling(input):
    """ apply conv with simple filter to input
    :type input: 1D ndarray
    """
    x = torch.from_numpy(input)
    x = x.view(1, 1, len(x)).float()
    output = F.max_pool1d(
        Variable(x),
        kernel_size=3,
        stride=1
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
# Invariant to translation
################################################

input = np.array([3, 0, 0, 4, 0])
input
shift(input)

pooling(input)
pooling(shift(input))
pooling(input) == pooling(shift(input))
