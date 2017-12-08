import torch
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7]]]))
F.max_pool1d(input, kernel_size=3, stride=2)