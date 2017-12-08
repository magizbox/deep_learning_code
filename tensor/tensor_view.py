import torch
import numpy as np

t = torch.from_numpy(
    np.random.random_integers(10, size=(2, 4)))
t
t.view(1, 8)
t.view(4, 2)