import numpy as np
import torch

a = torch.rand([1, 2, 3, 4])
b = torch.rand([1, 2, 3, 4])
print(a)
outer = torch.einsum("...bac,...dae->...bdce", a, b)
print(outer.shape)