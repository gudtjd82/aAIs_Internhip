import numpy as np
import torch


# 예제 텐서 생성
kv = torch.randn(2, 3, 4, 5, 10)  # shape: (4, 5, 10)

kv = torch.sum(kv, dim=-1)
print(kv.shape)