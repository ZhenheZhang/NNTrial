#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   nn_loss.py
@Author :   Zhenhe Zhang
@Date   :   2023/3/18
@Notes  :   L1loss, MSELoss, CrossEntropyLoss
"""

import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')                  # Default: reduction='mean'
result = loss(inputs, targets)

loss_mse = MSELoss(reduction='sum')             # Default: reduction='mean'
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss(reduction='sum')  # Default: reduction='mean'
result_cross = loss_cross(x, y)
print(result_cross)
