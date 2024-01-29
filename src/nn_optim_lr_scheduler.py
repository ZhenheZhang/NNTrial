#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   nn_optim_lr_scheduler.py
@Author :   Zhenhe Zhang
@Date   :   2023/3/20
@Notes  :
"""

import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class Dataset:
    def __init__(self):
        super(Dataset, self).__init__()
        self.data = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                                 transform=torchvision.transforms.ToTensor())

class NNSeq(nn.Module):
    def __init__(self):
        super(NNSeq, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == "__main__":
    nn_seq = NNSeq()
    print(nn_seq)
    cifar10 = Dataset()
    dataloader = DataLoader(cifar10.data, batch_size=64)

    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(nn_seq.parameters(), lr=0.01)
    # Assuming optimizer uses lr = 0.01 for all groups
    # lr = 0.01     if epoch < 30
    # lr = 0.001    if 30 <= epoch < 60
    # lr = 0.0001   if 60 <= epoch < 90
    # ...
    scheduler = StepLR(optim, step_size=3, gamma=0.1)

    print("="*60)
    print("===Start...")
    print(f"===Run Training in {__file__} ...")

    for epoch in range(10):
        running_loss = 0.0
        for data in dataloader:
            imgs, targets = data
            outputs = nn_seq(imgs)
            result_loss = loss(outputs, targets)
            optim.zero_grad()
            result_loss.backward()
            optim.step()
            running_loss = running_loss + result_loss
        scheduler.step()
        # print('epoch{}, lr={}: loss={}'.format(epoch, optim.state_dict()['param_groups'][0]['lr'], running_loss))
        print('epoch{}, lr={}: loss={}'.format(epoch, scheduler.get_last_lr(), running_loss))
