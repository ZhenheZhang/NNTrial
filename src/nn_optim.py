#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   nn_optim.py
@Author :   Zhenhe Zhang
@Date   :   2023/3/18
@Notes  :
"""

import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader


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
        print('epoch{}: loss={}'.format(epoch, running_loss))
