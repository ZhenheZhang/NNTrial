#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   nn_seq.py
@Author :   Zhenhe Zhang
@Date   :   2023/3/18
@Notes  :
"""

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
import torchvision
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


class NNMaxPool(nn.Module):
    def __init__(self):
        super(NNMaxPool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


if __name__ == "__main__":
    # # test NNSeq with simulated input
    # nn_seq = NNSeq()
    # print(nn_seq)
    # input = torch.ones((64, 3, 32, 32))
    # output = nn_seq(input)
    # print(output.shape)
    # writer = SummaryWriter("../logs/nn_seq")
    # writer.add_graph(nn_seq, input)
    # writer.close()

    # # test NNMaxPool with cifar10
    nn_maxpool = NNMaxPool()
    print(nn_maxpool)
    cifar10 = Dataset()
    dataloader = DataLoader(cifar10.data, batch_size=64)
    writer = SummaryWriter("../logs/nn_maxpool")
    step = 0
    for data in dataloader:
        imgs, targets = data
        writer.add_images("input", imgs, step)
        output = nn_maxpool(imgs)
        writer.add_images("output", output, step)
        step = step + 1
    writer.close()


""" Visualization in Tensorboard

tensorboard --logdir=logs/nn_seq
tensorboard --logdir=logs/nn_maxpool

"""