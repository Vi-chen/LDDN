from typing import List, Optional
from torch import Tensor, reshape, stack
import torch
import numpy as np
from torch import nn
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Module,
    PReLU,
    Sequential,
    Sigmoid,
    init
)


class Up(Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        self.convolution1x1 = Sequential(
            Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1),
            BatchNorm2d(in_channel),
            PReLU(),
        )
        self.convolution = Sequential(
            Conv2d(out_channel, out_channel, 3, 1, padding=1),
            BatchNorm2d(out_channel),
            PReLU(),
            Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            BatchNorm2d(out_channel),
            PReLU(),
        )

    def forward(self, x, y: Optional[Tensor] = None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
            x = self.convolution1x1(x)
            x = self.up(x)
            x = self.convolution(x)
        else:
            x = self.up(x)
            x = self.convolution(x)
        return x


class Classifier(Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.linear = Sequential(
            Conv2d(in_channel, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Concat(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in*2, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class Add(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = x1 + x2
        x = self.conv(x)
        return x


class Sub(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = x1 - x2
        x = self.conv(x)
        return x


