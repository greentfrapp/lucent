"""
Original work by ProGamerGov at
https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/helper_layers.py

The MIT License (MIT)

Copyright (c) 2020 ProGamerGov

Copyright (c) 2015 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditionLayer(nn.Module):
    def forward(self, input, input2):
        return input + input2


class MaxPool2dLayer(nn.Module):
    def forward(self, input, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False):
        return F.max_pool2d(input, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


class PadLayer(nn.Module):
    def forward(self, input, padding=(1, 1, 1, 1), value=None):
        if value == None:
            return F.pad(input, padding)
        else:
            return F.pad(input, padding, value=value)


class ReluLayer(nn.Module):
    def forward(self, input):
        return F.relu(input)


class SoftMaxLayer(nn.Module):
    def forward(self, input, dim=1):
        return F.softmax(input, dim=dim)


class DropoutLayer(nn.Module):
    def forward(self, input, p=0.4000000059604645, training=False, inplace=True):
        return F.dropout(input = input, p = p, training = training, inplace = inplace)


class CatLayer(nn.Module):
    def forward(self, input_list, dim=1):
        return torch.cat(input_list, dim)


class LocalResponseNormLayer(nn.Module):
    def forward(self, input, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0):
        return F.local_response_norm(input, size=size, alpha=alpha, beta=beta, k=k)


class AVGPoolLayer(nn.Module):
    def forward(self, input, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False):
        return F.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)