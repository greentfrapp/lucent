"""
Original work by ProGamerGov at
https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/models/inception/inception5h.py

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

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucent.modelzoo.inceptionv1 import helper_layers

model_urls = {
    # InceptionV1 model used in Lucid examples, converted by ProGamerGov
    "inceptionv1": "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth",
}


class InceptionV1(nn.Module):
    def __init__(self, pretrained=False, progress=True, redirected_ReLU=True):
        super(InceptionV1, self).__init__()
        self.conv2d0_pre_relu_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            groups=1,
            bias=True,
        )
        self.conv2d1_pre_relu_conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv2d2_pre_relu_conv = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=192,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=192,
            out_channels=96,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=192,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=192,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=96,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3a_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=256,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=128,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed3b_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=32,
            out_channels=96,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=480,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=480,
            out_channels=96,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=480,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=480,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=96,
            out_channels=204,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4a_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=16,
            out_channels=48,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=508,
            out_channels=160,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=508,
            out_channels=112,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=508,
            out_channels=24,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=508,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=112,
            out_channels=224,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4b_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=24,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=24,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4c_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=24,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=112,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=144,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=512,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=144,
            out_channels=288,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4d_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=528,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=528,
            out_channels=160,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=528,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=528,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=160,
            out_channels=320,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed4e_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=32,
            out_channels=128,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=160,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=48,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=160,
            out_channels=320,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5a_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_1x1_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=384,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=192,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=48,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels=832,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_3x3_pre_relu_conv = nn.Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.mixed5b_5x5_pre_relu_conv = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=(5, 5),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.softmax2_pre_activation_matmul = nn.Linear(
            in_features=1024, out_features=1008, bias=True
        )

        self.add_layers(redirected_ReLU)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls["inceptionv1"], progress=progress
                )
            )

    def add_layers(self, redirected_ReLU=True):
        if redirected_ReLU:
            relu = helper_layers.RedirectedReluLayer
        else:
            relu = helper_layers.ReluLayer
        self.conv2d0 = relu()
        self.maxpool0 = helper_layers.MaxPool2dLayer()
        self.conv2d1 = relu()
        self.conv2d2 = relu()
        self.maxpool1 = helper_layers.MaxPool2dLayer()
        self.mixed3a_pool = helper_layers.MaxPool2dLayer()
        self.mixed3a_1x1 = relu()
        self.mixed3a_3x3_bottleneck = relu()
        self.mixed3a_5x5_bottleneck = relu()
        self.mixed3a_pool_reduce = relu()
        self.mixed3a_3x3 = relu()
        self.mixed3a_5x5 = relu()
        self.mixed3a = helper_layers.CatLayer()
        self.mixed3b_pool = helper_layers.MaxPool2dLayer()
        self.mixed3b_1x1 = relu()
        self.mixed3b_3x3_bottleneck = relu()
        self.mixed3b_5x5_bottleneck = relu()
        self.mixed3b_pool_reduce = relu()
        self.mixed3b_3x3 = relu()
        self.mixed3b_5x5 = relu()
        self.mixed3b = helper_layers.CatLayer()
        self.maxpool4 = helper_layers.MaxPool2dLayer()
        self.mixed4a_pool = helper_layers.MaxPool2dLayer()
        self.mixed4a_1x1 = relu()
        self.mixed4a_3x3_bottleneck = relu()
        self.mixed4a_5x5_bottleneck = relu()
        self.mixed4a_pool_reduce = relu()
        self.mixed4a_3x3 = relu()
        self.mixed4a_5x5 = relu()
        self.mixed4a = helper_layers.CatLayer()
        self.mixed4b_pool = helper_layers.MaxPool2dLayer()
        self.mixed4b_1x1 = relu()
        self.mixed4b_3x3_bottleneck = relu()
        self.mixed4b_5x5_bottleneck = relu()
        self.mixed4b_pool_reduce = relu()
        self.mixed4b_3x3 = relu()
        self.mixed4b_5x5 = relu()
        self.mixed4b = helper_layers.CatLayer()
        self.mixed4c_pool = helper_layers.MaxPool2dLayer()
        self.mixed4c_1x1 = relu()
        self.mixed4c_3x3_bottleneck = relu()
        self.mixed4c_5x5_bottleneck = relu()
        self.mixed4c_pool_reduce = relu()
        self.mixed4c_3x3 = relu()
        self.mixed4c_5x5 = relu()
        self.mixed4c = helper_layers.CatLayer()
        self.mixed4d_pool = helper_layers.MaxPool2dLayer()
        self.mixed4d_1x1 = relu()
        self.mixed4d_3x3_bottleneck = relu()
        self.mixed4d_5x5_bottleneck = relu()
        self.mixed4d_pool_reduce = relu()
        self.mixed4d_3x3 = relu()
        self.mixed4d_5x5 = relu()
        self.mixed4d = helper_layers.CatLayer()
        self.mixed4e_pool = helper_layers.MaxPool2dLayer()
        self.mixed4e_1x1 = relu()
        self.mixed4e_3x3_bottleneck = relu()
        self.mixed4e_5x5_bottleneck = relu()
        self.mixed4e_pool_reduce = relu()
        self.mixed4e_3x3 = relu()
        self.mixed4e_5x5 = relu()
        self.mixed4e = helper_layers.CatLayer()
        self.maxpool10 = helper_layers.MaxPool2dLayer()
        self.mixed5a_pool = helper_layers.MaxPool2dLayer()
        self.mixed5a_1x1 = relu()
        self.mixed5a_3x3_bottleneck = relu()
        self.mixed5a_5x5_bottleneck = relu()
        self.mixed5a_pool_reduce = relu()
        self.mixed5a_3x3 = relu()
        self.mixed5a_5x5 = relu()
        self.mixed5a = helper_layers.CatLayer()
        self.mixed5b_pool = helper_layers.MaxPool2dLayer()
        self.mixed5b_1x1 = relu()
        self.mixed5b_3x3_bottleneck = relu()
        self.mixed5b_5x5_bottleneck = relu()
        self.mixed5b_pool_reduce = relu()
        self.mixed5b_3x3 = relu()
        self.mixed5b_5x5 = relu()
        self.mixed5b = helper_layers.CatLayer()
        self.softmax2 = helper_layers.SoftMaxLayer()

    def forward(self, x):
        conv2d0_pre_relu_conv_pad = F.pad(x, (2, 3, 2, 3))
        conv2d0_pre_relu_conv = self.conv2d0_pre_relu_conv(conv2d0_pre_relu_conv_pad)
        conv2d0 = self.conv2d0(conv2d0_pre_relu_conv)
        maxpool0_pad = F.pad(conv2d0, (0, 1, 0, 1), value=float("-inf"))
        maxpool0 = self.maxpool0(
            maxpool0_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False
        )
        localresponsenorm0 = F.local_response_norm(
            maxpool0, size=9, alpha=9.99999974738e-05, beta=0.5, k=1
        )
        conv2d1_pre_relu_conv = self.conv2d1_pre_relu_conv(localresponsenorm0)
        conv2d1 = self.conv2d1(conv2d1_pre_relu_conv)
        conv2d2_pre_relu_conv_pad = F.pad(conv2d1, (1, 1, 1, 1))
        conv2d2_pre_relu_conv = self.conv2d2_pre_relu_conv(conv2d2_pre_relu_conv_pad)
        conv2d2 = self.conv2d2(conv2d2_pre_relu_conv)
        localresponsenorm1 = F.local_response_norm(
            conv2d2, size=9, alpha=9.99999974738e-05, beta=0.5, k=1
        )
        maxpool1_pad = F.pad(localresponsenorm1, (0, 1, 0, 1), value=float("-inf"))
        maxpool1 = self.maxpool1(
            maxpool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False
        )
        mixed3a_1x1_pre_relu_conv = self.mixed3a_1x1_pre_relu_conv(maxpool1)
        mixed3a_3x3_bottleneck_pre_relu_conv = (
            self.mixed3a_3x3_bottleneck_pre_relu_conv(maxpool1)
        )
        mixed3a_5x5_bottleneck_pre_relu_conv = (
            self.mixed3a_5x5_bottleneck_pre_relu_conv(maxpool1)
        )
        mixed3a_pool_pad = F.pad(maxpool1, (1, 1, 1, 1), value=float("-inf"))
        mixed3a_pool = self.mixed3a_pool(
            mixed3a_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed3a_1x1 = self.mixed3a_1x1(mixed3a_1x1_pre_relu_conv)
        mixed3a_3x3_bottleneck = self.mixed3a_3x3_bottleneck(
            mixed3a_3x3_bottleneck_pre_relu_conv
        )
        mixed3a_5x5_bottleneck = self.mixed3a_5x5_bottleneck(
            mixed3a_5x5_bottleneck_pre_relu_conv
        )
        mixed3a_pool_reduce_pre_relu_conv = self.mixed3a_pool_reduce_pre_relu_conv(
            mixed3a_pool
        )
        mixed3a_3x3_pre_relu_conv_pad = F.pad(mixed3a_3x3_bottleneck, (1, 1, 1, 1))
        mixed3a_3x3_pre_relu_conv = self.mixed3a_3x3_pre_relu_conv(
            mixed3a_3x3_pre_relu_conv_pad
        )
        mixed3a_5x5_pre_relu_conv_pad = F.pad(mixed3a_5x5_bottleneck, (2, 2, 2, 2))
        mixed3a_5x5_pre_relu_conv = self.mixed3a_5x5_pre_relu_conv(
            mixed3a_5x5_pre_relu_conv_pad
        )
        mixed3a_pool_reduce = self.mixed3a_pool_reduce(
            mixed3a_pool_reduce_pre_relu_conv
        )
        mixed3a_3x3 = self.mixed3a_3x3(mixed3a_3x3_pre_relu_conv)
        mixed3a_5x5 = self.mixed3a_5x5(mixed3a_5x5_pre_relu_conv)
        mixed3a = self.mixed3a(
            (mixed3a_1x1, mixed3a_3x3, mixed3a_5x5, mixed3a_pool_reduce), 1
        )
        mixed3b_1x1_pre_relu_conv = self.mixed3b_1x1_pre_relu_conv(mixed3a)
        mixed3b_3x3_bottleneck_pre_relu_conv = (
            self.mixed3b_3x3_bottleneck_pre_relu_conv(mixed3a)
        )
        mixed3b_5x5_bottleneck_pre_relu_conv = (
            self.mixed3b_5x5_bottleneck_pre_relu_conv(mixed3a)
        )
        mixed3b_pool_pad = F.pad(mixed3a, (1, 1, 1, 1), value=float("-inf"))
        mixed3b_pool = self.mixed3b_pool(
            mixed3b_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed3b_1x1 = self.mixed3b_1x1(mixed3b_1x1_pre_relu_conv)
        mixed3b_3x3_bottleneck = self.mixed3b_3x3_bottleneck(
            mixed3b_3x3_bottleneck_pre_relu_conv
        )
        mixed3b_5x5_bottleneck = self.mixed3b_5x5_bottleneck(
            mixed3b_5x5_bottleneck_pre_relu_conv
        )
        mixed3b_pool_reduce_pre_relu_conv = self.mixed3b_pool_reduce_pre_relu_conv(
            mixed3b_pool
        )
        mixed3b_3x3_pre_relu_conv_pad = F.pad(mixed3b_3x3_bottleneck, (1, 1, 1, 1))
        mixed3b_3x3_pre_relu_conv = self.mixed3b_3x3_pre_relu_conv(
            mixed3b_3x3_pre_relu_conv_pad
        )
        mixed3b_5x5_pre_relu_conv_pad = F.pad(mixed3b_5x5_bottleneck, (2, 2, 2, 2))
        mixed3b_5x5_pre_relu_conv = self.mixed3b_5x5_pre_relu_conv(
            mixed3b_5x5_pre_relu_conv_pad
        )
        mixed3b_pool_reduce = self.mixed3b_pool_reduce(
            mixed3b_pool_reduce_pre_relu_conv
        )
        mixed3b_3x3 = self.mixed3b_3x3(mixed3b_3x3_pre_relu_conv)
        mixed3b_5x5 = self.mixed3b_5x5(mixed3b_5x5_pre_relu_conv)
        mixed3b = self.mixed3b(
            (mixed3b_1x1, mixed3b_3x3, mixed3b_5x5, mixed3b_pool_reduce), 1
        )
        maxpool4_pad = F.pad(mixed3b, (0, 1, 0, 1), value=float("-inf"))
        maxpool4 = self.maxpool4(
            maxpool4_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False
        )
        mixed4a_1x1_pre_relu_conv = self.mixed4a_1x1_pre_relu_conv(maxpool4)
        mixed4a_3x3_bottleneck_pre_relu_conv = (
            self.mixed4a_3x3_bottleneck_pre_relu_conv(maxpool4)
        )
        mixed4a_5x5_bottleneck_pre_relu_conv = (
            self.mixed4a_5x5_bottleneck_pre_relu_conv(maxpool4)
        )
        mixed4a_pool_pad = F.pad(maxpool4, (1, 1, 1, 1), value=float("-inf"))
        mixed4a_pool = self.mixed4a_pool(
            mixed4a_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed4a_1x1 = self.mixed4a_1x1(mixed4a_1x1_pre_relu_conv)
        mixed4a_3x3_bottleneck = self.mixed4a_3x3_bottleneck(
            mixed4a_3x3_bottleneck_pre_relu_conv
        )
        mixed4a_5x5_bottleneck = self.mixed4a_5x5_bottleneck(
            mixed4a_5x5_bottleneck_pre_relu_conv
        )
        mixed4a_pool_reduce_pre_relu_conv = self.mixed4a_pool_reduce_pre_relu_conv(
            mixed4a_pool
        )
        mixed4a_3x3_pre_relu_conv_pad = F.pad(mixed4a_3x3_bottleneck, (1, 1, 1, 1))
        mixed4a_3x3_pre_relu_conv = self.mixed4a_3x3_pre_relu_conv(
            mixed4a_3x3_pre_relu_conv_pad
        )
        mixed4a_5x5_pre_relu_conv_pad = F.pad(mixed4a_5x5_bottleneck, (2, 2, 2, 2))
        mixed4a_5x5_pre_relu_conv = self.mixed4a_5x5_pre_relu_conv(
            mixed4a_5x5_pre_relu_conv_pad
        )
        mixed4a_pool_reduce = self.mixed4a_pool_reduce(
            mixed4a_pool_reduce_pre_relu_conv
        )
        mixed4a_3x3 = self.mixed4a_3x3(mixed4a_3x3_pre_relu_conv)
        mixed4a_5x5 = self.mixed4a_5x5(mixed4a_5x5_pre_relu_conv)
        mixed4a = self.mixed4a(
            (mixed4a_1x1, mixed4a_3x3, mixed4a_5x5, mixed4a_pool_reduce), 1
        )
        mixed4b_1x1_pre_relu_conv = self.mixed4b_1x1_pre_relu_conv(mixed4a)
        mixed4b_3x3_bottleneck_pre_relu_conv = (
            self.mixed4b_3x3_bottleneck_pre_relu_conv(mixed4a)
        )
        mixed4b_5x5_bottleneck_pre_relu_conv = (
            self.mixed4b_5x5_bottleneck_pre_relu_conv(mixed4a)
        )
        mixed4b_pool_pad = F.pad(mixed4a, (1, 1, 1, 1), value=float("-inf"))
        mixed4b_pool = self.mixed4b_pool(
            mixed4b_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed4b_1x1 = self.mixed4b_1x1(mixed4b_1x1_pre_relu_conv)
        mixed4b_3x3_bottleneck = self.mixed4b_3x3_bottleneck(
            mixed4b_3x3_bottleneck_pre_relu_conv
        )
        mixed4b_5x5_bottleneck = self.mixed4b_5x5_bottleneck(
            mixed4b_5x5_bottleneck_pre_relu_conv
        )
        mixed4b_pool_reduce_pre_relu_conv = self.mixed4b_pool_reduce_pre_relu_conv(
            mixed4b_pool
        )
        mixed4b_3x3_pre_relu_conv_pad = F.pad(mixed4b_3x3_bottleneck, (1, 1, 1, 1))
        mixed4b_3x3_pre_relu_conv = self.mixed4b_3x3_pre_relu_conv(
            mixed4b_3x3_pre_relu_conv_pad
        )
        mixed4b_5x5_pre_relu_conv_pad = F.pad(mixed4b_5x5_bottleneck, (2, 2, 2, 2))
        mixed4b_5x5_pre_relu_conv = self.mixed4b_5x5_pre_relu_conv(
            mixed4b_5x5_pre_relu_conv_pad
        )
        mixed4b_pool_reduce = self.mixed4b_pool_reduce(
            mixed4b_pool_reduce_pre_relu_conv
        )
        mixed4b_3x3 = self.mixed4b_3x3(mixed4b_3x3_pre_relu_conv)
        mixed4b_5x5 = self.mixed4b_5x5(mixed4b_5x5_pre_relu_conv)
        mixed4b = self.mixed4b(
            (mixed4b_1x1, mixed4b_3x3, mixed4b_5x5, mixed4b_pool_reduce), 1
        )
        mixed4c_1x1_pre_relu_conv = self.mixed4c_1x1_pre_relu_conv(mixed4b)
        mixed4c_3x3_bottleneck_pre_relu_conv = (
            self.mixed4c_3x3_bottleneck_pre_relu_conv(mixed4b)
        )
        mixed4c_5x5_bottleneck_pre_relu_conv = (
            self.mixed4c_5x5_bottleneck_pre_relu_conv(mixed4b)
        )
        mixed4c_pool_pad = F.pad(mixed4b, (1, 1, 1, 1), value=float("-inf"))
        mixed4c_pool = self.mixed4c_pool(
            mixed4c_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed4c_1x1 = self.mixed4c_1x1(mixed4c_1x1_pre_relu_conv)
        mixed4c_3x3_bottleneck = self.mixed4c_3x3_bottleneck(
            mixed4c_3x3_bottleneck_pre_relu_conv
        )
        mixed4c_5x5_bottleneck = self.mixed4c_5x5_bottleneck(
            mixed4c_5x5_bottleneck_pre_relu_conv
        )
        mixed4c_pool_reduce_pre_relu_conv = self.mixed4c_pool_reduce_pre_relu_conv(
            mixed4c_pool
        )
        mixed4c_3x3_pre_relu_conv_pad = F.pad(mixed4c_3x3_bottleneck, (1, 1, 1, 1))
        mixed4c_3x3_pre_relu_conv = self.mixed4c_3x3_pre_relu_conv(
            mixed4c_3x3_pre_relu_conv_pad
        )
        mixed4c_5x5_pre_relu_conv_pad = F.pad(mixed4c_5x5_bottleneck, (2, 2, 2, 2))
        mixed4c_5x5_pre_relu_conv = self.mixed4c_5x5_pre_relu_conv(
            mixed4c_5x5_pre_relu_conv_pad
        )
        mixed4c_pool_reduce = self.mixed4c_pool_reduce(
            mixed4c_pool_reduce_pre_relu_conv
        )
        mixed4c_3x3 = self.mixed4c_3x3(mixed4c_3x3_pre_relu_conv)
        mixed4c_5x5 = self.mixed4c_5x5(mixed4c_5x5_pre_relu_conv)
        mixed4c = self.mixed4c(
            (mixed4c_1x1, mixed4c_3x3, mixed4c_5x5, mixed4c_pool_reduce), 1
        )
        mixed4d_1x1_pre_relu_conv = self.mixed4d_1x1_pre_relu_conv(mixed4c)
        mixed4d_3x3_bottleneck_pre_relu_conv = (
            self.mixed4d_3x3_bottleneck_pre_relu_conv(mixed4c)
        )
        mixed4d_5x5_bottleneck_pre_relu_conv = (
            self.mixed4d_5x5_bottleneck_pre_relu_conv(mixed4c)
        )
        mixed4d_pool_pad = F.pad(mixed4c, (1, 1, 1, 1), value=float("-inf"))
        mixed4d_pool = self.mixed4d_pool(
            mixed4d_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed4d_1x1 = self.mixed4d_1x1(mixed4d_1x1_pre_relu_conv)
        mixed4d_3x3_bottleneck = self.mixed4d_3x3_bottleneck(
            mixed4d_3x3_bottleneck_pre_relu_conv
        )
        mixed4d_5x5_bottleneck = self.mixed4d_5x5_bottleneck(
            mixed4d_5x5_bottleneck_pre_relu_conv
        )
        mixed4d_pool_reduce_pre_relu_conv = self.mixed4d_pool_reduce_pre_relu_conv(
            mixed4d_pool
        )
        mixed4d_3x3_pre_relu_conv_pad = F.pad(mixed4d_3x3_bottleneck, (1, 1, 1, 1))
        mixed4d_3x3_pre_relu_conv = self.mixed4d_3x3_pre_relu_conv(
            mixed4d_3x3_pre_relu_conv_pad
        )
        mixed4d_5x5_pre_relu_conv_pad = F.pad(mixed4d_5x5_bottleneck, (2, 2, 2, 2))
        mixed4d_5x5_pre_relu_conv = self.mixed4d_5x5_pre_relu_conv(
            mixed4d_5x5_pre_relu_conv_pad
        )
        mixed4d_pool_reduce = self.mixed4d_pool_reduce(
            mixed4d_pool_reduce_pre_relu_conv
        )
        mixed4d_3x3 = self.mixed4d_3x3(mixed4d_3x3_pre_relu_conv)
        mixed4d_5x5 = self.mixed4d_5x5(mixed4d_5x5_pre_relu_conv)
        mixed4d = self.mixed4d(
            (mixed4d_1x1, mixed4d_3x3, mixed4d_5x5, mixed4d_pool_reduce), 1
        )
        mixed4e_1x1_pre_relu_conv = self.mixed4e_1x1_pre_relu_conv(mixed4d)
        mixed4e_3x3_bottleneck_pre_relu_conv = (
            self.mixed4e_3x3_bottleneck_pre_relu_conv(mixed4d)
        )
        mixed4e_5x5_bottleneck_pre_relu_conv = (
            self.mixed4e_5x5_bottleneck_pre_relu_conv(mixed4d)
        )
        mixed4e_pool_pad = F.pad(mixed4d, (1, 1, 1, 1), value=float("-inf"))
        mixed4e_pool = self.mixed4e_pool(
            mixed4e_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed4e_1x1 = self.mixed4e_1x1(mixed4e_1x1_pre_relu_conv)
        mixed4e_3x3_bottleneck = self.mixed4e_3x3_bottleneck(
            mixed4e_3x3_bottleneck_pre_relu_conv
        )
        mixed4e_5x5_bottleneck = self.mixed4e_5x5_bottleneck(
            mixed4e_5x5_bottleneck_pre_relu_conv
        )
        mixed4e_pool_reduce_pre_relu_conv = self.mixed4e_pool_reduce_pre_relu_conv(
            mixed4e_pool
        )
        mixed4e_3x3_pre_relu_conv_pad = F.pad(mixed4e_3x3_bottleneck, (1, 1, 1, 1))
        mixed4e_3x3_pre_relu_conv = self.mixed4e_3x3_pre_relu_conv(
            mixed4e_3x3_pre_relu_conv_pad
        )
        mixed4e_5x5_pre_relu_conv_pad = F.pad(mixed4e_5x5_bottleneck, (2, 2, 2, 2))
        mixed4e_5x5_pre_relu_conv = self.mixed4e_5x5_pre_relu_conv(
            mixed4e_5x5_pre_relu_conv_pad
        )
        mixed4e_pool_reduce = self.mixed4e_pool_reduce(
            mixed4e_pool_reduce_pre_relu_conv
        )
        mixed4e_3x3 = self.mixed4e_3x3(mixed4e_3x3_pre_relu_conv)
        mixed4e_5x5 = self.mixed4e_5x5(mixed4e_5x5_pre_relu_conv)
        mixed4e = self.mixed4e(
            (mixed4e_1x1, mixed4e_3x3, mixed4e_5x5, mixed4e_pool_reduce), 1
        )
        maxpool10_pad = F.pad(mixed4e, (0, 1, 0, 1), value=float("-inf"))
        maxpool10 = self.maxpool10(
            maxpool10_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False
        )
        mixed5a_1x1_pre_relu_conv = self.mixed5a_1x1_pre_relu_conv(maxpool10)
        mixed5a_3x3_bottleneck_pre_relu_conv = (
            self.mixed5a_3x3_bottleneck_pre_relu_conv(maxpool10)
        )
        mixed5a_5x5_bottleneck_pre_relu_conv = (
            self.mixed5a_5x5_bottleneck_pre_relu_conv(maxpool10)
        )
        mixed5a_pool_pad = F.pad(maxpool10, (1, 1, 1, 1), value=float("-inf"))
        mixed5a_pool = self.mixed5a_pool(
            mixed5a_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed5a_1x1 = self.mixed5a_1x1(mixed5a_1x1_pre_relu_conv)
        mixed5a_3x3_bottleneck = self.mixed5a_3x3_bottleneck(
            mixed5a_3x3_bottleneck_pre_relu_conv
        )
        mixed5a_5x5_bottleneck = self.mixed5a_5x5_bottleneck(
            mixed5a_5x5_bottleneck_pre_relu_conv
        )
        mixed5a_pool_reduce_pre_relu_conv = self.mixed5a_pool_reduce_pre_relu_conv(
            mixed5a_pool
        )
        mixed5a_3x3_pre_relu_conv_pad = F.pad(mixed5a_3x3_bottleneck, (1, 1, 1, 1))
        mixed5a_3x3_pre_relu_conv = self.mixed5a_3x3_pre_relu_conv(
            mixed5a_3x3_pre_relu_conv_pad
        )
        mixed5a_5x5_pre_relu_conv_pad = F.pad(mixed5a_5x5_bottleneck, (2, 2, 2, 2))
        mixed5a_5x5_pre_relu_conv = self.mixed5a_5x5_pre_relu_conv(
            mixed5a_5x5_pre_relu_conv_pad
        )
        mixed5a_pool_reduce = self.mixed5a_pool_reduce(
            mixed5a_pool_reduce_pre_relu_conv
        )
        mixed5a_3x3 = self.mixed5a_3x3(mixed5a_3x3_pre_relu_conv)
        mixed5a_5x5 = self.mixed5a_5x5(mixed5a_5x5_pre_relu_conv)
        mixed5a = self.mixed5a(
            (mixed5a_1x1, mixed5a_3x3, mixed5a_5x5, mixed5a_pool_reduce), 1
        )
        mixed5b_1x1_pre_relu_conv = self.mixed5b_1x1_pre_relu_conv(mixed5a)
        mixed5b_3x3_bottleneck_pre_relu_conv = (
            self.mixed5b_3x3_bottleneck_pre_relu_conv(mixed5a)
        )
        mixed5b_5x5_bottleneck_pre_relu_conv = (
            self.mixed5b_5x5_bottleneck_pre_relu_conv(mixed5a)
        )
        mixed5b_pool_pad = F.pad(mixed5a, (1, 1, 1, 1), value=float("-inf"))
        mixed5b_pool = self.mixed5b_pool(
            mixed5b_pool_pad,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            ceil_mode=False,
        )
        mixed5b_1x1 = self.mixed5b_1x1(mixed5b_1x1_pre_relu_conv)
        mixed5b_3x3_bottleneck = self.mixed5b_3x3_bottleneck(
            mixed5b_3x3_bottleneck_pre_relu_conv
        )
        mixed5b_5x5_bottleneck = self.mixed5b_5x5_bottleneck(
            mixed5b_5x5_bottleneck_pre_relu_conv
        )
        mixed5b_pool_reduce_pre_relu_conv = self.mixed5b_pool_reduce_pre_relu_conv(
            mixed5b_pool
        )
        mixed5b_3x3_pre_relu_conv_pad = F.pad(mixed5b_3x3_bottleneck, (1, 1, 1, 1))
        mixed5b_3x3_pre_relu_conv = self.mixed5b_3x3_pre_relu_conv(
            mixed5b_3x3_pre_relu_conv_pad
        )
        mixed5b_5x5_pre_relu_conv_pad = F.pad(mixed5b_5x5_bottleneck, (2, 2, 2, 2))
        mixed5b_5x5_pre_relu_conv = self.mixed5b_5x5_pre_relu_conv(
            mixed5b_5x5_pre_relu_conv_pad
        )
        mixed5b_pool_reduce = self.mixed5b_pool_reduce(
            mixed5b_pool_reduce_pre_relu_conv
        )
        mixed5b_3x3 = self.mixed5b_3x3(mixed5b_3x3_pre_relu_conv)
        mixed5b_5x5 = self.mixed5b_5x5(mixed5b_5x5_pre_relu_conv)
        mixed5b = self.mixed5b(
            (mixed5b_1x1, mixed5b_3x3, mixed5b_5x5, mixed5b_pool_reduce), 1
        )
        avgpool0 = F.avg_pool2d(
            mixed5b,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(0,),
            ceil_mode=False,
            count_include_pad=False,
        )
        avgpool0_reshape = torch.reshape(input=avgpool0, shape=(-1, 1024))
        softmax2_pre_activation_matmul = self.softmax2_pre_activation_matmul(
            avgpool0_reshape
        )
        softmax2 = self.softmax2(softmax2_pre_activation_matmul)
        return softmax2
