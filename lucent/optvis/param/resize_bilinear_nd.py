# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides resize_bilinear_nd.
This module provides resize_bilinear_nd, a function for resizing a tensor
with bilinear interpolation in n dimensions. It iteratively
applies torch.nn.Upsample (which can only resize 2 dimensions).
"""

from __future__ import absolute_import, division, print_function

import torch


def product(l):
    """Multiply together the elements of a list."""
    prod = 1
    for x in l:
        prod *= x
    return prod


def collapse_shape(shape, a, b):
    """Collapse `shape` outside the interval (`a`,`b`).
    This function collapses `shape` outside the interval (`a`,`b`) by
    multiplying the dimensions before `a` into a single dimension,
    and mutliplying the dimensions after `b` into a single dimension.
    Args:
        shape: a tensor shape
        a: integer, position in shape
        b: integer, position in shape
    Returns:
        The collapsed shape, represented as a list.
    Examples:
        [1, 2, 3, 4, 5], (a=0, b=2) => [1, 1, 2, 60] => [1, 60, 1, 2]
        [1, 2, 3, 4, 5], (a=1, b=3) => [1, 2, 3, 20] => [1, 20, 2, 3]
        [1, 2, 3, 4, 5], (a=2, b=4) => [2, 3, 4, 5 ] => [2, 5, 3, 4]
        [1, 2, 3, 4, 5], (a=3, b=5) => [6, 4, 5, 1 ] => [6, 1, 4, 5]
    """
    shape = list(shape)
    if a < 0:
        n_pad = -a
        pad = n_pad * [1]
        return collapse_shape(pad + shape, a + n_pad, b + n_pad)
    if b > len(shape):
        n_pad = b - len(shape)
        pad = n_pad * [1]
        return collapse_shape(shape + pad, a, b)
    return [product(shape[:a])] + shape[a:b] + [product(shape[b:])]


def resize_bilinear_nd(t, target_shape):
    """Bilinear resizes a tensor t to have shape target_shape.
    This function bilinearly resizes a n-dimensional tensor by iteratively
    applying tf.image.resize_bilinear (which can only resize 2 dimensions).
    For bilinear interpolation, the order in which it is applied does not matter.
    Args:
        t: tensor to be resized
        target_shape: the desired shape of the new tensor.
    Returns:
     The resized tensor
    """
    shape = list(t.shape)
    target_shape = list(target_shape)
    assert len(shape) == len(target_shape)

    # We progressively move through the shape, resizing dimensions...
    d = 0
    while d < len(shape):

        # If we don't need to deal with the next dimension, step over it
        if shape[d] == target_shape[d]:
            d += 1
            continue

        # Otherwise, we'll resize the next two dimensions...
        # If d+2 doesn't need to be resized, this will just be a null op for it
        new_shape = shape[:]
        new_shape[d:d+2] = target_shape[d:d+2]

        # The helper collapse_shape() makes our shapes 4-dimensional with
        # the two dimensions we want to deal with on the outside.
        shape_ = collapse_shape(shape, d, d+2)
        new_shape_ = collapse_shape(new_shape, d, d+2)

        # We can then reshape and use torch.nn.Upsample() on the
        # outer two dimensions.
        t_ = t.view(shape_)
        # transpose [0, 1, 2, 3] to [0, 3, 1, 2]
        t_ = torch.transpose(t_, 1, 3)
        t_ = torch.transpose(t_, 2, 3)
        upsample = torch.nn.Upsample(size=new_shape_[1:3], mode='bilinear', align_corners=True)
        t_ = upsample(t_)
        t_ = torch.transpose(t_, 2, 3)
        t_ = torch.transpose(t_, 1, 3)
        
        # And then reshape back to our uncollapsed version, having finished resizing
        # two more dimensions in our shape.
        t = t_.reshape(new_shape)
        shape = new_shape
        d += 2

    return t
