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

"""Provides lowres_tensor()."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F

from lucent.optvis.param.resize_bilinear_nd import resize_bilinear_nd


def lowres_tensor(shape, underlying_shape, offset=None, sd=0.01):
    """Produces a tensor paramaterized by a interpolated lower resolution tensor.
    This is like what is done in a laplacian pyramid, but a bit more general. It
    can be a powerful way to describe images.
    Args:
        shape: desired shape of resulting tensor
        underlying_shape: shape of the tensor being resized into final tensor
        offset: Describes how to offset the interpolated vector (like phase in a
            Fourier transform). If None, apply no offset. If a scalar, apply the same
            offset to each dimension; if a list use each entry for each dimension.
            If a int, offset by that much. If False, do not offset. If True, offset by
            half the ratio between shape and underlying shape (analogous to 90
            degrees).
        sd: Standard deviation of initial tensor variable.
    Returns:
        A tensor paramaterized by a lower resolution tensorflow variable.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    underlying_t = (torch.randn(*underlying_shape) * sd).to(device).requires_grad_(True)
    if offset is not None:
        # Deal with non-list offset
        if not isinstance(offset, list):
            offset = len(shape) * [offset]
        # Deal with the non-int offset entries
        for n in range(len(offset)):
            if offset[n] is True:
                offset[n] = shape[n] / underlying_shape[n] / 2
            if offset[n] is False:
                offset[n] = 0
            offset[n] = int(offset[n])

    def inner():
        t = resize_bilinear_nd(underlying_t, shape)
        if offset is not None:
            # Actually apply offset by padding and then cropping off the excess.
            t = F.pad(t, offset, "reflect")
            t = t[: shape[0], : shape[1], : shape[2], : shape[3]]
        return t

    return [underlying_t], inner
