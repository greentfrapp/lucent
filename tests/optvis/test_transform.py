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

from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from lucent.optvis import transform
import lucent.util as ut


def test_pad_reflect():
    pad = transform.pad(1)
    tensor = torch.ones(1, 3, 2, 2).to(ut.DEFAULT_DEVICE)
    assert torch.all(pad(tensor).eq(torch.ones(1, 3, 4, 4).to(ut.DEFAULT_DEVICE)))


def test_pad_constant():
    pad = transform.pad(1, mode="constant")
    tensor = torch.ones(1, 3, 2, 2).to(ut.DEFAULT_DEVICE)
    assert torch.all(pad(tensor).eq(torch.tensor([[
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
    ]]).to(ut.DEFAULT_DEVICE)))


def test_random_scale_down():
    scale = transform.random_scale([0.33])
    tensor = torch.ones(1, 3, 3, 3).to(ut.DEFAULT_DEVICE)
    assert torch.all(scale(tensor).eq(torch.tensor([[
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ]]).to(ut.DEFAULT_DEVICE)))


def test_random_scale_up():
    scale = transform.random_scale([2])
    tensor = torch.ones(1, 3, 1, 1).to(ut.DEFAULT_DEVICE)
    assert torch.all(scale(tensor).eq(torch.ones(1, 3, 2, 2).to(ut.DEFAULT_DEVICE)))


def test_random_rotate_even_size():
    rotate = transform.random_rotate([np.pi/2], units="rads")
    tensor = torch.tensor([[
        [[0, 1], [0, 1]],
        [[0, 1], [0, 1]],
        [[0, 1], [0, 1]],
    ]]).float().to(ut.DEFAULT_DEVICE)

    target = torch.tensor([[
        [[1, 1], [0, 0]],
        [[1, 1], [0, 0]],
        [[1, 1], [0, 0]],
    ]]).float().to(ut.DEFAULT_DEVICE)
    # Looks like on some backends there is some numerical instability here, so we allow a small tolerance
    assert torch.allclose(rotate(tensor), target, atol=1e-5)


def test_random_rotate_odd_size():
    rotate = transform.random_rotate([90])
    tensor = torch.tensor([[
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    ]]).to(ut.DEFAULT_DEVICE)
    assert torch.all(rotate(tensor).eq(torch.tensor([[
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    ]]).to(ut.DEFAULT_DEVICE)))


def test_normalize():
    normalize = transform.normalize()
    tensor = torch.zeros(1, 3, 1, 1).to(ut.DEFAULT_DEVICE)
    assert torch.allclose(normalize(tensor), torch.tensor([[
        [[-0.485/0.229]],
        [[-0.456/0.224]],
        [[-0.406/0.225]]
    ]]).to(ut.DEFAULT_DEVICE))
