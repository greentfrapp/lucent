from __future__ import absolute_import, division, print_function

import pytest

import torch
import numpy as np
from lucent.optvis import transform


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_pad_reflect():
    pad = transform.pad(1)
    tensor = torch.ones(1, 3, 2, 2).to(device)
    assert torch.all(pad(tensor).eq(torch.ones(1, 3, 4, 4).to(device)))

def test_pad_constant():
    pad = transform.pad(1, mode="constant")
    tensor = torch.ones(1, 3, 2, 2).to(device)
    assert torch.all(pad(tensor).eq(torch.tensor([[
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5], [0.5, 0.5, 0.5, 0.5]],
    ]]).to(device)))

def test_random_scale_down():
    scale = transform.random_scale([0.33])
    tensor = torch.ones(1, 3, 3, 3).to(device)
    assert torch.all(scale(tensor).eq(torch.tensor([[
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ]]).to(device)))

def test_random_scale_up():
    scale = transform.random_scale([2])
    tensor = torch.ones(1, 3, 1, 1).to(device)
    assert torch.all(scale(tensor).eq(torch.ones(1, 3, 2, 2).to(device)))

def test_random_rotate_even_size():
    rotate = transform.random_rotate([np.pi/2], units="rads")
    tensor = torch.tensor([[
        [[0, 1], [0, 1]],
        [[0, 1], [0, 1]],
        [[0, 1], [0, 1]],
    ]]).to(device)
    assert torch.all(rotate(tensor).eq(torch.tensor([[
        [[1, 1], [0, 0]],
        [[1, 1], [0, 0]],
        [[1, 1], [0, 0]],
    ]]).to(device)))

def test_random_rotate_odd_size():
    rotate = transform.random_rotate([90])
    tensor = torch.tensor([[
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    ]]).to(device)
    assert torch.all(rotate(tensor).eq(torch.tensor([[
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    ]]).to(device)))

def test_normalize():
    normalize = transform.normalize()
    tensor = torch.zeros(1, 3, 1, 1).to(device)
    print(normalize(tensor))
    assert torch.allclose(normalize(tensor), torch.tensor([[
        [[-0.485/0.229]],
        [[-0.456/0.224]],
        [[-0.406/0.225]]
    ]]).to(device))
