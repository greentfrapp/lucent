import pytest

import torch
import numpy as np
from lucent.optvis import param, render, objectives


def xor_loss(T):
    tensor = T
    loss = -(torch.square(tensor[:, :, 0, 0]) + \
            torch.square(1 - tensor[:, :, -1, 0]) + \
            torch.square(tensor[:, :, -1, -1]) + \
            torch.square(1 - tensor[:, :, 0, -1]))
    return torch.sum(loss)

def test_cppn_fits_xor():
    params, image = param.cppn(16)
    optimizer = torch.optim.Adam(params, lr=0.01)
    objective_f = objectives.as_objective(xor_loss)
    for _ in range(200):
        optimizer.zero_grad()
        loss = objective_f(image())
        loss.backward()
        optimizer.step()
        vis = image()[0]
        close_enough = (
            vis[:, 0, 0].mean() > .99
            and vis[:, -1, -1].mean() > .99
            and vis[:, -1, 0].mean() < .01
            and vis[:, 0, -1].mean() < .01
        )
    if close_enough:
        return
    assert False, "fitting XOR took more than 200 steps, failing test"
