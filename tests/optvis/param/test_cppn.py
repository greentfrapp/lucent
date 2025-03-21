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
from lucent.optvis import param, objectives


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
