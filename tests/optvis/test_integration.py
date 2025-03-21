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

import pytest

import torch
from lucent.optvis import param, render
from lucent.modelzoo import inceptionv1
from lucent.util import DEFAULT_DEVICE


@pytest.fixture
def inceptionv1_model():
    device = torch.device(DEFAULT_DEVICE)
    model = inceptionv1().to(device).eval()
    return model


@pytest.mark.parametrize("decorrelate", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_integration(inceptionv1_model, decorrelate, fft):
    obj = "mixed3a_1x1_pre_relu_conv:0"
    param_f = lambda: param.image(224, decorrelate=decorrelate, fft=fft)
    optimizer = lambda params: torch.optim.Adam(params, lr=0.1)
    rendering = render.render_vis(
        inceptionv1_model,
        obj,
        param_f,
        optimizer=optimizer,
        thresholds=(1, 2),
        verbose=True,
        show_inline=True,
    )
    start_image, end_image = rendering
    assert (start_image != end_image).any()
