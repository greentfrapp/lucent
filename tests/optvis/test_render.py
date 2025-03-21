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

from lucent.optvis import render, param
from lucent.modelzoo import inceptionv1
from lucent.util import DEFAULT_DEVICE


@pytest.fixture
def inceptionv1_model():
    model = inceptionv1().to(DEFAULT_DEVICE).eval()
    return model


def test_render_vis(inceptionv1_model):
    thresholds = (1, 2)
    imgs = render.render_vis(inceptionv1_model, "mixed4a:0", thresholds=thresholds, show_image=False)
    assert len(imgs) == len(thresholds)
    assert imgs[0].shape == (1, 128, 128, 3)


def test_hook_model(inceptionv1_model):
    _, image_f = param.image(224)
    hook = render.hook_model(inceptionv1_model, image_f)
    inceptionv1_model(image_f())
    assert hook("input").shape == (1, 3, 224, 224)
    assert hook("labels").shape == (1, 1008)
