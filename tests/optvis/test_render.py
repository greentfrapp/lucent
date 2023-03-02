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
from lucent.optvis import render, param
from lucent.modelzoo import inceptionv1


@pytest.fixture(params=[True, False])
def inceptionv1_model(request):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1().to(device).eval()
    if request.param:
        model = torch.nn.DataParallel(model)
    return model

def test_render_vis(inceptionv1_model):
    thresholds = (1, 2)
    imgs = render.render_vis(inceptionv1_model, "mixed4a:0", thresholds=thresholds, show_image=False)
    assert len(imgs) == len(thresholds)
    assert imgs[0].shape == (1, 128, 128, 3)

def test_modelhook(inceptionv1_model):
    _, image_f = param.image(224)
    with render.ModelHook(inceptionv1_model, image_f) as hook:
        inceptionv1_model(image_f())
        assert hook("input").shape == (1, 3, 224, 224)
        assert hook("labels").shape == (1, 1008)

def test_partial_modelhook(inceptionv1_model):
    _, image_f = param.image(224)
    with render.ModelHook(inceptionv1_model, image_f, layer_names=["mixed4a"]) as hook:
        inceptionv1_model(image_f())
        assert hook("input").shape == (1, 3, 224, 224)
        assert hook("labels").shape == (1, 1008)
        assert hook("mixed4a").shape == (1, 508, 14, 14)
        with pytest.raises(AssertionError):
            print(hook("mixed4b").shape)
