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
import numpy as np
from lucent.optvis import objectives, param, render, transform
from lucent.modelzoo import inceptionv1


NUM_STEPS = 5


@pytest.fixture
def inceptionv1_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1().to(device).eval()
    return model


def assert_gradient_descent(objective, model):
    params, image = param.image(224, batch=2)
    optimizer = torch.optim.Adam(params, lr=0.05)
    T = render.hook_model(model, image)
    objective_f = objectives.as_objective(objective)
    model(image())
    start_value = objective_f(T)
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        model(image())
        loss = objective_f(T)
        loss.backward()
        optimizer.step()
    end_value = objective_f(T)
    assert start_value > end_value


def test_neuron(inceptionv1_model):
    objective = objectives.neuron("mixed3a_1x1_pre_relu_conv", 0)
    assert_gradient_descent(objective, inceptionv1_model)


def test_channel(inceptionv1_model):
    objective = objectives.channel("mixed3a_1x1_pre_relu_conv", 0)
    assert_gradient_descent(objective, inceptionv1_model)


def test_sum(inceptionv1_model):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = channel(21) + channel(32)
    assert_gradient_descent(objective, inceptionv1_model)


def test_linear_transform(inceptionv1_model):
    objective = 1 + 1 * -objectives.channel("mixed4a", 0) / 1 - 1
    assert_gradient_descent(objective, inceptionv1_model)


def test_mul_div_raises():
    with pytest.raises(Exception) as excinfo:   
        objective = objectives.channel("mixed4a", 0) / objectives.channel("mixed4a", 0)
    assert str(excinfo.value) == "Can only divide by int or float. Received type <class 'lucent.optvis.objectives.Objective'>"
    with pytest.raises(Exception) as excinfo:   
        objective = objectives.channel("mixed4a", 0) * objectives.channel("mixed4a", 0)
    assert str(excinfo.value) == "Can only multiply by int or float. Received type <class 'lucent.optvis.objectives.Objective'>"


def test_channel_interpolate(inceptionv1_model):
    objective = objectives.channel_interpolate("mixed4a", 465, "mixed4a", 460)
    assert_gradient_descent(objective, inceptionv1_model)


def test_diversity(inceptionv1_model):
    objective = objectives.channel("mixed4a", 0) - 100 * objectives.diversity("mixed4a")
    assert_gradient_descent(objective, inceptionv1_model)
