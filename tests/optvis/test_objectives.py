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
from lucent.util import set_seed
from lucent.optvis import objectives, param, render
from lucent.modelzoo import inceptionv1
from lucent.util import DEFAULT_DEVICE


set_seed(137)


NUM_STEPS = 5
device = torch.device(DEFAULT_DEVICE)

@pytest.fixture
def inceptionv1_model():
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

def test_neuron_weight(inceptionv1_model):
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.neuron_weight("mixed3a_1x1_pre_relu_conv", weight, x=None, y=None,)
    assert_gradient_descent(objective, inceptionv1_model)

def test_localgroup_weight(inceptionv1_model):
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.localgroup_weight("mixed3a_1x1_pre_relu_conv", weight, x=10, y=10, wx=3, wy=3)  # a 3 by 3 square start from (10, 10)
    assert_gradient_descent(objective, inceptionv1_model)

def test_channel_weight(inceptionv1_model):
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.channel_weight("mixed3a_1x1_pre_relu_conv", weight)
    assert_gradient_descent(objective, inceptionv1_model)


def test_sum(inceptionv1_model):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = objectives.Objective.sum([channel(21), channel(32)])
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


def test_blur_input_each_step(inceptionv1_model):
    objective = objectives.blur_input_each_step()
    assert_gradient_descent(objective, inceptionv1_model)


def test_channel_interpolate(inceptionv1_model):
    objective = objectives.channel_interpolate("mixed4a", 465, "mixed4a", 460)
    assert_gradient_descent(objective, inceptionv1_model)


def test_alignment(inceptionv1_model):
    objective = objectives.alignment("mixed4a")
    assert_gradient_descent(objective, inceptionv1_model)


def test_diversity(inceptionv1_model):
    objective = objectives.channel("mixed4a", 0) - 100 * objectives.diversity("mixed4a")
    assert_gradient_descent(objective, inceptionv1_model)


def test_direction(inceptionv1_model):
    direction = torch.rand(512, device=next(inceptionv1_model.parameters()).device) * 1000
    objective = objectives.direction(layer='mixed4c', direction=direction)
    assert_gradient_descent(objective, inceptionv1_model)


def test_direction_neuron(inceptionv1_model):
    direction = torch.rand(512, device=next(inceptionv1_model.parameters()).device) * 1000
    objective = objectives.direction_neuron(layer='mixed4c', direction=direction)
    assert_gradient_descent(objective, inceptionv1_model)
