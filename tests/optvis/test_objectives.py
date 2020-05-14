from __future__ import absolute_import, division, print_function

import pytest

import torch
import numpy as np
from lucent.optvis import objectives, param, render, transform
from lucent.modelzoo import InceptionV1


NUM_STEPS = 3


@pytest.fixture
def inceptionv1():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionV1().to(device)
    model.eval()
    return model

def assert_gradient_descent(objective, model):
    params, image = param.image(224)
    optimizer = torch.optim.Adam(params, lr=0.1)
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

def test_neuron(inceptionv1):
    objective = objectives.neuron("mixed3a_1x1_pre_relu_conv", 0)
    assert_gradient_descent(objective, inceptionv1)

def test_channel(inceptionv1):
    objective = objectives.channel("mixed3a_1x1_pre_relu_conv", 0)
    assert_gradient_descent(objective, inceptionv1)

def test_sum(inceptionv1):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = channel(21) + channel(32)
    assert_gradient_descent(objective, inceptionv1)
