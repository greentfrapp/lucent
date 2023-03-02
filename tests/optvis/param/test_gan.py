import pytest

import torch
import numpy as np
from lucent.optvis import param, render, objectives
from lucent.optvis.param.gan import upconvGAN
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 5

@pytest.fixture
def inceptionv1_model():
    model = inceptionv1().to(device).eval()
    return model

def test_fc6gan_load():
    """ Test if gan could be downloaded and loaded
    It will download the model and store it locally
    """
    G = upconvGAN("fc6").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
        imagef = lambda:  G.visualize(code)
        return [code], imagef
    code, imagef = GANparam(batch=2, sd=1)
    img = imagef()
    assert img.shape == (2, 3, 256, 256), "Cannot forward fc6 GAN, shape incorrect."

def test_fc7gan_load():
    G = upconvGAN("fc7").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
        imagef = lambda: G.visualize(code)
        return [code], imagef
    code, imagef = GANparam(batch=2, sd=1)
    img = imagef()
    assert img.shape == (2, 3, 256, 256), "Cannot forward fc7 GAN, shape incorrect."

def test_fc8gan_load():
    G = upconvGAN("fc8").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
        imagef = lambda: G.visualize(code)
        return [code], imagef
    code, imagef = GANparam(batch=2, sd=1)
    img = imagef()
    assert img.shape == (2, 3, 256, 256), "Cannot forward fc8 GAN, shape incorrect."

def test_pool5gan_load():
    G = upconvGAN("pool5").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen, 6, 6)) * sd).to(device).requires_grad_(True)
        imagef = lambda: G.visualize(code)
        return [code], imagef
    code, imagef = GANparam(batch=2, sd=1)
    img = imagef()
    assert img.shape == (2, 3, 256, 256), "Cannot forward fc8 GAN, shape incorrect."


def assert_gan_gradient_descent(GANparam, objective, model):
    params, image = GANparam()
    optimizer = torch.optim.Adam(params, lr=0.05)
    with render.ModelHook(model, image) as T:
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


def test_gan_img_optim(inceptionv1_model):
    """ Test if GAN generated image could be optimized """
    G = upconvGAN("fc6").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
        imagef = lambda:  G.visualize(code)
        return [code], imagef
    objective = objectives.neuron("input", 0)
    assert_gan_gradient_descent(GANparam, objective, inceptionv1_model)


def test_gan_deep_optim(inceptionv1_model):
    """ Test if GAN generated image could be optimized """
    G = upconvGAN("fc6").to(device)

    def GANparam(batch=1, sd=1):
        code = (torch.randn((batch, G.codelen)) * sd).to(device).requires_grad_(True)
        imagef = lambda:  G.visualize(code)
        return [code], imagef

    objective = objectives.channel("mixed3a_1x1_pre_relu_conv", 0)
    assert_gan_gradient_descent(GANparam, objective, inceptionv1_model)
