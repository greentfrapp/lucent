import pytest

import torch
from lucent.optvis import objectives, param, render, transform
from lucent.modelzoo import InceptionV1


@pytest.mark.parametrize("decorrelate", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_integration(decorrelate, fft):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionV1().to(device)
    model.eval()
    obj = "mixed3a_1x1_pre_relu_conv:0"
    params, image = param.image(224, decorrelate=decorrelate, fft=fft)
    optimizer = torch.optim.Adam(params, lr=0.1)
    rendering = render.render_vis(
        model,
        obj,
        image,
        optimizer=optimizer,
        thresholds=(1, 2),
        verbose=True,
        show_image=False,
    )
    start_image, end_image = rendering

    assert (start_image != end_image).any()
