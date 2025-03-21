import torch

from lucent.optvis import render, param
from lucent.modelzoo import inceptionv1

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1(pretrained=True)
    model.to(device).eval()

    CPPN = False

    SPATIAL_DECORRELATION = True
    CHANNEL_DECORRELATION = True

    if CPPN:
        # CPPN parameterization
        param_f = lambda: param.cppn(224, device=device)
        opt = lambda params: torch.optim.Adam(params, 5e-3)
        # Some objectives work better with CPPN than others
        obj = "mixed4d_3x3_bottleneck_pre_relu_conv:139"
    else:
        param_f = lambda: param.image(224, fft=SPATIAL_DECORRELATION, decorrelate=CHANNEL_DECORRELATION, device=device)
        opt = lambda params: torch.optim.Adam(params, 5e-2)
        obj = "mixed4a:476"

    render.render_vis(model, obj, param_f, opt)


if __name__ == "__main__":
    main()
