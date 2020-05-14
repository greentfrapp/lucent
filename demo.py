import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import InceptionV1

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionV1()
    model.to(device)
    # Path to model, see https://github.com/greentfrapp/lucent/tree/master/lucent/modelzoo
    modelpath = "inception5h.pth"
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    CPPN = False

    SPATIAL_DECORRELATION = True
    CHANNEL_DECORRELATION = True

    if CPPN:
        # CPPN parameterization
        params, image = param.cppn(224)
        lr = 5e-3
        # Some objectives work better with CPPN than others
        obj = "mixed4d_3x3_bottleneck_pre_relu_conv:139"
    else:
        params, image = param.image(224, fft=SPATIAL_DECORRELATION, decorrelate=CHANNEL_DECORRELATION)
        lr = 5e-2
        obj = "mixed4a:476"

    scaled_image = lambda: image() * 255 # InceptionV1 takes [0, 255] input
    optimizer = torch.optim.Adam(params, lr=lr)
    render.render_vis(model, obj, scaled_image, optimizer, thresholds=(512,))


if __name__ == "__main__":
    main()
