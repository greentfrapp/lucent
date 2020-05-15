import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Path to model, see https://github.com/greentfrapp/lucent/tree/master/lucent/modelzoo
    modelpath = "inception5h.pth"
    model = inceptionv1(pretrained=True, modelpath=modelpath)
    model.to(device).eval()

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

    # Following transforms from the Lucid tutorial
    transforms = [
        transform.pad(16),
        transform.jitter(8),
        transform.random_scale([n/100. for n in range(80, 120)]),
        transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
        transform.jitter(2),
    ]

    scaled_image = lambda: image() * 255 # InceptionV1 takes [0, 255] input
    optimizer = torch.optim.Adam(params, lr=lr)
    render.render_vis(model, obj, scaled_image, optimizer, transforms=transforms, thresholds=(512,))


if __name__ == "__main__":
    main()
