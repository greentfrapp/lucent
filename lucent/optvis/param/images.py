"""High-level wrapper for paramaterizing images."""

import torch
from collections import OrderedDict
import numpy as np

from lucent.optvis.param.spatial import pixel_image, fft_image
from lucent.optvis.param.color import to_valid_rgb


def image(w, h=None, sd=None, decorrelate=True,
          fft=True, alpha=False, channels=None):
    h = h or w
    ch = channels or (4 if alpha else 3)
    shape = [1, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, t = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(t, decorrelate=False)
    else:
        output = to_valid_rgb(t, decorrelate=decorrelate)
        # if alpha:
        #     def inner():
        #         a = tf.nn.sigmoid(t()[..., 3:])
        #         output = torch.cat([output, a], -1)
        #         return output
        #     return params, inner
    return params, output

class composite_activation(torch.nn.Module):
    def __init__(self):
        super(composite_activation, self).__init__()

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)

def image_cppn(size,
    num_output_channels=3,
    num_hidden_channels=24,
    num_layers=8,
    activation_fn=composite_activation,
    normalize=False):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.unsqueeze(torch.stack([x, y], dim=0), dim=0).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = 3
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i),  activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, np.sqrt(1/m.in_channels))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    return net.parameters(), lambda: net(input_tensor)