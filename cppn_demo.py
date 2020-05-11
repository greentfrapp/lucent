import torch
from torchvision import models
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image

from lucent.optvis import render


class composite_activation(torch.nn.Module):
    def __init__(self):
        super(composite_activation, self).__init__()

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)

def xor_loss(T):
    tensor = T("input")
    loss = -(torch.square(tensor[:, :, 0, 0]) + \
            torch.square(1 - tensor[:, :, -1, 0]) + \
            torch.square(tensor[:, :, -1, -1]) + \
            torch.square(1 - tensor[:, :, 0, -1]))
    return torch.sum(loss)

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

    input_tensor = torch.unsqueeze(torch.stack([x, y], dim=0), dim=0).requires_grad_(True).to(device)

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
    # torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    return net.parameters(), lambda: net(input_tensor)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(pretrained=True).to(device)
    model.eval()
    params, image = image_cppn(224)
    optimizer = torch.optim.Adam(params, lr=0.005)
    
    render.render_vis(model, "inception4b:87", image, optimizer=optimizer, thresholds=(256,))
    # render.render_vis(model, xor_loss, image, optimizer=optimizer, thresholds=(10,))


if __name__ == "__main__":
    main()