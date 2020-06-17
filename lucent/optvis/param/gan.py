"""
Copyright 2020 Binxu Wang
Use GAN as prior to do feature visualization.
This method is inspired by the work
    Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., & Clune, J.
    Synthesizing the preferred inputs for neurons in neural networks via deep generator networks.(2016) NIPS

The GAN model is imported from
    A. Dosovitskiy, T. Brox `Generating Images with Perceptual Similarity Metrics based on Deep Networks` (2016), NIPS.
    https://lmb.informatik.uni-freiburg.de/people/dosovits/code.html
the author translated the models (pool5-fc8) into pytorch and hosts the weights online.

Jun.4th 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
from os.path import join
from sys import platform
load_urls = True  # If you have downloaded the pt files you can set the netsdir and set load_urls as False.
netsdir = "~"      # the place you put the networks

model_urls = {"pool5" : "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA",
            "fc6": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4",
            "fc7": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw",
            "fc8": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU"}


def load_statedict_from_online(name="fc6"):
    torchhome = torch.hub._get_torch_home()
    ckpthome = join(torchhome, "checkpoints")
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, "upconvGAN_%s.pt"%name)
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None,
                                   progress=True)
    SD = torch.load(filepath)
    return SD


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


RGB_mean = torch.tensor([123.0, 117.0, 104.0])
RGB_mean = torch.reshape(RGB_mean, (1, 3, 1, 1))


class upconvGAN(nn.Module):
    def __init__(self, name="fc6", pretrained=True):
        """ `name`: can be ["fc6", "fc7", "fc8", "pool5"] """
        super(upconvGAN, self).__init__()
        self.name = name
        if name == "fc6" or name == "fc7":
            self.G = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ]))
            self.codelen = self.G[0].in_features
        elif name == "fc8":
            self.G = nn.Sequential(OrderedDict([
      ("defc7", nn.Linear(in_features=1000, out_features=4096, bias=True)),
      ("relu_defc7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("defc6", nn.Linear(in_features=4096, out_features=4096, bias=True)),
      ("relu_defc6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("defc5", nn.Linear(in_features=4096, out_features=4096, bias=True)),
      ("relu_defc5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("reshape", View((-1, 256, 4, 4))),
      ("deconv5", nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ("relu_deconv5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("conv5_1", nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
      ("relu_conv5_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("deconv4", nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ("relu_deconv4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("conv4_1", nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
      ("relu_conv4_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("deconv3", nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ("relu_deconv3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("conv3_1", nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
      ("relu_conv3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ("relu_deconv2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("deconv1", nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ("relu_deconv1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
      ("deconv0", nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
      ]))
            self.codelen = self.G[0].in_features
        elif name == "pool5":
            self.G = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ]))
            self.codelen = self.G[0].in_channels
        # load pre-trained weight from online or local folders
        if pretrained:
            if load_urls:
                SDnew = load_statedict_from_online(name)
            else:
                savepath = {"fc6": join(netsdir, "upconvGAN_%s.pt"%name),
                            "fc7": join(netsdir, "upconvGAN_%s.pt"%name),
                            "fc8": join(netsdir, "upconvGAN_%s.pt"%name),
                            "pool5": join(netsdir, "upconvGAN_%s.pt"%name)}
                SD = torch.load(savepath[name])
                SDnew = OrderedDict()
                for name, W in SD.items():  # discard this inconsistency
                    name = name.replace(".1.", ".")
                    SDnew[name] = W
            self.G.load_state_dict(SDnew)
        self.G.requires_grad_(False)

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)[:, [2, 1, 0], :, :]
        return torch.clamp(raw + RGB_mean.to(raw.device), 0, 255.0) / 255.0 * scale
