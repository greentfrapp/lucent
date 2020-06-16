"""
Copyright 2020 Binxu Wang

This method is inspired by the work
    Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2017). Deep Image Prior.  https://doi.org/10.1007/s11263-020-01303-4
So the CNN model is copied from their repo, and I wrap it up to match the Lucent interface

Jun.4th 2020
"""
import torch
import numpy as np
import sys
sys.path.append(r"E:\DL_Projects\Vision\deep-image-prior")
from models import skip
input_depth = 32
pad = 'reflection'
net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 4, 4, 4, 4, 4],
                           filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3],
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)



def DeepImagePrior(imsize=256, input_depth=32, batch=1, sd=1):
    """Sample a noise tensor with standard deviation sd, and pass it through an un-trained UNet architecture to form an image """
    net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[0, 4, 4, 4, 4, 4],
               filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
               upsample_mode='bilinear', downsample_mode='avg',
               need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
    net.requires_grad_(True)
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)
    noise = (torch.randn((batch, input_depth, imsize, imsize)) * sd).to("cuda").requires_grad_(False)
    imagef = lambda:  net(noise)
    return list(net.parameters()), imagef


def DeepImagePrior_rand(imsize=256, input_depth=32, batch=1, sd=1):
    """Same as Deep Image Prior, but the noise is not fixed, but resampled each time like a GAN"""
    net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[0, 4, 4, 4, 4, 4],
               filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
               upsample_mode='bilinear', downsample_mode='avg',
               need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
    net.requires_grad_(True)
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)
    imagef = lambda:  net((torch.randn((batch, input_depth, imsize, imsize)) * sd).to("cuda").requires_grad_(False))
    return list(net.parameters()), imagef