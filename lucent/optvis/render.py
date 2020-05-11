# import torch
# from torchvision import models
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image

from lucent.optvis import objectives


def render_vis(model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=True, relu_gradient_override=True, use_fixed_seed=False):
    
    transform_f = make_transform_f(transforms)
    t_image = param_f
    T = hook_model(model, t_image)
    objective_f = objectives.as_objective(objective_f)

    layer = SaveFeatures(dict(model.named_children())["inception4b"])

    for i in tqdm(range(max(thresholds)+1)):
        optimizer.zero_grad()
        model(t_image())
        loss = objective_f(T)
        loss.backward()
        optimizer.step()

    view(t_image())

def view(tensor):
    image = tensor.cpu().detach().numpy()[0]
    image = np.transpose(image, [1, 2, 0])
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).show()

def make_transform_f(transforms):
    # Just dummy for now
    def dummy(tensor): return tensor
    return dummy

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output
    def close(self):
        self.hook.remove()

def hook_model(model, t_image):
    # if t_image_raw is None:
    #     t_image_raw = t_image
    features = {}
    for name, layer in dict(model.named_children()).items():
        features[name] = SaveFeatures(layer)
    def T(layer):
        if layer == "input": return t_image()
        if layer == "labels": return features["fc"].features
        return features[layer].features
    return T

