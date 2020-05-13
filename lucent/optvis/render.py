import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image

from lucent.optvis import objectives, param, transform


def render_vis(model, objective_f, param_f, optimizer,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=False, relu_gradient_override=True, use_fixed_seed=False,
               save_image=False, image_name=None):
    
    transforms = transforms or [transform.jitter(8)]
    transform_f = transform.compose(transforms)
    T = hook_model(model, param_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(param_f()))
        print("Initial loss: {:.3f}".format(objective_f(T)))

    for i in tqdm(range(max(thresholds)+1)):
        optimizer.zero_grad()
        model(transform_f(param_f()))
        loss = objective_f(T)
        loss.backward()
        optimizer.step()
        if verbose and i in thresholds:
            print("Loss at step {}: {:.3f}".format(i, objective_f(T)))

    if save_image:
        export(param_f(), image_name)
    else:
        view(param_f())

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()[0]
    image = np.transpose(image, [1, 2, 0])
    if np.max(image) <= 1: # infer whether to scale
        image *= 255
    image = image.astype(np.uint8)
    return image

def view(tensor):
    image = tensor_to_img_array(tensor)
    Image.fromarray(image).show()

def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    Image.fromarray(image).save(image_name)

class ModuleHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output
    def close(self):
        self.hook.remove()

def hook_model(model, t_image):
    features = OrderedDict()
    for name, layer in OrderedDict(model.named_children()).items():
        features[name] = ModuleHook(layer)
    def T(layer):
        if layer == "input": return t_image()
        if layer == "labels": return list(features.values())[-1].features
        return features[layer].features
    return T

