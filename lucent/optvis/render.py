from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image

from lucent.optvis import objectives, transform
from lucent.misc.io import show


def render_vis(model, objective_f, param_f, optimizer,
               transforms=None, thresholds=(512,), verbose=False,
               show_image=True, save_image=False, image_name=None, show_inline=False):

    if transforms is None:
        transforms = [transform.jitter(8)]
    transform_f = transform.compose(transforms)
    hook = hook_model(model, param_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(param_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []

    for i in tqdm(range(max(thresholds)+1)):
        optimizer.zero_grad()
        model(transform_f(param_f()))
        loss = objective_f(hook)
        loss.backward()
        optimizer.step()
        if i in thresholds:
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
            images.append(tensor_to_img_array(param_f()))

    if save_image:
        export(param_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(param_f()).astype(np.float32) / 255)
    elif show_image:
        view(param_f())
    return images


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
        self.module = None
        self.features = None
    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output
    def close(self):
        self.hook.remove()

def hook_model(model, t_image):
    features = OrderedDict()
    for name, layer in OrderedDict(model.named_children()).items():
        features[name] = ModuleHook(layer)
    def hook(layer):
        if layer == "input":
            return t_image()
        if layer == "labels":
            return list(features.values())[-1].features
        return features[layer].features
    return hook
