# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from lucent.misc.io import show
from lucent.optvis import objectives, param, transform
from lucent.optvis.hooks import ModelHook

ObjectiveT = Union[str, Callable[[torch.Tensor], torch.Tensor]]
ParamT = Callable[[], Tuple[Sequence[torch.Tensor], Callable[[], torch.Tensor]]]
OptimizerT = Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer]


def render_vis(
    model: nn.Module,
    objective_f: ObjectiveT,
    param_f: Optional[ParamT] = None,
    optimizer_f: Optional[OptimizerT] = None,
    transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    thresholds: Sequence[int] = (512,),
    verbose: bool = False,
    preprocess: bool = True,
    progress: bool = True,
    show_image: bool = True,
    save_image: bool = False,
    image_name: Optional[str] = None,
    show_inline: bool = False,
    fixed_image_size: Optional[int] = None,
    iteration_callback: Optional[
        Callable[[ModelHook, torch.Tensor, torch.Tensor], None]
    ] = None,
):
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer_f is None:
        optimizer_f = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer_f(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    objective_f = objectives.as_objective(objective_f)

    with ModelHook(model, image_f) as hook:
        if verbose:
            model(transform_f(image_f()))
            print("Initial loss: {:.3f}".format(objective_f(hook)))

        images = []
        try:
            for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                loss = objective_f(hook)
                loss.backward()
                optimizer.step()

                if i in thresholds:
                    image = tensor_to_img_array(image_f())
                    if verbose:
                        print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                        if show_inline:
                            show(image)
                    images.append(image)
                if iteration_callback:
                    iteration_callback(hook, loss, image_f())
        except KeyboardInterrupt:
            print("Interrupted optimization at step {:d}.".format(i))
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
            images.append(tensor_to_img_array(image_f()))

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return images


def tensor_to_img_array(tensor: torch.Tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor: torch.Tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor: torch.Tensor, image_name: Optional[str] = None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)
