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

"""Utility functions for modelzoo models."""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from typing import List, Optional, Union

import torch
from torch import nn


def get_model_layers(
    model: nn.Module, get_layer_representation: bool = False
) -> Union[OrderedDict[str, str], List[str]]:
    """
    Args:
        model: Model to get layer names for.
        get_layer_representation: Get layer representations and not just layer names.
    Returns:
        If get_layer_representation is True, return a OrderedDict of layer names, layer representation
        string pair. If it's False, just return a list of layer names
    """
    layer_name_representations = OrderedDict()

    def get_layers(module: nn.Module, prefix=[]):
        # Recursive function to get layers.

        if hasattr(module, "_modules"):
            for name, layer in module._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                layer_name_representations["_".join(prefix + [name])] = layer.__repr__()
                get_layers(layer, prefix=prefix + [name])

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    get_layers(model)

    if get_layer_representation:
        return layer_name_representations
    else:
        return list(layer_name_representations.keys())


def get_model_layer(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """Get a model's layer based on its internal name.

    Args:
        model: Module to extract layer from.
        layer_name: Name of the layer to extract.

    Returns:
        Extracted layer if the name is valid, None otherwise.
    """

    def get_layer(net, prefix=[]):
        # Recursive function to test layers.
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                current_layer_name = "_".join(prefix + [name])
                if current_layer_name == layer_name:
                    return layer
                inner = get_layer(layer, prefix=prefix + [name])
                if inner:
                    return inner
        return None

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    return get_layer(model)
