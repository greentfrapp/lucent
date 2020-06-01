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

"""Utility functions"""

from __future__ import absolute_import, division, print_function

import torch
import random
from collections import OrderedDict


def set_seed(seed):
    # Set global seeds to for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

def lucent_layernames(net, prefix=[]):
    """ Return the layername and str representation of the layer """
    layernames = OrderedDict()
    def hook_layernames(net, prefix=[]):
        """Recursive function to return the layer name"""
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                layernames["_".join(prefix+[name])] = layer.__repr__()
                hook_layernames(layer, prefix=prefix+[name])
    hook_layernames(net, prefix)
    return layernames