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

import pytest

import torch
from lucent.modelzoo import inceptionv1, util


important_layer_names = [
    "mixed3a",
    "mixed3b",
    "mixed4a",
    "mixed4b",
    "mixed4c",
    "mixed4d",
    "mixed4e",
    "mixed5a",
    "mixed5b",
]


def test_inceptionv1_graph_import():
    model = inceptionv1()
    layer_names = util.get_model_layers(model)
    for layer_name in important_layer_names:
        assert layer_name in layer_names

def test_inceptionv1_import_layer_repr():
    model = inceptionv1()
    layer_names = util.get_model_layers(model, getLayerRepr=True)
    for layer_name in important_layer_names:
        assert layer_names[layer_name] == 'CatLayer()'