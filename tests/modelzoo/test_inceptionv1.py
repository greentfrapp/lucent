import pytest

import torch
from lucent.modelzoo import inceptionv1

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
    layer_names = list(dict(model.named_children()).keys())
    for layer_name in important_layer_names:
        assert layer_name in layer_names
