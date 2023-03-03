from types import TracebackType
from typing import Callable, Dict, Optional, Sequence, Type

import torch
from torch import nn


class ModuleHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self._features: Dict[str, torch.Tensor] = dict()

    @property
    def features(self):
        keys = list(sorted(self._features.keys()))
        if len(keys) == 0:
            return None
        elif len(keys) == 1:
            return self._features[keys[0]]
        else:
            return torch.nn.parallel.gather([self._features[k] for k in keys], keys[0])

    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        device = output.device
        self._features[str(device)] = output

    def close(self):
        self.hook.remove()


class ModelHook:
    def __init__(
        self,
        model: nn.Module,
        image_f: Optional[Callable[[], torch.Tensor]] = None,
        layer_names: Optional[Sequence[int]] = None,
    ):
        self.model = model
        self.image_f = image_f
        self.features: Dict[str, ModuleHook] = {}
        self.layer_names = layer_names

    def __enter__(self):
        # recursive hooking function
        def hook_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                layers = list(net._modules.items())
                for i, (name, layer) in enumerate(layers):
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue

                    if self.layer_names is not None and i < len(layers) - 1:
                        # only save activations for chosen layers
                        if name not in self.layer_names:
                            continue

                    self.features["_".join(prefix + [name])] = ModuleHook(layer)
                    hook_layers(layer, prefix=prefix + [name])

        if isinstance(self.model, torch.nn.DataParallel):
            hook_layers(self.model.module)
        else:
            hook_layers(self.model)

        def hook(layer):
            if layer == "input":
                out = self.image_f()
            elif layer == "labels":
                out = list(self.features.values())[-1].features
            else:
                assert (
                    layer in self.features
                ), f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
                out = self.features[layer].features
            assert (
                out is not None
            ), "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
            return out

        return hook

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        for k in self.features.copy():
            self.features[k].close()
            del self.features[k]
