import torch


class ModuleHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output
    def close(self):
        self.hook.remove()

def channel(layer, n_channel, batch=None):
    """Visualize a single channel"""
    def inner(T):
        return -T(layer)[:, n_channel].mean()
    return inner

def as_objective(obj):
    """Convert obj into Objective class.
    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.
    Args:
    obj: string or Objective.
    Returns:
    Objective
    """
    # if isinstance(obj, Objective):
    #     return obj
    if callable(obj):
        return obj
    elif isinstance(obj, str):
        layer, n = obj.split(":")
        layer, n = layer.strip(), int(n)
        return channel(layer, n)
