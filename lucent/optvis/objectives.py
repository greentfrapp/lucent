import torch
from decorator import decorator
from lucent.optvis.objectives_util import _make_arg_str, _extract_act_pos

class Objective(object):

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, T):
        return self.objective_func(T)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda T: other + self(T)
            name = self.name
            description = self.description
        else:
            objective_func = lambda T: self(T) + other(T)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

def wrap_objective(require_format=None, handle_batch=False):
    @decorator
    def inner(f, *args, **kwds):
        objective_func = f(*args, **kwds)
        objective_name = f.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str

        return Objective(lambda T: objective_func(T),
                         objective_name, description)
    return inner

class ModuleHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output
    def close(self):
        self.hook.remove()

@wrap_objective()
def neuron(layer, n_channel, x=None, y=None):
    """Visualize a single neuron of a single channel.
    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.
    Odd width & height:               Even width & height:
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+
    """
    def inner(T):
        extracted_layer = T(layer)
        extracted_layer = _extract_act_pos(extracted_layer, x, y)
        return -extracted_layer[:, n_channel].mean()
    return inner

@wrap_objective()
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
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    elif isinstance(obj, str):
        layer, n = obj.split(":")
        layer, n = layer.strip(), int(n)
        return channel(layer, n)
