from decorator import decorator
from lucent.optvis.objectives_util import _make_arg_str, _extract_act_pos

class Objective():

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

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

def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner

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
    def inner(model):
        extracted_layer = model(layer)
        extracted_layer = _extract_act_pos(extracted_layer, x, y)
        return -extracted_layer[:, n_channel].mean()
    return inner

@wrap_objective()
def channel(layer, n_channel):
    """Visualize a single channel"""
    def inner(model):
        return -model(layer)[:, n_channel].mean()
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
    if isinstance(obj, str):
        layer, chn = obj.split(":")
        layer, chn = layer.strip(), int(chn)
        return channel(layer, chn)
