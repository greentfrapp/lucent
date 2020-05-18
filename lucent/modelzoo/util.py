
"""Utility functions for modelzoo models."""

def get_model_layers(model):
    layers = []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                layers.append("_".join(prefix+[name]))
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers
