import torch
import torch.nn.functional as F
import numpy as np


def jitter(d):
    """
    Given a (N, H, W, C) tensor,
    randomly crops to (N, H-d, W-d, C)
    and pads back to (N, H, W, C)
    """
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)
    def inner(t_image):
        size = t_image.shape[2]
        x1 = np.random.choice(d)
        x2 = x1 + size - d
        y1 = np.random.choice(d)
        y2 = y1 + size - d
        return F.pad(t_image[:, :, y1:y2, x1:x2], [np.ceil(d/2).astype(int)]*4)
    return inner

def compose(transforms):
    if transforms is None:
        return lambda x: x
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x
    return inner
