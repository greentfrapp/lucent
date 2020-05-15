# modelzoo

This contains the InceptionV1 model corresponding to that used in Lucid examples, as well as acts as a thin wrapper around `torchvision` models.

## InceptionV1

Credits to [ProGamerGov](https://github.com/ProGamerGov/) for converting the InceptionV1 Tensorflow model to PyTorch!

See the original repository [here](https://github.com/ProGamerGov/pytorch-old-tensorflow-models) and code for helper layers [here](https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/helper_layers.py).

### Instructions

Download the InceptionV1 model file [here](https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth).

Then run the following to load the model in `eval` mode.

```python
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1(pretrained=True, modelpath="inception5h.pth")
model.to(device).eval()
```

## Other Models

Lucent works off the shelf with `torchvisions.models`.

```python
import torch
from torchvision.models import resnet50
from lucent.optvis import render, param, transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True)
model.to(device).eval()

transforms = [
	transform.jitter(8),
	# Add ImageNet normalization for torchvision models
	# see https://pytorch.org/docs/stable/torchvision/models.html
	transform.normalize(),
]

params, image = param.image(224)
obj = "layer2:9"
optimizer = torch.optim.Adam(params, lr=5e-2)

render.render_vis(model, obj, image, optimizer, transforms=transforms)
```

`lucent.modelzoo` also acts as a thin wrapper for `torchvision.models`, so these two lines are equivalent.

```python
from torchvision.models import resnet50
from lucent.modelzoo import resnet50
```
