# modelzoo

This contains the InceptionV1 model corresponding to that used in Lucid examples, as well as acts as a thin wrapper around `torchvision` models.

## InceptionV1

Just run the following to load the InceptionV1 model in `eval` mode. This automatically downloads the model to your local `torch` cache directory, just like other `torchvision` models.

```python
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1(pretrained=True)
model.to(device).eval()
```

Credits to [ProGamerGov](https://github.com/ProGamerGov/) for converting the InceptionV1 Tensorflow model to PyTorch!

See the original repository [here](https://github.com/ProGamerGov/pytorch-old-tensorflow-models) and code for helper layers [here](https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/helper_layers.py).

## TorchVision Models

Lucent also works off the shelf with `torchvision.models`.

```python
import torch
from torchvision.models import resnet50
from lucent.optvis import render, param, transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True)
model.to(device).eval()

obj = "layer2:9" # a ResNet50 layer and channel
render.render_vis(model, obj)
```

`lucent.modelzoo` also acts as a thin wrapper for `torchvision.models`, so these two lines are equivalent.

```python
# These lines are equivalent
from torchvision.models import resnet50
from lucent.modelzoo import resnet50
```
