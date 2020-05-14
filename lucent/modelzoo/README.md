# modelzoo

Currently only has the InceptionV1 model corresponding to that used in Lucid examples.

Credits to [ProGamerGov](https://github.com/ProGamerGov/) for converting the Tensorflow model to PyTorch!

See the original repository [here](https://github.com/ProGamerGov/pytorch-old-tensorflow-models) and code for helper layers [here](https://github.com/ProGamerGov/neural-dream/blob/master/neural_dream/helper_layers.py).

## Instructions

Download the InceptionV1 model file [here](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip) and extract the `inception5h.pth` file.

Then run the following to load the model in `eval` mode.

```python
from lucent.modelzoo import InceptionV1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = InceptionV1()
model.to(device)
model.load_state_dict(torch.load("inception5h.pth")) # use appropriate path here
model.eval()
```
