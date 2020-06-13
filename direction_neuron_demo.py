import torch
from lucent.optvis import render, param, objectives

# from lucid.misc.io import show
# import lucid.optvis.objectives as objectives
# import lucid.optvis.param as param
# import lucid.optvis.render as render
# import lucid.optvis.transform as transform

from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True)
model.to(device).eval()

param_f = lambda: param.image(224, fft=True, decorrelate=True)
opt = lambda params: torch.optim.Adam(params, 5e-2)
direction = torch.rand(256) * 1000
obj = objectives.direction_neuron(layer_name='layer3', direction=direction)

render.render_vis(model, obj, param_f, opt)

# nn.CosineSimilarity(dim=1)(torch.rand(256).reshape((1, -1, 1, 1)),
#                            model(layer)[0:1, :, 4:5, 4:5]).mean()