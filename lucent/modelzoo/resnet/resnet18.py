import torch

from .resnet import build_resnet
from .resnet import BasicBlock
from torchvision.models.resnet import ResNet18_Weights

def resnet18():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.verify("IMAGENET1K_V1")
    model = build_resnet(BasicBlock, [2, 2, 2, 2], weights=weights, progress=True, use_linear_modules_only=False, skip_batchnorm = True).to(device).eval()

    return model