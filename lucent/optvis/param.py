import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_image(size):
	tensor = (torch.randn(1, 3, size, size) * 0.01).to(device).requires_grad_(True)
	return [tensor], lambda: torch.sigmoid(tensor)
