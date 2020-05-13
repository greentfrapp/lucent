import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_image(size):
    tensor = (torch.randn(1, 3, size, size) * 0.01).to(device).requires_grad_(True)
    return [tensor], lambda: torch.sigmoid(tensor)

# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]

def fft_image(size, decay_power=1, decorrelate=True):
    freqs = rfft2d_freqs(size, size)
    batch = 1
    ch = 3
    w = h = size
    init_val_size = (batch, ch) + freqs.shape + (2,)

    spectrum_real_imag_t = (torch.randn(*init_val_size) * 0.01).to(device).requires_grad_(True)
    
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None,None,...,None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h,w))
        image = image[:batch, :ch, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image /= magic
        if decorrelate:
            image_flat = image.permute(0, 2, 3, 1).view(-1, 3)
            image_flat = torch.matmul(image_flat, torch.tensor(color_correlation_normalized.T).to(device))
            image = image_flat.view(*image.permute(0, 2, 3, 1).shape)
            image = image.permute(0, 3, 1, 2)
        return torch.sigmoid(image)
    return [spectrum_real_imag_t], inner
