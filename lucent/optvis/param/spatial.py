import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_image(shape, sd=None):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor

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

def fft_image(shape, sd=None, decay_power=1, decorrelate=True):
    batch, ch, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, ch) + freqs.shape + (2,) # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None,None,...,None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h,w))
        image = image[:batch, :ch, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image /= magic
        return image
    return [spectrum_real_imag_t], inner
