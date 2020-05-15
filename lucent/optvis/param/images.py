"""High-level wrapper for paramaterizing images."""

from lucent.optvis.param.spatial import pixel_image, fft_image
from lucent.optvis.param.color import to_valid_rgb


def image(w, h=None, sd=None, decorrelate=True,
          fft=True, alpha=False, channels=None):
    h = h or w
    ch = channels or (4 if alpha else 3)
    shape = [1, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
        # if alpha:
        #     def inner():
        #         a = tf.nn.sigmoid(t()[..., 3:])
        #         output = torch.cat([output, a], -1)
        #         return output
        #     return params, inner
    return params, output
