import numpy as np
from numpy.fft import fft2, fftshift


def get_fourier_transform(img, fshift=False, use_absolute=False):
    fourier_transf = fft2(img)

    if fshift:
        fourier_transf = fftshift(fourier_transf)
    if use_absolute:
        fourier_transf = np.log(np.abs(fourier_transf))

    return fourier_transf
