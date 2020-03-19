import numpy as np
from numpy.fft import fft2, fftshift


def get_fourier_transform(img, fshift=False, use_absolute=True):
    fourier_transf = fft2(img)

    if fshift:
        fourier_transf = fftshift(fourier_transf)
    magnitude_spectrum = fourier_transf
    if use_absolute:
        magnitude_spectrum = np.log(np.abs(fourier_transf))
    return magnitude_spectrum
