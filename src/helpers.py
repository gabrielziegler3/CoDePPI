import numpy as np
from numpy.fft import fft2, fftshift


def signal_to_error(ground_truth: np.array, reconstructed_signal: np.array) -> tuple:
    """
    Takes two signals of same dimension and calculate the error between
    Returns:
        signal to error ratio, signal to error ratio in db
    """
    err = reconstructed_signal - ground_truth
    # energy: sum of the square of the modulus
    signal_energy = np.sum(np.abs(ground_truth) ** 2)
    error_energy = np.sum(np.abs(err) ** 2)
    signal_error_ratio = signal_energy / error_energy
    # calculate in decibels
    ser_db = 10 * np.log10(signal_error_ratio)

    return signal_error_ratio, ser_db


def get_fourier_transform(img, fshift=False, use_absolute=False):
    fourier_transf = fft2(img)

    if fshift:
        fourier_transf = fftshift(fourier_transf)
    if use_absolute:
        fourier_transf = np.log(np.abs(fourier_transf))

    return fourier_transf
