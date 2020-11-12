import numpy as np

from numpy.fft import fft2, fftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


def calculate_snr(signal, reconstructed_signal):
    """
    Takes two signals of same dimension and calculate the error between
    Returns:
        signal to error ratio, signal to error ratio in db
    """
    # energia: somatorio do modulo ao quadrado
    signal_energy = np.sum(np.abs(signal)**2)
    error_energy = np.sum(np.abs(signal - reconstructed_signal)**2)
    signal_error_ratio = signal_energy / error_energy
    #  decibel
    ser_db = 10 * np.log10(signal_error_ratio)

    return ser_db


def calculate_metrics(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2)
    img_ssim = ssim(img1, img2)
    snr = calculate_snr(img1, img2)
    mse = mean_squared_error(img1, img2)

    print('============================')
    print(f'PSNR: {psnr}')
    print(f'SSIM: {img_ssim}')
    print(f'SNR: {snr}')
    print(f'MSE: {mse}')
    print('============================')

    return psnr, img_ssim, snr, mse


def zero_fill(b, samples_rows, rows, cols):
    zero_filled = np.zeros((rows * cols), dtype="complex128")
    zero_filled[samples_rows] = b
    zero_filled = np.reshape(zero_filled, (rows, cols))
    zero_filled = np.fft.ifft2(zero_filled)
    return zero_filled


# Deprecated
def get_fourier_transform(img, fshift=False, use_absolute=False):
    fourier_transf = fft2(img)

    if fshift:
        fourier_transf = fftshift(fourier_transf)
    if use_absolute:
        fourier_transf = np.log(np.abs(fourier_transf))

    return fourier_transf
