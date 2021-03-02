import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from numpy.fft import fft2, fftshift
from torch import Tensor


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
    # decibel
    ser_db = 10 * np.log10(signal_error_ratio)

    return ser_db


def calculate_metrics_1d(img1, img2, verbose=True):
    snr = calculate_snr(img1, img2)
    mse = mean_squared_error(img1, img2)

    if verbose:
        print('============================')
        print(f'SNR: {snr}')
        print(f'MSE: {mse}')
        print('============================')

    return snr, mse


def calculate_metrics(img1, img2, verbose=True):
    if isinstance(img1, Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, Tensor):
        img2 = img2.cpu().detach().numpy()

    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)

    psnr = peak_signal_noise_ratio(img1, img2)
    img_ssim = ssim(img1, img2, data_range=img2.max() - img2.min())
    snr = calculate_snr(img1, img2)
    mse = mean_squared_error(img1, img2)

    if verbose:
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


def get_proportion(f_sampled, u_sampled):
    prop = np.array(u_sampled).flatten(
    ).shape[0] / np.array(f_sampled).flatten().shape[0]
    return prop


# Deprecated
def get_fourier_transform(img, fshift=False, use_absolute=False):
    fourier_transf = fft2(img)

    if fshift:
        fourier_transf = fftshift(fourier_transf)
    if use_absolute:
        fourier_transf = np.log(np.abs(fourier_transf))

    return fourier_transf
