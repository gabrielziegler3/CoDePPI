import numpy as np
import scipy

from typing import Union, List, Any
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from torch import Tensor


def calculate_snr(signal: np.ndarray, reconstructed_signal: np.ndarray) -> float:
    """
    Takes two signals of same dimension and calculate the error between
    Returns:
        signal to error ratio, signal to error ratio in dB
    """
    # energy: sum of the module squared
    signal_energy = np.sum(np.abs(signal)**2)
    error_energy = np.sum(np.abs(signal - reconstructed_signal)**2)
    signal_error_ratio = signal_energy / error_energy
    ser_db = 10 * np.log10(signal_error_ratio)

    return ser_db


# Deprecated. Might have some weird behaviour
def calculate_metrics_1d(gt_img, recon_img, verbose=True):
    snr = calculate_snr(gt_img, recon_img)
    mse = mean_squared_error(gt_img, recon_img)

    if verbose:
        print('============================')
        print(f'SNR: {snr}')
        print(f'MSE: {mse}')
        print('============================')

    return snr, mse


def calculate_metrics(gt_img: np.ndarray, recon_img: np.ndarray, verbose=True) -> tuple:
    """
    Display PSNR, SSIM, SNR and MSE for reconstructed image
    against ground truth.

    Args:
        gt_img: groud truth image
        recon_img: reconstructed image
        verbose: wether to print the metrics or just return them
    Returns:
        tuple of metrics
    """
    assert gt_img.shape == recon_img.shape

    if isinstance(gt_img, Tensor):
        gt_img = gt_img.cpu().detach().numpy()
    if isinstance(recon_img, Tensor):
        recon_img = recon_img.cpu().detach().numpy()

    gt_img = np.array(gt_img, dtype=np.float64)
    recon_img = np.array(recon_img, dtype=np.float64)

    psnr = peak_signal_noise_ratio(
        gt_img, recon_img, data_range=recon_img.max() - recon_img.min()
    )
    img_ssim = ssim(gt_img, recon_img,
                    data_range=recon_img.max() - recon_img.min())
    snr = calculate_snr(gt_img, recon_img)
    mse = mean_squared_error(gt_img, recon_img)

    if verbose:
        print('============================')
        print(f'PSNR: {psnr}')
        print(f'SSIM: {img_ssim}')
        print(f'SNR: {snr}')
        print(f'MSE: {mse}')
        print('============================')

    return psnr, img_ssim, snr, mse


def zero_fill(b: np.ndarray, indices: Union[np.ndarray, List], rows: int, cols: int):
    """
    Args:
        b: 1d complex array of measurements
        indices: 1d array of indices of the collected measurements.
                 This is obtained from the undersampling trajectory used.
        rows: image to be reconstructed's row size
        cols: image to be reconstructed's column size
    Returns:
        2d complex zero-filled reconstruction.
    """
    zero_filled = np.zeros((rows * cols), dtype="complex128")
    zero_filled[indices] = b
    zero_filled = np.reshape(zero_filled, (rows, cols))
    zero_filled = np.fft.ifft2(zero_filled)
    return zero_filled


def get_proportion(f_sampled: np.ndarray, u_sampled: np.ndarray) -> float:
    """
    Returns undersampling proportion.
    """
    # must have same dimensions
    assert len(f_sampled.shape) == len(u_sampled.shape)

    prop = np.array(u_sampled).flatten(
    ).shape[0] / np.array(f_sampled).flatten().shape[0]
    return prop


def create_mask(indices: Union[np.ndarray, List], rows=256, cols=256, transpose=False) -> np.ndarray:
    """
    Create 2d mask from positive elements indices.
    """
    mask = np.zeros((rows * cols), dtype=int)
    mask[indices] = 1
    mask = np.reshape(mask, (rows, cols))
    if transpose:
        mask = mask.T
    return mask


def to_matlab(var: Any, var_name: str):
    """
    Save variable in Matlab format
    """
    var = {var_name: var}
    scipy.io.savemat(var_name + ".mat", var)


def from_matlab(filename: str) -> np.ndarray:
    """
    Load Matlab variable
    """
    return scipy.io.loadmat(filename)
