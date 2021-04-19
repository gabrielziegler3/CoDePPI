import numpy as np
import h5py
import matplotlib.pylab as plt
import fastmri
import torch
import xml.etree.ElementTree as etree

from helpers import calculate_metrics, get_proportion
from typing import Tuple
from fastmri.data import transforms as T
from fastmri.data.mri_data import et_query
from fastmri.data.subsample import create_mask_for_mask_type


class AdaptedFastMRI:
    def __init__(self):
        """
        shape = (int, int)
        kspace = Tensor
        masked_kspace = Tensor
        cropped_kspace = Tensor
        masked_cropped_kspace = Tensor
        gt_image = np.array
        reconstruction = np.array
        mask = Tensor
        samples_rows = np.array
        b = np.array
        cropped_shape = (int, int)
        """
        self.shape = (0, 0)
        self.kspace = None
        self.masked_kspace = None
        self.cropped_kspace = None
        self.masked_cropped_kspace = None
        self.gt_image = None
        self.reconstruction = None
        self.mask = None
        self.samples_rows = None
        self.b = None
        self.cropped_shape = (0, 0)


    def __call__(self, filename: str):
        """
        Load file and retrieve information
        """
        # take one slice only
        n_slice = 20
        with h5py.File(filename, mode='r') as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            # extract target image width, height from ismrmrd header
            enc = ["encoding", "encodedSpace", "matrixSize"]
            self.shape= (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
            )

            volume_kspace = hf['kspace'][()]
            slice_kspace = volume_kspace[n_slice]
            self.kspace = T.to_tensor(slice_kspace)
            self.gt_image = hf['reconstruction_esc'][()][n_slice]
            self.cropped_shape = self.gt_image.shape

        self.cropped_kspace = T.to_tensor(np.fft.fftshift(np.fft.fft2(self.gt_image)))
        self.masked_cropped_kspace, self.mask = self.apply_mask(self.cropped_kspace, "equispaced")
        self.extract_measurements(self.masked_cropped_kspace)

        print(f"Using {100 * get_proportion(self.gt_image.flatten(), self.b)}% k-space points")


    def apply_mask(self, kspace, mask_type_str: str):
        """
        Works for "equispaced" and "random" masks
        """
        if kspace.ndim < 3:
            kspace = T.to_tensor(kspace)

        mask_func = create_mask_for_mask_type(
            mask_type_str=mask_type_str, center_fractions=[0.08], accelerations=[4])
        masked_kspace, mask = T.apply_mask(kspace, mask_func)

        return masked_kspace, mask


    def zero_fill(self, masked_kspace: torch.Tensor):
        zero_filled = fastmri.ifft2c(masked_kspace)
        zero_filled = fastmri.fftshift(zero_filled)
        zero_filled = fastmri.complex_abs(zero_filled)
        print("Zero filled reconstruction:")
        calculate_metrics(self.gt_image, zero_filled)

        return zero_filled


    def get_2dmask(self):
        return torch.ones(self.cropped_shape) * self.mask.squeeze() + 0.0


    def extract_measurements(self, masked_kspace):
        """
        extract b and samples_rows
        """
        assert masked_kspace.shape[0] == masked_kspace.shape[1], "Matrix must be squared for our broken L1 minimization"

        self.samples_rows = np.nonzero(
            T.tensor_to_complex_np(masked_kspace).flatten())[0]
        self.b = T.tensor_to_complex_np(masked_kspace).flatten()[self.samples_rows]


    # def crop_center(self):
    #     """
    #     Crop real image 2d numpy
    #     """
    #     y, x = self.shape
    #     new_x, new_y = self.cropped_shape
    #     startx = x//2 - (new_x // 2)
    #     starty = y//2 - (new_y // 2)
    #     print(self.gt_image.shape)
    #     cropped_img = self.gt_image[starty:starty+new_y, startx:startx+new_x]
    #     print(cropped_img.shape)
    #     cropped_tensor = T.to_tensor(cropped_img)
    #     self.cropped_kspace = fastmri.fft2c(cropped_tensor)
