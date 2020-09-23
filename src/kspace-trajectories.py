import numpy as np

from helpers import get_fourier_transform


def spiral_samples_trajectory(width=512,
                              height=512,
                              starting_angle=0,
                              n_turns=10,
                              r=np.linspace(0, 1, 1000000)):

    t = np.linspace(0, 1, len(r))

    for curr_angle in range(starting_angle + 1):
        x = np.cos(2 * np.pi * n_turns * t + curr_angle) * r
        y = np.sin(2 * np.pi * n_turns * t + curr_angle) * r

    # 0 - 511
    x = (x/2 + 0.5) * (height - 1)
    y = (y/2 + 0.5) * (width - 1)

    i = np.round(width - y).astype(int)
    j = np.round(x).astype(int)
    spiral_matrix = np.zeros((width, height))

    for k in range(len(i)):
        try:
            spiral_matrix[i[k], j[k]] = 1
        except Exception as e:
            print(e)

    spiral_matrix = np.fft.ifftshift(spiral_matrix)
    spiral_matrix = np.reshape(spiral_matrix, [width * height, 1])
    samples_rows = np.nonzero(spiral_matrix)[0]
    samples_rows = np.sort(samples_rows)
    spiral_matrix = np.reshape(spiral_matrix, [width, height])

    return samples_rows, i, j, spiral_matrix


def kspace_measurements_spiral(img, samples_rows):
    fft_img = get_fourier_transform(img)

    stacked_fft_img = np.reshape(
        fft_img, [fft_img.shape[0] * fft_img.shape[1], 1])

    fft_img_spiral = np.array(
        [p[0] for idx, p in enumerate(stacked_fft_img) if idx in samples_rows])

    return fft_img_spiral


def minimum_energy_reconstruction(measurements, width, height, samples_rows):
    stacked_img = np.zeros((width*height,), dtype=np.complex_)

    stacked_img[samples_rows] = measurements
    sparse_img = np.reshape(stacked_img, [width, height])
    reconstructed_img = np.fft.ifft2(sparse_img)

    return reconstructed_img
