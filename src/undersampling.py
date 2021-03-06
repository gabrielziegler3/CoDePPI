import numpy as np
import scipy

from helpers import get_fourier_transform


def spiral_samples_trajectory(width=512,
                              height=512,
                              starting_angle=0,
                              n_turns=10,
                              r=np.linspace(0, 1, 1000000)):

    t = np.linspace(0, 1, len(r))

    x = np.cos(2 * np.pi * n_turns * t + starting_angle) * r
    y = np.sin(2 * np.pi * n_turns * t + starting_angle) * r

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


# Extracted from
# https://github.com/mathialo/master_code/blob/3ab986938817f0ee327d4b79b62756abb5cefeda/mastl_package/mastl/patterns.py
def radial_sampling(len_x, len_y, line_num, dilations=1, close=True):
    """
    Creates a line sampling pattern where a specified number of lines is evenly
    distributed on angles between 0 and 2pi, with lines stroking from the center
    to the edges of the frame.
    Args:
        len_x (int):            Size of output mask in x direction (width)
        len_y (int):            Size of output mask in y direction (height)
        line_num (int):         Number of lines to add
        dilations (int):        Number of morphological dilations to perform
                                after the lines have been sampled. This affects
                                the line width.
        close (bool):           Whether to perform morphological closing or not
                                on the sampling pattern before applying dilations
    Returns:
        np.ndarray: A boolean numpy array (mask) depicting sampling pattern.
    """
    mask = np.zeros([len_y, len_x], dtype=np.bool)

    center = len_y // 2, len_x // 2

    thetas = np.arange(line_num) / line_num * 2 * np.pi

    # Sample a greater amount of points than what is strictly needed in order to
    # capture all the discretized points.
    point_num = int(np.floor(np.sqrt(len_x ** 2 + len_y ** 2)))

    # Points along radius
    points = np.linspace(0, 1, point_num)

    for line in range(line_num):
        theta = thetas[line]

        # Find length of the line from the center to the edge of the frame along
        # given angle.
        if np.pi / 4 < theta < 3 * np.pi / 4 or 5 * np.pi / 4 < theta < 7 * np.pi / 4:
            r = np.abs(len_y // 2 / np.sin(theta))
        else:
            r = np.abs(len_x // 2 / np.cos(theta))

        # Sample line
        for r_ in r * points:
            # Sample points along radius
            x = int(np.cos(theta) * r_) + center[1]
            y = int(np.sin(theta) * r_) + center[0]

            # Truncate x and y to avoid out-of-bounds errors at the edges
            if y >= len_y:
                y = len_y - 1
            if x >= len_x:
                x = len_x - 1
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            mask[y, x] = 1

    # Perform morphological actions to better pattern
    if close:
        mask = scipy.ndimage.morphology.binary_closing(mask)

    for i in range(dilations):
        mask = scipy.ndimage.morphology.binary_dilation(mask)

    return mask
