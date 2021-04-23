# ----| Imported libraries |-------------------------------------------------------------------{{{
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import List, Tuple
# ---}}}

# ----| gaussian_2d_domain |-------------------------------------------------------------------{{{
def gaussian_2d_domain(i, j, C, rows, columns):
    I = np.arange(0.0, rows, 1.0)
    J = np.arange(0.0, columns, 1.0)
    [J, I] = np.meshgrid(J, I)
    sigma1 = np.sqrt(C[0, 0])
    sigma2 = np.sqrt(C[1, 1])
    rho = C[0, 1] / (sigma1 * sigma2);
    k = 1.0 / (2.0 * np.pi * sigma1 * sigma2 * np.sqrt(1.0 - rho ** 2))
    L = -1.0 / (2.0 * (1.0 - rho ** 2))
    J = J - float(j)
    I = I - float(i)
    z = (J * J) / (sigma1 ** 2) + (I * I) / (sigma2 ** 2) - 2 * rho / (sigma1 * sigma2) * (J * I)
    G = k * np.exp(L * z)
    return G, I, J
# ---}}}

# ----| index2ij |-------------------------------------------------------------------{{{
def index2ij(index: int, rows: int, columns: int) -> Tuple:
    M = np.zeros(shape=(rows * columns,), dtype=np.uint32)
    M[index] = 1
    M = M.reshape((rows, columns))
    indices = np.nonzero(M)
    i = indices[0][0]
    j = indices[1][0]
    return i, j
# ---}}}


# ----| weights_from_prior |-------------------------------------------------------------------{{{
# def weights_from_prior(phis: List, rows: int, columns: int, taufactor=1e6, taumin=1, cov_matrix=np.array([[100, 0], [0, 100]])):
#     z_size = len(phis)
#     weights = np.zeros(shape=(rows, columns, z_size))
#     for idx, phi in enumerate(phis):
#         n = 0
#         for position in phi:
#             if isinstance(cov_matrix, list):
#                 C_ = cov_matrix[idx][n]
#             else:
#                 C_ = cov_matrix
#             i, j = index2ij(position, rows, columns)
#             prob, I, J = gaussian_2d_domain(i, j, C_, rows, columns)
#             prob = (prob * taufactor + taumin)
#             weights[..., idx] += prob
#             n += 1
#     return weights, I, J
# ---}}}


def weights_from_prior(phis: dict, rows: int, columns: int, cov_matrix: dict,
                       taufactor=1e6, taumin=1):
    # z_size = len(phis)
    # weights = np.zeros(shape=(rows, columns, z_size))
    weights = np.zeros(shape=(rows, columns))

    for k, v in phis.items():
        cov = cov_matrix[k]
        print(cov)
        for phi in v:
            for position in phi:
                i, j = index2ij(position, rows, columns)
                prob, I, J = gaussian_2d_domain(i, j, cov,
                                                rows, columns)
                prob = (prob * taufactor + taumin)
                weights += prob
    return weights, I, J


# ----| test1 |-------------------------------------------------------------------{{{
def test1():
    i = 128
    j = 128
    rows = 256
    columns = 256
    C = np.array([[100.0, 0.0], [0.0, 100.0]])
    t0 = time()
    for k in range(0, 1000):
        G1, _, _ = gaussian_2d_domain(i, j, C, rows, columns)
    t1 = time()
    print(t1 - t0)
    t0 = time()
    for k in range(0, 1000):
        G2, _, _ = gaussian_2d_domain_v2(i, j, C, rows, columns)
    t2 = time()
    print(t2 - t0)
# ---}}}

# ----| test2 |-------------------------------------------------------------------{{{
def test2():
    n1 = 128 * 256 + 128
    n2 = n1 + 40
    n3 = 128 * 346 + 128
    phis = [[n1, n2, n3], [n1, n2, n3], [n1, n2, n3]]
    rows = 256
    columns = 256
    C = np.array([[100, 0], [0, 100]])
    W, I, J = weights_from_prior(phis, rows, columns, taufactor=1e6, taumin=1, cov_matrix=C)
    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(J, I, W[:, :, 0], cmap='turbo', linewidth=0)
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('W')
    plt.show()
# ---}}}

# ----| test3 |-------------------------------------------------------------------{{{
def test3():
    n1 = 128 * 256 + 128
    n2 = n1 + 80
    n3 = 128 * 346 + 128
    phis = [[n1, n2, n3], [], []]
    rows = 1024
    columns = 1024
    C = [[np.array([[100, 0], [0, 100]]), np.array([[400, 0], [0, 400]]), np.array([[40, 0], [0, 40]])], [], []]
    W, I, J = weights_from_prior(phis, rows, columns, taufactor=1e6, taumin=1, cov_matrix=C)
    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(J, I, W[:, :, 0], cmap='turbo', linewidth=0)
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('W')
    plt.show()
# ---}}}


def visualize3d(weights: List, J: int, I: int, frame=0):
    """
    weights: weights matrix
    frame: which frame from the 3rd dimension to display
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(J, I, weights[..., frame], cmap='turbo', linewidth=0)
    ax.set_xlabel('J')
    ax.set_ylabel('I')
    ax.set_zlabel('W')
    plt.show()


# ----| main |-------------------------------------------------------------------{{{
if __name__ == '__main__':
    test3()
# ---}}}
