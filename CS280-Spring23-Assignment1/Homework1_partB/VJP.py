import numpy as np


def VJP(Z, Y_bar):
    exp_Z = np.exp(Z)
    R = np.sum(exp_Z, axis=1)
    R_bar = -np.sum(Y_bar * exp_Z / (R ** 2)[:, None], axis=1)
    Z_bar = Y_bar * exp_Z / R[:, None] + R_bar[:, None] * exp_Z
    return Z_bar


def naive(Z, Y_bar):
    Z_bar = np.zeros_like(Z)
    exp_Z = np.exp(Z)
    R = np.sum(exp_Z, axis=1)
    for i in range(Z.shape[0]):
        r_bar = -np.sum(Y_bar[i] * exp_Z[i] / (R ** 2)[i])
        for j in range(Z.shape[1]):
            Z_bar[i, j] = Y_bar[i, j] * exp_Z[i, j] / R[i] + r_bar * exp_Z[i, j]
    return Z_bar


if __name__ == "__main__":
    Z = np.array([
        [0.7, 0.23, 3.45],
        [4, 5.1, 0.654],
    ])
    Y_bar = np.array([
        [0.1, 1.2, 1],
        [1, 1, 1.08],
    ])
    a = VJP(Z, Y_bar)
    b = naive(Z, Y_bar)
    print(a)
    print(b)
