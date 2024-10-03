
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from matplotlib.lines import Line2D
import os

n = 2
n_sq = np.square(n)
Initial_state = np.zeros((4 * n_sq, 1), dtype=complex)
Initial_state[n_sq] = 1


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def entagl(Jhat, ebs, lam, omega_1, omega_2):
    Z_sq = np.zeros((n_sq, n_sq))
    I_sq = np.eye(n_sq)
    Z = np.zeros((n, n))
    I = np.eye(n)

    k = ebs * I_sq
    J = Jhat * I_sq

    Hsystem = np.block([[2 * k, Z_sq, Z_sq, J], [Z_sq, k, J, Z_sq], [Z_sq, J, k, Z_sq], [J, Z_sq, Z_sq, Z_sq]])

    Hb_1 = np.zeros((4 * n_sq, 4 * n_sq))

    it = 0

    for r in range(0, 4):
        for i in range(0, n):
            val = omega_1 * (2 * i + 1) / 2
            for t in range(0, n):
                Hb_1[it][it] = val
                it = it + 1

    it = 0

    Hb_2 = np.zeros((4 * n_sq, 4 * n_sq))

    for r in range(0, 4):
        for t in range(0, n):
            for i in range(0, n):
                Hb_2[it][it] = omega_2 * (2 * i + 1) / 2
                it = it + 1

    Hbath = Hb_1 + Hb_2

    x = np.zeros((n, n))

    for i in range(1, n):
        x[i - 1][i] = np.sqrt(i)
        x[i][i - 1] = np.sqrt(i)

    Y_1 = np.kron(x, I)

    Hsb_1 = lam * np.block([[Z_sq, Y_1, Y_1, Z_sq],
                            [Y_1, Z_sq, Z_sq, Y_1],
                            [Y_1, Z_sq, Z_sq, Y_1],
                            [Z_sq, Y_1, Y_1, Z_sq]])

    Y_2 = np.kron(I, x)

    Hsb_2 = lam * np.block([[Y_2, Z_sq, Z_sq, Y_2],
                           [Z_sq, Y_2, Y_2, Z_sq],
                           [Z_sq, Y_2, Y_2, Z_sq],
                           [Y_2, Z_sq, Z_sq, Y_2]])

    Hsystem_and_bath = Hsb_1 + Hsb_2

    # print(f"Hsystem =  {Hsystem}")
    # print(f"Hbath = {Hbath}")
    # print(f"Hsystem_bath = {Hsystem_and_bath}")

    Htotal = Hsystem + Hbath + Hsystem_and_bath

    eigen_values, eigen_vectors = np.linalg.eig(Htotal)

    return eigen_values, gram_schmidt_columns(eigen_vectors)


entagl(1, 2, 0, 0, 0)