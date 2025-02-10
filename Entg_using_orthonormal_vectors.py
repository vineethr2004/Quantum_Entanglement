from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from matplotlib.lines import Line2D
import os

n = 25

Initial_state = np.zeros((4 * n, 1), dtype=complex)
Initial_state[n] = 1
Initial_state[2 * n] = 0


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def entagl(Jhat, ebs, lam):
    Z = np.zeros((n, n))
    O = np.eye(n)
    k = ebs * O
    J = Jhat * O

    Hsystem = np.block([[2 * k, Z, Z, J], [Z, k, J, Z], [Z, J, k, Z], [J, Z, Z, Z]])
    Hb = np.zeros((n, n))

    for i in range(0, n):
        Hb[i][i] = 1 * (2 * (i + 1) - 1) / 2

    Hbath = np.block([[Hb, Z, Z, Z],
                      [Z, Hb, Z, Z],
                      [Z, Z, Hb, Z],
                      [Z, Z, Z, Hb]])

    x = np.zeros((n, n))

    for i in range(1, n):
        x[i - 1][i] = np.sqrt(i)
        x[i][i - 1] = np.sqrt(i)

    Hsystem_and_bath = lam * np.block([[Z, x, x, Z],
                                       [x, Z, Z, x],
                                       [x, Z, Z, x],
                                       [Z, x, x, Z]])
    # print(f"Hsystem =  {Hsystem}")
    # print(f"Hbath = {Hbath}")
    # print(f"Hsystem_bath = {Hsystem_and_bath}")
    Htotal = Hsystem + Hbath + Hsystem_and_bath

    eigen_values, eigen_vectors = np.linalg.eig(Htotal)
    return eigen_values, gram_schmidt_columns(eigen_vectors)


def coeff_of_eigen_vectors(eigen_values, eigen_vectors, time):
    coeffs = np.zeros((4 * n, 1), dtype=complex)

    # print(f"shape of eigen values: {eigen_values.shape}")
    # print(f"shape of eigen vectors: {eigen_vectors.shape}")

    for i in range(0, 4 * n):
        initial_coeff = np.dot(np.conj(eigen_vectors[:, i]), Initial_state)
        # print(np.dot(eigen_vectors[:, 0], eigen_vectors[:, i]))
        # print(np.linalg.norm(eigen_vectors[:, i]))
        # print(eigen_values[i])
        coeffs[i] = initial_coeff * np.exp(-1j * eigen_values[i] * time)

    return coeffs


def psi_t(eigen_vectors, coeffs_t):
    psi_at_time_t = np.zeros((4 * n, 1), dtype=complex)

    for i in range(0, 4 * n):
        psi_at_time_t = psi_at_time_t + coeffs_t[i] * eigen_vectors[:, i].reshape((4 * n, 1))

    return psi_at_time_t


def density_matrix_construction_at_time_t(psi_at_time_t):
    d_matrix_at_time_t = psi_at_time_t @ np.conj(np.transpose(psi_at_time_t))
    # print(np.trace(d_matrix_at_time_t))
    return d_matrix_at_time_t


def partial_trace_over_bath_at_time_t(d_matrix_at_time_t):
    projections = []
    Z = np.zeros((n, n))
    I = np.eye(n)

    pt_rd = np.zeros((4, 4), dtype=complex)

    p1 = np.block([[I, Z, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, Z, Z]])

    projections.append(p1)

    p2 = np.block([[Z, Z, Z, Z],
                   [Z, I, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, Z, Z]])

    projections.append(p2)

    p3 = np.block([[Z, Z, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, I, Z],
                   [Z, Z, Z, Z]])

    projections.append(p3)

    p4 = np.block([[Z, Z, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, Z, Z],
                   [Z, Z, Z, I]])

    projections.append(p4)

    for i in range(0, 4):
        for j in range(0, 4):
            pt_rd[i][j] = np.trace(projections[i] @ d_matrix_at_time_t @ projections[j])

    # print(np.trace(pt_rd))
    return pt_rd


def reduced_density_matrix_over_1_qubit(rd_matrix_):
    partial_reduced_density_matrix_over_single_qubit = np.zeros((2, 2), dtype=complex)

    partial_reduced_density_matrix_over_single_qubit[0][0] = rd_matrix_[0][0] + rd_matrix_[1][1]
    partial_reduced_density_matrix_over_single_qubit[0][1] = rd_matrix_[0][2] + rd_matrix_[1][3]
    partial_reduced_density_matrix_over_single_qubit[1][0] = rd_matrix_[2][0] + rd_matrix_[3][1]
    partial_reduced_density_matrix_over_single_qubit[1][1] = rd_matrix_[2][2] + rd_matrix_[3][3]

    return partial_reduced_density_matrix_over_single_qubit


def varying_trace_of_entropy_with_time(eigen_values, eigen_vectors, time_step, end_time):
    trace_of_entropy_matrix = []
    time_vector = []

    for t in range(0, int(end_time / time_step) + 1):
        current_time = t * time_step
        time_vector.append(current_time)

        coefficients = coeff_of_eigen_vectors(eigen_values, eigen_vectors, current_time)

        psi_at_time_t = psi_t(eigen_vectors, coefficients)

        d_matrix_cal = density_matrix_construction_at_time_t(psi_at_time_t)

        rd_matrix_bath = partial_trace_over_bath_at_time_t(d_matrix_cal)
        # print(np.abs(np.trace(rd_matrix_bath @ rd_matrix_bath)))

        prd_matrix_qubit_one = reduced_density_matrix_over_1_qubit(rd_matrix_bath)
        print(np.trace(np.abs(prd_matrix_qubit_one @ prd_matrix_qubit_one)))
        new_matrix = -prd_matrix_qubit_one @ logm(prd_matrix_qubit_one)
        val = np.trace(new_matrix)

        trace_of_entropy_matrix.append(np.round(abs(val), 6))

    plt.plot(time_vector, trace_of_entropy_matrix)
    plt.xlabel('Time')
    plt.ylabel('Trace of the entropy matrix')
    plt.ylim([0, 1])
    plt.show()

    return trace_of_entropy_matrix


def varying_e_values(J, ebs_vector, lam, time_step, end_time):
    legend_handles = []
    trace_matrix = []
    for ebs in ebs_vector:
        eigen_values, eigen_vectors = entagl(J, ebs, lam)
        trace_of_entropy_matrix = []
        time_vector = []

        for t in range(0, int(end_time / time_step) + 1):
            current_time = t * time_step
            time_vector.append(current_time)

            coefficients = coeff_of_eigen_vectors(eigen_values, eigen_vectors, current_time)

            psi_at_time_t = psi_t(eigen_vectors, coefficients)

            d_matrix_cal = density_matrix_construction_at_time_t(psi_at_time_t)

            rd_matrix_bath = partial_trace_over_bath_at_time_t(d_matrix_cal)

            prd_matrix_qubit_one = reduced_density_matrix_over_1_qubit(rd_matrix_bath)

            new_matrix = -prd_matrix_qubit_one @ logm(prd_matrix_qubit_one)
            val = np.trace(new_matrix)

            trace_of_entropy_matrix.append(np.round(abs(val), 6))

        trace_matrix.append(trace_of_entropy_matrix)
        line, = plt.plot(time_vector, trace_of_entropy_matrix)
        legend_handles.append(Line2D([0], [0], color=line.get_color(), lw=3))

    # plt.xlabel('Time')
    # plt.ylabel('Entropy')
    # plt.title('Entropy varying with time')
    plt.ylim(None, 1)
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, handlelength=3, handletextpad=1,
               labelspacing=1.5)

    plt.savefig(r'D:\Physics\Graphs\varying Entropy\initial_state_not_entangled\oscillations\fig.png')
    plt.show()
    return np.array(trace_matrix), time_step


def varying_lam_values(J, ebs, lam_vector, time_step, end_time):
    legend_handles = []
    trace_matrix = []
    for lam in lam_vector:
        eigen_values, eigen_vectors = entagl(J, ebs, lam)
        trace_of_entropy_matrix = []
        time_vector = []

        for t in range(0, int(end_time / time_step) + 1):
            current_time = t * time_step
            time_vector.append(current_time)

            coefficients = coeff_of_eigen_vectors(eigen_values, eigen_vectors, current_time)

            psi_at_time_t = psi_t(eigen_vectors, coefficients)

            d_matrix_cal = density_matrix_construction_at_time_t(psi_at_time_t)

            rd_matrix_bath = partial_trace_over_bath_at_time_t(d_matrix_cal)

            prd_matrix_qubit_one = reduced_density_matrix_over_1_qubit(rd_matrix_bath)

            new_matrix = -prd_matrix_qubit_one @ logm(prd_matrix_qubit_one)
            val = np.trace(new_matrix)

            trace_of_entropy_matrix.append(np.round(abs(val), 6))

        trace_matrix.append(trace_of_entropy_matrix)
        line, = plt.plot(time_vector, trace_of_entropy_matrix)
        legend_handles.append(Line2D([0], [0], color=line.get_color(), lw=3))

    # plt.xlabel('Time')
    # plt.ylabel('Entropy')
    # plt.title('Entropy varying with time')
    plt.ylim(None, 1)
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, handlelength=3, handletextpad=1,
               labelspacing=1.5)
    plt.savefig(r'D:\Physics\Graphs\varying Entropy\initial_state_not_entangled\convergence\w=100_times_lam^2\fig.png')
    plt.show()
    return np.array(trace_matrix), time_step


eigenvalues, eigenvectors = entagl(0, 0, 1.0)

coeffs_at_t = coeff_of_eigen_vectors(eigenvalues, eigenvectors, 0.0)

# for i in range(1, 100):
#     print(eigenvalues[i], coeffs_at_t[i])
#
# psit = psi_t(eigenvectors, coeffs_at_t)

# for i in range(0, 400):
#     if np.round(abs(psit[i]) ** 2, 4) > 0:
#         print(i, abs(psit[i]) ** 2)

# dmatrix = density_matrix_construction_at_time_t(psit)
# print(np.round(dmatrix, 5))

# rdmatrix = partial_trace_over_bath_at_time_t(dmatrix)
# print(rdmatrix)

# prd_matrix = reduced_density_matrix_over_1_qubit(rdmatrix)
# print(prd_matrix)

varying_trace_of_entropy_with_time(eigenvalues, eigenvectors, 2, 10)

# mat, st = varying_e_values(0, [0.0], 1.0, 0.2, 13)
# mat, st = varying_e_values(0, [0.0], 0.3, 0.2, 60)

# varying_trace_of_entropy_with_time(eigenvalues, eigenvectors, 1, 20)
# mat, st = varying_lam_values(0.0, 0.0, [0.1], 1, 100)
# mat, st = varying_lam_values(0.0, 0.0, [0.03], 0.1, 10)

# folder_path = r"D:\Physics\Graphs\varying Entropy\initial_state_not_entangled\convergence"
# mat, st = varying_lam_values_parallel(0.0, 0.0, [0.19, 0.51], 2, 180, folder_path)
# file_path = os.path.join(folder_path, 'mat100.npy')
# np.save(file_path, mat)

# mat, st = varying_lam_values(0, 0.0, [0.2, 0.3, 0.4], 0.2, 6)
# freq_domain(mat, st)

# varying_e_values(0, [0.1], 0.2, 0.05, 60)
# varying_e_values(0, [0.0, 0.2], 0.1, 0.5, 4)



