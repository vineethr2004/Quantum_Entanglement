from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from matplotlib.lines import Line2D
import os

n = 20

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
        Hb[i][i] = 100 * 0.0009 * (2 * (i + 1) - 1) / 2

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

    Htotal = Hsystem + Hbath + Hsystem_and_bath

    eigen_values, eigen_vectors = np.linalg.eig(Htotal)

    return eigen_values, gram_schmidt_columns(eigen_vectors)


def coeff_of_eigen_vectors(eigen_values, eigen_vectors, time):
    coeffs = np.zeros((4 * n, 1), dtype=complex)
    for i in range(0, 4 * n):
        initial_coeff = np.dot(np.conj(eigen_vectors[:, i]), Initial_state)
        coeffs[i] = initial_coeff * np.exp(-1j * eigen_values[i] * time)

    return coeffs


def psi_t(eigen_vectors, coeffs_t):
    psi_at_time_t = np.zeros((4 * n, 1), dtype=complex)

    for i in range(0, 4 * n):
        psi_at_time_t = psi_at_time_t + coeffs_t[i] * eigen_vectors[:, i].reshape((4 * n, 1))

    return psi_at_time_t


def density_matrix_construction_at_time_t(psi_at_time_t):
    return psi_at_time_t @ np.conj(np.transpose(psi_at_time_t))


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

    return pt_rd


def reduced_density_matrix_over_1_qubit(rd_matrix_):
    partial_reduced_density_matrix_over_single_qubit = np.zeros((2, 2), dtype=complex)

    partial_reduced_density_matrix_over_single_qubit[0][0] = rd_matrix_[0][0] + rd_matrix_[1][1]
    partial_reduced_density_matrix_over_single_qubit[0][1] = rd_matrix_[0][2] + rd_matrix_[1][3]
    partial_reduced_density_matrix_over_single_qubit[1][0] = rd_matrix_[2][0] + rd_matrix_[3][1]
    partial_reduced_density_matrix_over_single_qubit[1][1] = rd_matrix_[2][2] + rd_matrix_[3][3]

    return partial_reduced_density_matrix_over_single_qubit


def process_lambda(J, ebs, lam, time_step, end_time, folder_path):
    legend_handles = []
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

    # Plot and save graph for the current lambda
    plt.figure()

    line, = plt.plot(time_vector, trace_of_entropy_matrix)
    legend_handles.append(Line2D([0], [0], color=line.get_color(), lw=3))
    # plt.plot(time_vector, trace_of_entropy_matrix)
    # plt.ylim(None, 1)
    plt.ylim(None, 1)
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, handlelength=3, handletextpad=1,
               labelspacing=1.5)
    # plt.title(f'Entropy varying with time for λ={lam}')
    # plt.xlabel('Time')
    # plt.ylabel('Entropy')

    # Save the figure
    file_name = f'lambda_{lam}.png'
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.close()

    return trace_of_entropy_matrix


def varying_lam_values_parallel(J, ebs, lam_vector, time_step, end_time, folder_path):
    trace_matrix = []

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_lambda, J, ebs, lam, time_step, end_time, folder_path)
                   for lam in lam_vector]

        # Collect results
        for future in futures:
            trace_matrix.append(future.result())

    return np.array(trace_matrix), time_step


if __name__ == '__main__':
    folder_path = r"D:\Physics\Graphs\varying Entropy\initial_state_not_entangled\convergence"
    mat, st = varying_lam_values_parallel(0.0, 0.0, [0.03], 20, 600, folder_path)
    file_path = os.path.join(folder_path, 'mat100.npy')
    np.save(file_path, mat)
