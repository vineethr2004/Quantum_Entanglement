import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm

n = 150

Initial_state = np.zeros((4 * n, 1), dtype=complex)
Initial_state[n] = 1/np.sqrt(2)
Initial_state[2 * n] = 1/np.sqrt(2)


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
        Hb[i][i] = (2 * (i + 1) - 1) / 2

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
    # print(f"shape of eigen values: {eigen_values.shape}")
    # print(f"shape of eigen vectors: {eigen_vectors.shape}")
    for i in range(0, 4 * n):
        initial_coeff = np.dot(np.conj(eigen_vectors[:, i]), Initial_state)
        # print(f"Shape of initial_coeff: {initial_coeff.shape}")
        # print(f"Shape of eigen_values[i]: {eigen_values[i].shape}")
        coeffs[i] = initial_coeff * np.exp(-1j * eigen_values[i] * time)

    return np.round(coeffs, 5)


def density_matrix_construction_at_time_t(coeffs_at_time_t):
    d_matrix = np.zeros((4 * n, 4 * n), dtype=complex)

    for i in range(0, 4 * n):
        for j in range(0, 4 * n):
            d_matrix[i][j] = coeffs_at_time_t[i, 0] * np.conj(coeffs_at_time_t[j, 0])

    return d_matrix


def partial_trace_over_bath_at_time_t(d_matrix_at_time_t, eigen_vectors):
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

    d_matrix_initial_basis = np.transpose(np.conj(eigen_vectors)) @ d_matrix_at_time_t @ eigen_vectors

    for i in range(0, 4):
        for j in range(0, 4):
            pt_rd[i][j] = np.trace(projections[i] @ d_matrix_initial_basis @ projections[j])

    return pt_rd


def reduced_density_matrix_over_1_qubit(rd_matrix_):
    prd_matrix = np.zeros((2, 2), dtype=complex)

    prd_matrix[0][0] = rd_matrix_[0][0] + rd_matrix_[1][1]
    prd_matrix[0][1] = rd_matrix_[0][2] + rd_matrix_[1][3]
    prd_matrix[1][0] = rd_matrix_[2][0] + rd_matrix_[3][1]
    prd_matrix[1][1] = rd_matrix_[2][2] + rd_matrix_[3][3]

    return prd_matrix


def varying_trace_of_entropy_with_time(eigen_values, eigen_vectors, time_step, end_time):
    trace_of_entropy_matrix = []
    time_vector = []

    for t in range(0, int(end_time / time_step) + 1):
        current_time = t * time_step
        time_vector.append(current_time)

        matrix = reduced_density_matrix_over_1_qubit(
            partial_trace_over_bath_at_time_t(density_matrix_construction_at_time_t(
                coeff_of_eigen_vectors(eigen_values, eigen_vectors, current_time)), eigen_vectors))

        new_matrix = -matrix @ logm(matrix)
        val = np.trace(new_matrix)
        trace_of_entropy_matrix.append(np.round(abs(val), 4))

    plt.plot(time_vector, trace_of_entropy_matrix)
    plt.xlabel('Time')
    plt.ylabel('Trace of the entropy matrix')
    plt.show()

    return trace_of_entropy_matrix


eigenvalues, eigenvectors = entagl(1, 0, 0.1)

# print(eigenvalues[0], eigenvalues[2])
#
# print(np.dot(np.conjugate(eigenvectors[0]), eigenvectors[2]))


# print(eigenvalues == np.real(eigenvalues))

# print(np.round(np.transpose(np.conj(eigenvectors)) @ eigenvectors, 5) == np.eye(600))

#
# print(eigenvalues.shape, eigenvectors.shape)

# print((np.conj(eigenvectors[:, 1]) @ Initial_state).size)

coeffs = coeff_of_eigen_vectors(eigenvalues, eigenvectors, 0)

# print(coeffs)

d_matrix = density_matrix_construction_at_time_t(coeffs)

# print(d_matrix.shape)

rdmatrix = partial_trace_over_bath_at_time_t(d_matrix, eigenvectors)

print(rdmatrix)

# prd_matrix = reduced_density_matrix_over_1_qubit(rd_matrix)

# print(prd_matrix)

# varying_trace_of_entropy_with_time(eigenvalues, eigenvectors, 0.2, 2)

