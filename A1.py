import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm

n = 150


def entagl(ebs, lam, omega):

    beta = np.sqrt((np.power(lam, 2) / np.power(omega, 2)) - (np.power(ebs, 2) / (4 * np.power(lam, 2))))
    R = np.sqrt(4 * np.power(lam, 2) * np.power(beta, 2) + np.power(ebs, 2))
    neta = R / ebs

    sinTheta = (2 * lam * beta) / R
    cosTheta = ebs / R
    sinHalf = np.square((1 - 1 / neta) / 2)
    cosHalf = np.square((1 + 1 / neta) / 2)

    tempSystemMatrix = np.zeros((4, 1), dtype=complex)
    tempBathMatrix = np.zeros((n, 1), dtype=complex)

    tempSystemMatrix[0] = np.power(sinHalf, 2) - sinTheta / np.sqrt(2)
    tempSystemMatrix[1] = -np.sqrt(2) * sinHalf * cosHalf + cosTheta
    tempSystemMatrix[2] = np.power(cosHalf, 2) + sinTheta / np.sqrt(2)
    tempBathMatrix[0] = 1

    firstTerm = (1 + (beta * lam * cosTheta) / (np.sqrt(2) * R)) * np.kron(tempSystemMatrix, tempBathMatrix)
    tempBathMatrix[0] = 0

    tempSystemMatrix[0] = np.power(sinHalf, 2) + sinTheta / np.sqrt(2)
    tempSystemMatrix[1] = -np.sqrt(2) * sinHalf * cosHalf - cosTheta
    tempSystemMatrix[2] = np.power(cosHalf, 2) - sinTheta / np.sqrt(2)
    tempBathMatrix[1] = 1

    secondTerm = (beta + (lam * cosTheta) / (np.sqrt(2) * R)) * np.kron(tempSystemMatrix, tempBathMatrix)
    tempBathMatrix[1] = 0

    tempSystemMatrix[0] = sinTheta / np.sqrt(2)
    tempSystemMatrix[1] = -cosTheta
    tempSystemMatrix[2] = -sinTheta / np.sqrt(2)
    tempBathMatrix[2] = 1

    thirdTerm = ((beta * np.sqrt(2) * lam * cosTheta) / R) * np.kron(tempSystemMatrix, tempBathMatrix)
    tempBathMatrix[2] = 0

    psi_0 = firstTerm + secondTerm + thirdTerm

    omega_0 = omega * np.sqrt(1 - 1 / np.power(neta, 2))
    Egs = -0.5 * ebs * (neta + 1 / neta) + 0.5 * omega_0

    print()
    return Egs, psi_0


def psi_t(Egs, psi_0, t):
    psi_at_time_t = np.exp(-1j * Egs * t) * psi_0
    return psi_at_time_t


def density_matrix_construction_at_time_t(psi_at_time_t):
    d_matrix_at_time_t = psi_at_time_t @ np.conj(np.transpose(psi_at_time_t))
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


def varying_trace_of_entropy_with_time(lam, ebs, omega, time_step, end_time):
    trace_of_entropy_matrix = []
    time_vector = []

    for t in range(0, int(end_time / time_step) + 1):
        current_time = t * time_step
        time_vector.append(current_time)

        Egs, psi_0 = entagl(lam, ebs, omega)
        # print(psi_0.shape)

        psi_at_time_t = psi_t(Egs, psi_0, current_time)
        # print(psi_at_time_t.shape)

        d_matrix_cal = density_matrix_construction_at_time_t(psi_at_time_t)
        # print(d_matrix_cal.shape)

        rd_matrix_bath = partial_trace_over_bath_at_time_t(d_matrix_cal)
        # print(rd_matrix_bath.shape)
        # print(np.abs(np.trace(rd_matrix_bath @ rd_matrix_bath)))

        prd_matrix_qubit_one = reduced_density_matrix_over_1_qubit(rd_matrix_bath)
        # print(np.trace(np.abs(prd_matrix_qubit_one @ prd_matrix_qubit_one)))

        new_matrix = -prd_matrix_qubit_one @ logm(prd_matrix_qubit_one)
        val = np.trace(new_matrix)
        print(val)
        trace_of_entropy_matrix.append(np.round(abs(val), 6))

    plt.plot(time_vector, trace_of_entropy_matrix)
    plt.xlabel('Time')
    plt.ylabel('Trace of the entropy matrix')
    plt.ylim([0, 1])
    plt.show()

    return trace_of_entropy_matrix


egs, initial_state = entagl(100, 10, 0.1)

# print(egs)
# print(np.abs(initial_state))

varying_trace_of_entropy_with_time(10, 1, 0.1, 5, 50)


