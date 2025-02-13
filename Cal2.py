import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm


def entagl(ebs, lam, omega, aCoeff, time):

    beta = np.sqrt((np.power(lam, 2) / np.power(omega, 2)) - (np.power(ebs, 2) / (4 * np.power(lam, 2))))
    R = np.sqrt(4 * np.power(lam, 2) * np.power(beta, 2) + np.power(ebs, 2))
    neta = R / ebs

    sinTheta = (2 * lam * beta) / R
    cosTheta = ebs / R
    sinHalf = np.square((1 - 1 / neta) / 2)
    cosHalf = np.square((1 + 1 / neta) / 2)

    const = (np.power(lam, 2) * np.power(cosTheta, 2)) / (2 * np.power(R, 2))

    unitaryMatrix = np.zeros((4, 4), dtype=complex)

    unitaryMatrix[0][0] = np.power(sinHalf, 2)
    unitaryMatrix[0][1] = -sinTheta / np.sqrt(2)
    unitaryMatrix[0][2] = np.power(cosHalf, 2)
    unitaryMatrix[1][0] = -sinTheta / np.sqrt(2)
    unitaryMatrix[1][1] = cosTheta
    unitaryMatrix[1][2] = sinTheta / np.sqrt(2)
    unitaryMatrix[2][0] = np.power(cosHalf, 2)
    unitaryMatrix[2][1] = sinTheta / np.sqrt(2)
    unitaryMatrix[2][2] = np.power(sinHalf, 2)

    matrixOne = np.zeros((4, 4), dtype=complex)
    matrixTwo = np.zeros((4, 4), dtype=complex)
    matrixThree = np.zeros((4, 4), dtype=complex)

    matrixOne[0][0] = 1
    matrixTwo[0][1] = 1
    matrixTwo[1][0] = 1
    matrixThree[1][1] = 1

    matrixTwo = -((aCoeff * np.sqrt(1 - np.power(aCoeff, 2)) * lam * cosTheta * (np.exp(1j * omega * time) + np.exp(-1j * omega * time)))/(np.sqrt(2) * R)) * matrixTwo

    matrixThree = const * matrixThree

    Matrix = (1 / (1 + const)) * (matrixOne + matrixTwo + matrixThree)

    reducedMatrix = unitaryMatrix @ Matrix @ unitaryMatrix

    prdMatrix = np.zeros((2, 2), dtype=complex)

    prdMatrix[0][0] = reducedMatrix[0][0] + reducedMatrix[1][1]
    prdMatrix[0][1] = reducedMatrix[0][2] + reducedMatrix[1][3]
    prdMatrix[1][0] = reducedMatrix[2][0] + reducedMatrix[3][1]
    prdMatrix[1][1] = reducedMatrix[2][2] + reducedMatrix[3][3]

    newMatrix = -prdMatrix @ logm(prdMatrix)

    val = np.trace(np.abs(newMatrix))

    return val


def varyingEntropyWithTime(ebs, lam, omega, aCoeff, timeStep, timeEnd):
    trace_of_entropy_matrix = []
    time_vector = []

    for t in range(0, int(timeEnd / timeStep) + 1):
        current_time = t * timeStep
        time_vector.append(current_time)

        val = entagl(ebs, lam, omega, aCoeff, current_time)
        trace_of_entropy_matrix.append(val)

    plt.plot(time_vector, trace_of_entropy_matrix)
    plt.xlabel('Time')
    plt.ylabel('Trace of the entropy matrix')
    plt.ylim([0, 1])
    plt.show()

    return trace_of_entropy_matrix


varyingEntropyWithTime(2000, 30, 0.05, 0.5, 0.1, 10)


