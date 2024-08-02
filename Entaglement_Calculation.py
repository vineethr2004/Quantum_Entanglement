import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def entagl(Jhat, ebs, lam, n):
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

    return Htotal


def time_eval(Htotal, time, n):
    initial_state = np.zeros((4 * n, 1))
    initial_state[150] = 1 / np.sqrt(2)
    initial_state[300] = 1 / np.sqrt(2)

    time_evolution_operator = expm(-1j * Htotal * time)

    # print(np.all(time_evolution_operator == np.diag(np.diagonal(time_evolution_operator))))

    state_at_time_t = time_evolution_operator @ initial_state
    # print(state_at_time_t)

    value = (np.conj(state_at_time_t)).T @ initial_state

    value = abs(value)
    # print(value)

    return round(value.item(), 5)


def varying_Jhat(Jhat_vector, ebs, lam, time_step, final_time, n):
    for Jhat in Jhat_vector:
        H_total = entagl(Jhat, ebs, lam, n)
        val_vector = []
        time_vector = []

        for t in range(0, int(final_time / time_step) + 1):
            current_time = t * time_step
            time_vector.append(current_time)
            value = time_eval(H_total, current_time, n)
            val_vector.append(value)

        plt.plot(time_vector, val_vector, label=f'Jhat={Jhat}')

    plt.xlabel('Time')
    plt.ylabel('Inner Product')
    plt.title('Inner Product varying with time and Jhat')
    plt.legend()
    plt.show()
    return plt


def varying_timesteps(Jhat, ebs, lam, time_step_vector, final_time, n):
    Htotal = entagl(Jhat, ebs, lam, n)

    for time_step in time_step_vector:
        val_vector = []
        time_vector = []
        for t in range(0, int(final_time / time_step) + 1):
            current_time = t * time_step
            time_vector.append(current_time)
            val = time_eval(Htotal, current_time, n)
            val_vector.append(val)

        plt.plot(time_vector, val_vector, label=f'time_step = {time_step}')

    plt.xlabel('Time')
    plt.ylabel('Inner product value')
    plt.title('Inner Product varying with time')
    plt.legend()
    plt.show()
    return plt


def varying_n_value(Jhat, ebs, lam, n_vector, time_step, final_time):
    for n in n_vector:
        Htotal = entagl(Jhat, ebs, lam, n)
        val_vector = []
        time_vector = []

        for t in range(0, int(final_time / time_step) + 1):
            current_time = t * time_step
            time_vector.append(current_time)
            val = time_eval(Htotal, current_time, n)
            val_vector.append(val)

        plt.plot(time_vector, val_vector, label=f'n_value = {n}')

    plt.xlabel('Time')
    plt.ylabel('Inner product value')
    plt.title('Inner Product varying with time')
    plt.legend()
    plt.show()
    return plt


def density_matrix(Htotal, time, n):
    initial_state = np.zeros((4 * n, 1))
    initial_state[150] = 1 / np.sqrt(2)
    initial_state[300] = 1 / np.sqrt(2)

    time_evolution_operator = expm(-1j * Htotal * time)
    state_at_time_t = time_evolution_operator @ initial_state

    d_matrix = state_at_time_t @ (np.conj(state_at_time_t)).T

    return d_matrix


def reduced_density_matrix_of_system(d_matrix, n):
    red_matrix = np.zeros((4, 4), dtype=complex)

    for i in range(0, 4):
        for j in range(0, 4):
            for t in range(0, n):
                red_matrix[i][j] = red_matrix[i][j] + d_matrix[i * n + t][j * n + t]

    return red_matrix


def partial_transpose_of_reduced_density_matrix(red_matrix):
    ptrd_matrix = red_matrix
    ptrd_matrix[0][1], ptrd_matrix[1][0] = ptrd_matrix[1][0], ptrd_matrix[0][1]
    ptrd_matrix[0][3], ptrd_matrix[1][2] = ptrd_matrix[1][2], ptrd_matrix[0][3]
    ptrd_matrix[2][1], ptrd_matrix[3][0] = ptrd_matrix[3][0], ptrd_matrix[2][1]
    ptrd_matrix[2][3], ptrd_matrix[3][2] = ptrd_matrix[3][2], ptrd_matrix[2][3]

    return ptrd_matrix


def time_evolution_of_trace(Htotal, time_step, end_time, n):
    trace_of_d = []
    trace_of_d_square = []
    time_vector = []
    for t in range(0, int(end_time / time_step) + 1):
        current_time = t * time_step
        time_vector.append(current_time)
        matrix = reduced_density_matrix_of_system(density_matrix(Htotal, current_time, n), n)
        val = np.trace(matrix)
        tmp = np.trace(matrix @ matrix)
        trace_of_d.append(round(abs(val), 5))
        trace_of_d_square.append(round(abs(tmp), 5))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot trace of d
    axs[0].plot(time_vector, trace_of_d, label='Trace of d')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Trace of d')
    axs[0].set_title('Trace of d vs Time')
    axs[0].legend()
    axs[0].grid(True)

    # Plot trace of d^2
    axs[1].plot(time_vector, trace_of_d_square, label='Trace of d^2', color='orange')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Trace of d^2')
    axs[1].set_title('Trace of d^2 vs Time')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def time_evolution_of_eigen_values(Htotal, time_step, end_time, n):
    eigen_values = []
    time_vector = []
    for t in range(0, int(end_time / time_step)):
        current_time = t * time_step
        time_vector.append(current_time)
        Arr = partial_transpose_of_reduced_density_matrix(
            reduced_density_matrix_of_system(density_matrix(Htotal, current_time, n), n))
        eigen_value = min(np.linalg.eigvals(Arr))
        # print(eigen_value)
        if eigen_value > 0:
            eigen_value = 0
        eigen_values.append(eigen_value)

    # for value in eigen_values:
    #     plt.scatter(time_vector, value)
    plt.scatter(time_vector, eigen_values)
    plt.xlabel('Time')
    plt.ylabel('Minimum Eigen Value')
    plt.title('Minimum Eigen Value varying with time')
    # plt.legend()
    plt.show()


time_evolution_of_eigen_values(entagl(1, 0, 0.8, 150), 0.1, 20, 150)

# red_matr = reduced_density_matrix_of_system(density_matrix(entagl(1, 1, 0, 150), 10, 150), 150)
# print(red_matr)

# Arr = partial_transpose_of_reduced_density_matrix(red_matr)
# print(Arr)
# print(min(np.linalg.eigvals(Arr)))

# print(min(np.linalg.eigvals(Arr)))

# time_evolution_of_trace(entagl(1, 1, 0, 150), 0.1, 10, 150)

# rd_m = reduced_density_matrix_of_system(density_matrix(1, 1, 1, 10, 150), 150)
# print(rd_m)
#
# print(partial_transpose_of_reduced_density_matrix(rd_m))

# print(time_eval(entagl(2, 0, 0), 1, 150))

# varying_Jhat([1.0], 1.3, 1.1, 0.01, 3, 150)

# varying_timesteps(1, 1, 1, [0.1], 10, 150)

# Ht, Hs, Hb, Hsb = entagl(0.0, 0.0, 0.0, 150)

# time_eval(Ht, 10)

# varying_n_value(1.5, 1.5, 1.5, [50, 100, 150], 0.2, 50)

# my code
