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
    initial_state[0] = 1

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


# print(time_eval(entagl(2, 0, 0), 1, 150))

# varying_Jhat([1.0], 1.3, 1.1, 0.01, 3, 150)

varying_timesteps(1, 1, 1.5, [0.1, 0.7, 1.0], 50, 150)

# Ht, Hs, Hb, Hsb = entagl(0.0, 0.0, 0.0, 150)

# time_eval(Ht, 10)

# varying_n_value(1, 1, 1.5, [50, 150], 0.05, 10)

# my code

