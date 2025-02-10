import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from matplotlib.lines import Line2D
import os


def calculate(E, t_step, end_time):

    time_vector = []
    prd_vector = []
    for t in range(0, int(end_time / t_step) + 1):
        current_time = t * t_step
        val = (E * current_time) / 2
        mat = np.block([[(np.cos(val))**2, 0], [0, (np.sin(val))**2]])
        time_vector.append(current_time)
        prd = -mat @ logm(mat)
        prd_vector.append(np.round(np.abs(np.trace(prd)), 5))

    plt.plot(time_vector, prd_vector)
    plt.show()
    plt.ylim(None, 1)


calculate(2, 0.01, 10)