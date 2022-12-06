import numpy as np
import cla_utils
from numpy import random
import matplotlib.pyplot as plt

def Mat_D(N, s):
    """
    Generates the matrix D for given values of N and s

    :param N: Dimension of vector u
    :param s: constant that governs the strength of correlation

    :return D: return the matrix D as defined in the question
    """
    N2 = N ** 2
    s2 = s ** 2
    D = np.eye(N2) * (1 + 4 * s2)
    sn1 = -s2 * np.ones(N2-1)
    D += np.diag(sn1, 1) + np.diag(sn1, -1)
    sn2 = -s2 * np.ones(N2-N)
    D += np.diag(sn2, N) + np.diag(sn2, -N)
    return D

def sim(N, s, option):
    N2 = N ** 2
    w = np.random.randn(N2) 
    D = Mat_D(N,s)
    if option=='Banded':
        u_hat = cla_utils.solve_LU(D, w, N+1, N+1)
    elif option=='Original':
        u_hat = cla_utils.solve_LU(D, w)
    u = np.reshape(u_hat, (N, N))
    c = plt.imshow(u, cmap='hot', vmin=-4, vmax=4)
    plt.colorbar(c)
    plt.show()

sim(50, 1, 'Banded')