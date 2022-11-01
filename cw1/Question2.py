import cla_utils
import math
import numpy as np

A1 = np.loadtxt('A1.dat')
Q, R = cla_utils.householder_qr(A1)
print(Q, R)

Q = np.array([[4, 1, 5, 7], [0, 2, 2, 4], [2, 0, 2, 1], [1, 1, 2, 7]])
print(cla_utils.householder(np.array([[1, 2, -1], [0, 15, 18], [-2, -4, -4], [-2, -4, -10]])))