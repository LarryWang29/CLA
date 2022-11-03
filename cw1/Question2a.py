import cla_utils
import numpy as np
A1 = np.loadtxt('A1.dat')

Q, R = cla_utils.householder_qr(A1)
b = np.linalg.matrix_rank(R)
print(b)
