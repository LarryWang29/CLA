import numpy as np
import cla_utils
from numpy import random
A100 = np.loadtxt('AA100.dat')
A100 = np.reshape(A100, (100, 100))

b = random.randn(100)
print(cla_utils.GMRES(A100, b, 1000, 0.001, None, True, False)) # 48 iterations

B100 = np.loadtxt('BB100.dat')
B100 = np.reshape(B100, (100, 100))
print(cla_utils.GMRES(B100, b, 1000, 0.001, None, True, False)) # 5 iterations

C100 = np.loadtxt('CC100.dat')
C100 = np.reshape(C100, (100, 100))
print(cla_utils.GMRES(C100, b, 1000, 0.001, None, True, False)) # 10 iterations

print(np.linalg.matrix_rank(C100))