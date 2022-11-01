import cla_utils
import math
import numpy as np
from numpy import random

A1 = np.loadtxt('A1.dat')
Q, R = cla_utils.householder_qr(A1)

m = random.randint(3, 193)
n = random.randint(3, m)
m = 200
n = 57
random.seed(4732*m + 1238*n)
A = random.randn(m, n)
Q, R = cla_utils.householder_qr(A)
x = random.randint(1,m)
zero_id = set(random.randint(1, m, x))
R = [([0] * n if i in zero_id else R[i]) for i in range(m)]
B = Q @ R
print(np.linalg.matrix_rank(B))