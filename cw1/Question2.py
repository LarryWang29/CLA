import cla_utils
import math
import numpy as np
from numpy import random
A1 = np.loadtxt('A1.dat')

m = random.randint(3, 193)
n = random.randint(3, m)
m = 200
n = 59
random.seed(4732*m + 1238*n)
A = random.randn(m, n)
Q, R = cla_utils.householder_qr(A)
x = random.randint(1,m)
zero_id = set(random.randint(1, m, x))
R = [([0] * n if i in zero_id else R[i]) for i in range(m)]
B = Q @ R
print(B)
results = []
for i in range(20):
    s = 10 ** (-i+3)
    C = B.copy()
    R_hat = cla_utils.householder(C, kmax=None, swap=True, reduced_tol=s)[1]
    results.append(np.shape(R_hat)[0])

A2 = A1.copy()
A1_rank = np.shape(cla_utils.householder(A2, kmax=None, swap=True, reduced_tol=1.0e-1000000)[1])[0]
print(A1_rank)
print(A2)