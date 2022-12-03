import numpy as np
import cla_utils
from numpy import random


m = 20
random.seed(1302*m)
A = random.randn(m, m) + 1j*random.randn(m, m)
A = 0.5*(A + np.conj(A).T)
e, _ = np.linalg.eig(A)
x0 = random.randn(m) + 0.j
mu = e[m//2] + random.randn() + 1j*random.randn()

B = A - mu * np.eye(m)
w1 = np.linalg.solve(B, x0) 
w = cla_utils.householder_solve(B, x0)
print(w1)
print(w)