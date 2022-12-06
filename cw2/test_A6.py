import cla_utils
import numpy as np
from numpy import random

def An(n):
    A = -1 * np.tri(n) + 2 * np.eye(n)
    A[:,-1] = np.ones(n)
    cla_utils.LUP_inplace(A)
    L = np.eye(n)
    i1 = np.tril_indices(n, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    return A

print(An(6))
# n = 60
# B = -1 * np.tri(n, dtype=np.float64) + 2 * np.eye(n,dtype=np.float64)
# B[:,-1] = np.ones(60,dtype=np.float64)
# A60 = An(n)
# L = np.eye(n)
# i1 = np.tril_indices(n, k=-1)
# L[i1] = A60[i1]
# U = np.triu(A60)
# print(np.linalg.norm(B - L @ U) / np.linalg.norm(B))

# random.seed(12984*n)
# x = random.randn(n)
# y = B @ x
# B1 = np.copy(B)
# x1 = cla_utils.solve_LUP(B1, y)
# print(np.linalg.norm(B @ x1 - y) / np.linalg.norm(B @ x1))
