import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import Q1efgh

# Question 3(c)
def compute_D(A, ret_H = False):
    m = np.shape(A)[0]
    H = np.zeros((2*m, 2*m))
    H[m:, :m] = A
    H[:m, m:] = A.T
    H1 = np.copy(H)
    evals = Q1efgh.shifted_QR(H1, 2000, 1.0e-5)
    evals = np.sort(evals)
    D = np.diag(evals[-m:][::-1])
    if ret_H:
        return D, H
    return D

A = np.random.randn(3, 3)
# D = compute_D(A)
# print(np.linalg.eigvals(D))
# A = np.random.randn(5, 5)
# print(np.linalg.eigvals(-A @ A.T))
# print(cla_utils.pure_QR(-A @ A.T, 1000, 1.0e-5).diagonal())

# Question 3(d)
def H_evec(A):
    D, H = compute_D(A, True)
    evals = D.diagonal()
    evals = np.append(evals, -evals)
    m = np.shape(H)[0]
    evecs = np.empty((m,0))
    for i in evals:
        H1 = np.copy(H)
        evec = cla_utils.inverse_it(H1, np.ones(m), i, 1.0e-5, 1000)[0]
        evecs = np.append(evecs, np.array([evec]).T, axis=1)

    print(evecs)

    # return evecs

H_evec(A)

