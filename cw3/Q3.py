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
    # evals = Q1efgh.shifted_QR(H1, 2000, 1.0e-5)
    mu = 1
    # Ak = cla_utils.pure_QR(mu*np.eye(2*m) + H1, 2000, 1.0e-5)
    evals = cla_utils.pure_QR(mu*np.eye(2*m) + H1, 2000, 1.0e-5).diagonal()
    evals = np.sort(evals)
    evals -= mu
    # print(np.linalg.eigvals(H))
    # print(evals)
    D = np.diag(evals[-m:][::-1])
    if ret_H:
        return D, H
    return D

A = np.random.randn(10, 10)
# D = compute_D(A)
# print(D)
# print(np.linalg.eigvals(D))
# A = np.random.randn(5, 5)
# print(np.linalg.eigvals(-A @ A.T))
# print(cla_utils.pure_QR(-A @ A.T, 1000, 1.0e-5).diagonal())

# Question 3(d)
def H_evec(D, H):
    evals = D.diagonal()
    evals = np.append(evals, -evals)
    m = np.shape(H)[0]
    n = np.shape(D)[0]
    evecs = np.empty((m,0))
    for i in evals:
        H1 = np.copy(H)
        evec = cla_utils.inverse_it(H1, 1.2 * np.ones(m), i, 1.0e-5, 1000)[0]
        evecs = np.append(evecs, np.array([evec]).T, axis=1)
    for i in range(n):
        if np.abs(evecs[0,i] - evecs[0,i+n]) <= 1.0e-03:
            continue
        else:
            evecs[:,i+n] *= -1
    # print(evecs)
    print(np.linalg.norm(evecs[:n,:n] - evecs[:n,n:]))
    print(np.linalg.norm(evecs[n:,n:] + evecs[n:,:n]))
    return evecs
D, H = compute_D(A, True)
H_evec(D, H)

# Question 3(e)
def min_norm_sol(A, b):
    mu = 1.0
    m = np.shape(A)[0]
    D, H = compute_D(A, True)
    print(D)
    D += np.eye(m) * mu
    print(np.linalg.eigvals(H))
    H += np.eye(2*m) * mu
    H1 = H_evec(D, H)
    D -= np.eye(m) * mu
    H -= np.eye(2*m) * mu
    V = H1[:m,:m]
    U = H1[m:,:m]
    print(H1[:m,:m] - H1[:m,m:])
    print(H1[m:,m:] + H1[m:,:m])
    for i in range(m):
        if D[i,i] > 1.0e-6:
            D[i,i] = 1 / D[i,i]
    x = V @ D @ U.T @ b
    return x

# B = np.random.randn(2, 2)
# B[1,:] = 0
# b = np.random.randn(2)
# x1 = min_norm_sol(B, b)
# print(B @ x1)
