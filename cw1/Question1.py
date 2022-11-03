import cla_utils
import math
import numpy as np

def GetCoeffs(x, fx, N, option):
    M = len(x)
    delx = 1 / (N-1)
    def phi(x, i):
        xi = i * delx
        return np.exp(- ((x - xi) ** 2) / (delx ** 2))
    A = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            A[i][j] = phi(x[i], j)
    if option == 'householder':
        b = cla_utils.householder_ls(A, fx)
    elif option == 'GS':
        R = cla_utils.GS_modified(A)
        R_hat = R[:N, :N]
        Q_hat = A[:, :N]
        b = cla_utils.solve_U(R_hat, np.dot(np.transpose((Q_hat)), fx))
    else:
        raise NotImplementedError("Option must be householder or modified Gram-Schmidt")

    return b
