import cla_utils
import math
import numpy as np

def GetCoeffs(x, fx, option):
    N = len(x)
    def phi(x, i):
        return np.exp(-(x * (N-1) - i) ** 2)
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            A[i][j] = phi(x[j], i)
    print(A)
    if option == 'householder':
        Q, R = cla_utils.householder_qr(A)
    elif option == 'QR':
        Q, R = cla_utils.GS_modified_R(A)
    else:
        raise NotImplementedError("Option must be householder or modified Gram-Schmidt")
    b = cla_utils.solve_U(R, np.dot(np.transpose(np.conjugate(Q)), fx))
    b = np.ndarray.flatten(x)
    print(b)
    return b
