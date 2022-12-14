import numpy as np
import cla_utils
from numpy import random

def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(np.random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = np.random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = Q1 @ R1
        Q2, R2 = np.linalg.qr(A)
        A1 = Q2 @ R2
        print(np.linalg.norm(Q1 - Q2), np.linalg.norm(R1 - R2), np.linalg.norm(A - A1))


def back_stab_solve_U(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        R = randomR(m)
        b = random.randn(m)
        x = R @ b
        x1 = cla_utils.solve_U(R,x)
        print(np.linalg.norm(x1-b))


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    for k in range(20):
        A = random.randn(m ,m)
        x = random.randn(m)
        b = A @ x
        x1 = cla_utils.householder_solve(A, b)
        print(np.linalg.norm(x1-x))


# back_stab_householder_solve(50)