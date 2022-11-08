import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    K = A.T @ A
    v = np.linalg.eig(K)[0]
    v.sort()
    o2norm = np.sqrt(v[-1])

    return o2norm


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    K = A.T @ A
    v = np.linalg.eig(K)[0]
    v.sort()
    A_cond = np.sqrt(v[-1]) 
    A_invcon = np.sqrt(1 / v[0])
    ncond = A_cond * A_invcon

    return ncond
