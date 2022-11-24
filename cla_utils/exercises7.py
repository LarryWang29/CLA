import numpy as np
import cla_utils

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    p[i], p[j] = p[j], p[i]
    return p


def LUP_inplace(A, p_count=False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
    m = np.shape(A)[0]
    p = np.arange(0,m)
    I = np.eye(m)
    sign = 1
    for k in range(m-1):
        i = np.argmax(np.abs(A[k:,k])) + k
        p = perm(p, k, i)
        if p_count:
            if i != k:
                sign = sign * (-1)
        A[[k, i]] = A[[i, k]]
        Lk = A[(k+1):, k] / A[k][k]
        A[(k+1):, k:] = A[(k+1):, k:] - np.outer(Lk, A[k,k:m])
        A[(k+1):, k] = Lk
    if p_count:
        return p, sign
    else:
        return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    m = np.shape(A)[0]
    p = LUP_inplace(A)
    b = b[p]
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    Ux = cla_utils.solve_L(L, b)
    x = cla_utils.solve_U(U, Ux)
    return x

def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
    m = np.shape(A)[0]
    p, sign = LUP_inplace(A, p_count=True)
    for i in range(m):
        sign *= A[i][i]
    return sign
