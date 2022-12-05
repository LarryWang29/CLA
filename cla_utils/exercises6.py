import cla_utils
import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k dimensional numpy array.
    :return Lk: an mxm dimensional numpy array.

    """
    Lk = np.identity(m)
    k_star = len(lvec)
    Lk[-k_star:, m-k_star-1] = -lvec
    return Lk


def LU_inplace(A, bl=None, bu=None):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    m = np.shape(A)[0]
    for k in range(m-1):
        m1, m2 = m, m
        if bl:
            m1 = min(k+bl, m)
        Lk = A[(k+1):m1, k] / A[k][k]
        if bu:
            m2 = min(k+bu, m)
        A[(k+1):m1, k:m2] = A[(k+1):m1, k:m2] - np.outer(Lk, A[k,k:m2])
        A[(k+1):m1, k] = Lk


def solve_L(L, b, bl=None):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    k = b.ndim
    if k == 1:
        b = np.array([b])
        b = np.transpose(b)
    m, k = np.shape(b)
    x = np.zeros((m,k))
    x[0,:] = b[0,:] / L[0, 0]
    for i in range(1,m):
        j = 0
        if bl:
            j = max(0, k-bl-1)
        x[i,:] = (b[i,:] - np.dot(L[i,j:i], x[j:i,:])) / L[i, i]
    if k == 1:
        return x[:,0] 
    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    LU_inplace(A)
    m = np.shape(A)[0]
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    B = np.eye(m)
    y = cla_utils.solve_L(L, B)
    x = cla_utils.solve_U(U, y)
    return x

