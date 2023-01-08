import numpy as np
import numpy.random as random
import cla_utils

def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m = np.shape(A)[0]
    Q = np.zeros((m, k+1), dtype=b.dtype)
    H = np.zeros((k+1, k), dtype=b.dtype)
    Q[:,0] = b / np.linalg.norm(b)
    for i in range(k):
        v = A @ Q[:,i]
        H[:i+1, i] = Q[:,:i+1].conj().T @ v
        v -= np.sum((H[:i+1, i] * Q[:,:i+1]), axis=1)
        H[i+1, i] = np.linalg.norm(v)
        Q[:,i+1] = v / np.linalg.norm(v)
    return Q, H


def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    nits = 0
    m = np.shape(A)[0]
    Q = np.zeros((m, 2), dtype=A.dtype)
    H = np.zeros((2, 1), dtype=A.dtype)
    Q[:,0] = b / np.linalg.norm(b)
    if x0 is None:
        x0 = b
    e1 = np.zeros(2)
    if return_residual_norms:
        rnorms = []
    if return_residuals:
        r = np.empty((m, 0))
    e1[0] = 1
    while nits < maxit:
        v = A @ Q[:,nits]
        H[:nits+1, nits] = Q[:,:nits+1].conj().T @ v
        v -= np.sum((H[:nits+1, nits] * Q[:,:nits+1]), axis=1)
        H[nits+1, nits] = np.linalg.norm(v)
        Q[:,nits+1] = v / np.linalg.norm(v)
        y = cla_utils.householder_ls(H, np.linalg.norm(b) * e1)
        xn = Q[:,:nits+1] @ np.array(y)
        Rn = np.linalg.norm(H @ y - np.linalg.norm(b) * e1)
        if return_residual_norms:
            rnorms.append(Rn)
        if return_residuals:
            r = np.append(r, np.array([A @ xn - b]).T, axis=1)
        nits += 1
        if Rn < tol:
            if return_residual_norms and return_residuals:
                return xn, nits, rnorms, r
            elif return_residual_norms:
                return xn, nits, rnorms
            elif return_residuals:
                return xn, nits, r
            else:
                return xn, nits
        H = np.c_[H, np.zeros(nits+1)] # Update the dimensions of H
        H = np.r_[H, np.array([np.zeros(nits+1)])]
        Q = np.c_[Q, np.zeros(m)] # Update the dimensions of Q
        e1 = np.insert(e1, nits+1, 0) # Change dimension of e1 vector
    return xn, -1


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ', dtype='float64')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ', dtype='float64')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
