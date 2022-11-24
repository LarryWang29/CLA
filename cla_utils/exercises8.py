import numpy as np

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    m = np.shape(A)[0]
    I = np.eye(m)
    v = A[:,0]
    Q1 = I - 2 * np.outer(v, np.conjugate(v)) / np.dot(v, np.conjugate(v))
    A1 = Q1 @ A @ np.conjugate(Q1.T)
    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = np.shape(A)[0]
    for k in range(m-2):
        x = A[k+1:,k]
        if x[0] == 0:
            sgn = 1.0
        else:
            sgn = np.sign(x[0])
        e1 = np.zeros(m-k-1)
        e1[0] = 1.0
        vk = sgn * np.sqrt(np.inner(x,x)) * e1 + x
        vk = vk / np.sqrt(np.inner(vk, vk))
        A[k+1:, k:] = A[k+1:, k:] - 2.0 * np.outer(vk, np.dot(vk, A[k+1:, k:]))
        A[:,k+1:] = A[:,k+1:] - 2.0 * np.outer(A[:,k+1:] @ vk, np.conjugate(vk))


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m = np.shape(A)[0]
    Q = np.eye(m)
    for k in range(m-2):
        x = A[k+1:,k]
        if x[0] == 0:
            sgn = 1.0
        else:
            sgn = np.sign(x[0])
        e1 = np.zeros(m-k-1)
        e1[0] = 1.0
        vk = sgn * np.sqrt(np.inner(x,x)) * e1 + x
        vk = vk / np.sqrt(np.inner(vk, vk))
        A[k+1:, k:] = A[k+1:, k:] - 2.0 * np.outer(vk, np.dot(vk, A[k+1:, k:]))
        A[:,k+1:] = A[:,k+1:] - 2.0 * np.outer(A[:,k+1:] @ vk, np.conjugate(vk))
        Q[k+1:, :] = Q[k+1:, :] - 2.0 * np.outer(vk, np.dot(vk, Q[k+1:, :]))
    return Q.T

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    raise NotImplementedError
