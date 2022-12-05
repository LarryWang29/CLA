import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    n = Q.shape[1]
    u = np.zeros(n, dtype='complex_')
    r = v
    for i in range(n):
        k = np.dot(v,Q[:,i])
        u[i] = k
        r = r - k*(Q[:,i])

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    Q_star = np.linalg.inv(Q)
    x = Q_star.dot(b)

    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    Q_star = np.transpose(Q.conjugate())
    P = Q @ Q_star
    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """

    Q = np.linalg.qr(V)[0]
    m, n = np.shape(V)
    P = np.identity(np.shape(Q)[0])
    k = np.shape(Q)[1]
    P = P - Q @ np.transpose(Q.conjugate())
    P = P[:,:(m-k)]

    return P


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    n = np.shape(A)[1]
    R = np.zeros((n,n), dtype = 'complex_')
    for j in range(n):
        v = A[:,j]
        
        if j > 0:
            R[0:j,j] = np.transpose(A[:,0:j].conjugate()) @ (A[:,j])
            arr = A[:,0:j] @ R[0:j,j]
            v = v - arr

        R[j,j] = np.linalg.norm(v)
        A[:,j] = v / R[j,j]

    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    n = np.shape(A)[1]
    R = np.zeros((n,n), dtype = A.dtype)
    for j in range(n):
        v = A[:,j]
        R[j,j] = np.linalg.norm(v)
        q = v / R[j,j]
        A[:,j] = q

        R[j,(j+1):] = np.transpose(A[:,(j+1):n].conjugate()) @ (A[:,j])
        A[:,(j+1):] -= np.dot(A[:,j][:,None], R[j,(j+1):][None,:])

    return R


def GS_modified_get_R(A, k):
    """ 
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    B = A.copy()
    n = np.shape(A)[1]
    R_original = GS_modified(B)
    R = np.identity(n, dtype = 'complex_')
    Rkk = R_original[k-1,k-1]
    R[k-1,k:n] = R_original[k-1,k:n] / Rkk * (-1)
    R[k-1,k-1] = (1 / (Rkk * 1.0))

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
