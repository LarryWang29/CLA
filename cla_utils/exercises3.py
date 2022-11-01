import mailcap
import numpy as np

def householder(A, kmax=None, swap=None, reduced_tol=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = np.shape(A)
    Permutated_A = A.copy()
    if kmax is None:
        kmax = n
    NormArray = np.linalg.norm(A, axis=0)
    for k in range(kmax):
        if swap:
            if k != 0:
                top_sq = np.square(A[k, k:n])
                NormArray[k:n] = NormArray[k:n] - top_sq
            if reduced_tol != None and reduced_tol > 0:
                if NormArray.all() < reduced_tol:
                    break
            k_star = np.argmax(NormArray[k:n])
            A[:,[k, k+k_star]] = A[:, [k+k_star, k]]
            Permutated_A[:,[k, k+k_star]] = Permutated_A[:, [k+k_star, k]]
        x = A[k:m,k]
        if x[0] == 0:
            sgn = 1.0
        else:
            sgn = np.sign(x[0])
        e1 = np.zeros(m-k)
        e1[0] = 1.0
        vk = sgn * np.linalg.norm(x) * e1 + x
        vk = vk / np.linalg.norm(vk)
        A[k:m, k:n] = A[k:m, k:n] - 2.0 * np.outer(vk, np.dot(vk, A[k:m, k:n]))
    
    if swap:
        return Permutated_A, A
    else:
        return A


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    if b.ndim == 1:
        b = np.array([b])
        b = np.transpose(b)
    m, k = np.shape(b)
    x = np.zeros((m,k))
    x[m-1,:] = b[m-1,:] / U[m-1][m-1]
    for i in reversed(range(m-1)):
        x[i,:] = (b[i,:] - np.dot(U[i][i+1:m], x[i+1:m,:])) / U[i][i]        
    return x


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m = np.shape(A)[1]
    A_hat = np.hstack((A,b))
    R = householder(A_hat, m)
    x = solve_U(R[:,0:m], R[:,m:])
    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = np.shape(A)
    I = np.identity(m)
    A_hat = np.hstack((A, I))
    A_star = householder(A_hat, n)
    R, Q = A_star[:,0:n], np.transpose(np.conjugate((A_star[:,n:])))

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, n = np.shape(A)
    Q, R = householder_qr(A)
    R_hat, Q_hat = R[0:n,:], Q[:,0:n]
    x = solve_U(R_hat, np.dot(np.transpose(np.conjugate(Q_hat)), b))
    x = np.ndarray.flatten(x)
    return x
