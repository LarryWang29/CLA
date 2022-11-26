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
    Permutated_A = A.copy() # Create a copy of A that keeps track of permuations
    if kmax is None:
        kmax = n
    NormArray = np.square(np.linalg.norm(A, axis=0))
    for k in range(kmax):
        if swap:
            if k != 0:
                top_sq = np.square(A[k-1, k:]) 
                NormArray[k:] = NormArray[k:] - top_sq # Update NormArray

            if reduced_tol != None:
                if ((abs(NormArray[k:])) < reduced_tol ** 2).all(): # Check if norms are smaller than tolerance
                    return Permutated_A, A[:k,:]

            k_star = np.argmax(NormArray[k:])
            A[:,[k, k+k_star]] = A[:, [k+k_star, k]]
            NormArray[k], NormArray[k+k_star] = NormArray[k+k_star], NormArray[k]
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
    if reduced_tol:
        return Permutated_A, A[:n, :n]
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
    k = b.ndim
    if k == 1:
        b = np.array([b])
        b = np.transpose(b)
    m, k = np.shape(b)
    x = np.zeros((m,k))
    x[m-1,:] = b[m-1,:] / U[m-1][m-1]
    for i in reversed(range(m-1)):
        x[i,:] = (b[i,:] - np.dot(U[i][i+1:m], x[i+1:m,:])) / U[i][i]
    if k == 1:
        x = np.ndarray.flatten(x)        
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
    A_star = np.hstack((A, np.transpose(np.array([b]))))
    A_star = householder(A_star)
    R_hat, QB = A_star[0:n,:n], A_star[:n,-1]
    x = solve_U(R_hat, QB)
    return x
