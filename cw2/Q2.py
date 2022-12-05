import numpy as np
import cla_utils
import time
A2 = np.loadtxt('A2.dat')

def MGS_solve_ls(A, b):
    """
    Given a mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    R_hat = cla_utils.GS_modified(A) # Get the modified GS decomposition
    z = np.conj(A.T) @ b # Form vector z
    x = cla_utils.solve_U(R_hat, z) # Obtain the solution
    return x

def Get_ls_Error(A):
    """
    Measures the error in the MGS algorithm, provided the matrix A

    :param A: an mxm-dimensional numpy array

    :return Err: norm of difference between numerical solution and actual solution
    """
    N = np.shape(A)[1]
    Error, Error1 = [], []
    for i in range(10):
        a = int(time.time())
        np.random.seed(a)
        x_star = np.random.random(N)
        b = A @ x_star
        A1 = np.copy(A)
        x_hat = MGS_solve_ls(A1, b)
        Error.append(np.sqrt(np.inner(x_star - x_hat, x_star - x_hat)))
        A2 = np.copy(A)
        x_hat1 = cla_utils.householder_ls(A2, b)
        Error1.append(np.sqrt(np.inner(x_star - x_hat1, x_star - x_hat1)))
    return Error, Error1

# print(Get_ls_Error(A2))

def MGS_solve_ls_modified(A, b):
    """
    Solves least squares problem using Modified Gram-Schmidt by using augmented 
    matrix A_hat, where the vector b is added a an extra column to the right of A.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    n = np.shape(A)[1]
    A_hat = np.hstack([A, b]) # Create the Augmented matrix
    R = cla_utils.GS_modified(A_hat)
    R_hat, Q_hat, q, rho = R[:n, :n], A[:,:n], A[:,-1], R[-1,-1]
    z = np.conj(Q_hat.T) @ q * rho / np.dot(q,q)
    x = cla_utils.solve_U(R_hat, z)
    return x