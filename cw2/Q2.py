import numpy as np
import cla_utils
A2 = np.loadtxt('A2.dat')

def MGS_solve_ls(A, b):
    """
    Given a mxm matrix A, use the Modified Gram Schmidt transformation 
    to solve Ax_i=b_i, i=1,2,...,k.

    :param A: an mxn-dimensional numpy array
    :param b: a length n numpy array

    :return x: an mxk-dimensional numpy array, the least squares solution to the 
    system Ax=b
    """
    R_hat = cla_utils.GS_modified(A) # Get the modified GS decomposition
    z = np.conj(A.T) @ b # Form vector z
    x = cla_utils.solve_U(R_hat, z) # Obtain the solution
    return x

def MGS_solve_ls_modified(A, b):
    """
    Solves least squares problem using Modified MGS algorithm by using augmented 
    matrix A_hat, where the vector b is added a an extra column to the right of A.

    :param A: an mxn-dimensional numpy array
    :param b: a length n numpy array

    :return x: an mxk-dimensional numpy array, the least squares solution to the 
    system Ax=b
    """
    n = np.shape(A)[1]
    # Create the Augmented matrix 
    A_hat = np.column_stack((A, b))
    # Get z and R_hat
    R = cla_utils.GS_modified(A_hat)
    z = R[:n,-1]
    R_hat = R[:n, :n]
    # Solve for x
    x = cla_utils.solve_U(R_hat, z)
    return x