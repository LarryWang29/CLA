import numpy as np
import cla_utils

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

def MGS_ls_Error(A):
    """
    Measures the error in the MGS algorithm, provided the matrix A

    :param A: an mxm-dimensional numpy array

    :return Err: norm of difference between numerical solution and actual solution
    """
    