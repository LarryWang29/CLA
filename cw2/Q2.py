import numpy as np
import cla_utils
import time
import matplotlib.pyplot as plt

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
    for i in range(50):
        # a = int(time.time())
        # np.random.seed(a)
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
    A_hat = np.column_stack((A, b)) # Create the Augmented matrix  
    R = cla_utils.GS_modified(A_hat)
    z = R[:n,-1]
    R_hat = R[:n, :n]
    x = cla_utils.solve_U(R_hat, z)
    return x

def Modified_error(A):
    N = np.shape(A)[1]
    Error = []
    for i in range(50):
        # a = int(time.time())
        # np.random.seed(a)
        x_star = np.random.random(N)
        b = A @ x_star
        A1 = np.copy(A)
        x_hat = MGS_solve_ls_modified(A1, b)
        Error.append(np.sqrt(np.inner(x_star - x_hat, x_star - x_hat)))
    return Error

def Compare_Error(A):
    x_arr = np.linspace(1, 51, 50)
    Unmodified = Get_ls_Error(A)[0]
    Modified = Modified_error(A)
    print(Unmodified, Modified)
    plt.plot(x_arr, np.log(Unmodified), label='MGS without augmented A')
    plt.plot(x_arr, np.log(Modified), label='MGS with augmented A')
    plt.xlabel("Number of iteration")
    plt.ylabel("Log of the error")
    plt.legend(loc="center")
    plt.show()

# Compare_Error(A2)

def Outrange_Error(m,n):
    Error, Error1 = [], []
    for i in range(1, 51):
        # Use time as seed so that different values are generated each time
        # np.random.seed(int(time.time()) + i)
        A = np.random.randn(m,n)
        x = np.random.randn(n)
        # Change the seed so that r isn't equal to x
        np.random.seed(int(time.time()) + 100 + i)
        r = np.random.randn()
        b = A @ x + r
        A1 = A.copy()
        x_hat = MGS_solve_ls_modified(A1, b)
        x_hat1 = MGS_solve_ls(A1, b)
        Error.append(np.sqrt(np.inner(x - x_hat, x - x_hat)))
        Error1.append(np.sqrt(np.inner(x - x_hat1, x - x_hat1)))
    x_arr = np.linspace(1, 51, 50)
    Error_arr = np.array(Error1) - np.array(Error)
    plt.plot(x_arr, Error_arr, label='Difference in error between the two methods')
    plt.xlabel("Number of iteration")
    plt.ylabel("Difference of error")
    plt.legend(loc="upper center")
    plt.show()

Outrange_Error(100, 20)