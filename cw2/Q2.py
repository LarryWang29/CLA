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

def Get_Error1(A):
    """
    Measures the error in both MGS and householder algorithm, provided the matrix A

    :param A: an mxm-dimensional numpy array

    :return Err: norm of difference between numerical solution and actual solution
    """
    N = np.shape(A)[1]
    Error, Error1 = [], []
    for i in range(100):
        # Set different seeds so different x_star are generated
        np.random.seed(i)
        # Generate vector x_star
        x_star = np.random.random(N)
        b = A @ x_star
        A1 = np.copy(A)
        # Obtain solution using MGS method
        x_hat = MGS_solve_ls(A1, b)
        Error.append(np.linalg.norm(x_star - x_hat))
        A2 = np.copy(A)
        # Obtain solution using MGS1 method
        x_hat1 = cla_utils.householder_ls(A2, b)
        Error1.append(np.linalg.norm(x_star - x_hat1))
    return Error, Error1
x_arr = np.linspace(1, 100, 100)
MGS, Householder = Get_Error1(A2)
plt.plot(x_arr, np.log10(MGS), label='$\log_{10}$(MGS Error)')
plt.plot(x_arr, np.log10(Householder), label='$\log_{10}$(Householder Error)')
plt.xlabel("Iteration index")
plt.ylabel("$\log_{10}$(Error)")
plt.legend(loc="center")
plt.title('Comparison of error between Householder and MGS')
plt.show()

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
    # Create the Augmented matrix 
    A_hat = np.column_stack((A, b))
    # Get z and R_hat
    R = cla_utils.GS_modified(A_hat)
    z = R[:n,-1]
    R_hat = R[:n, :n]
    # Solve for x
    x = cla_utils.solve_U(R_hat, z)
    return x

def Modified_error(A):
    N = np.shape(A)[1]
    Error = []
    for i in range(100):
        # a = int(time.time())
        # np.random.seed(a)
        x_star = np.random.random(N)
        b = A @ x_star
        A1 = np.copy(A)
        x_hat = MGS_solve_ls_modified(A1, b)
        Error.append(np.linalg.norm(x_star - x_hat))
    return Error

def Compare_Error(A):
    x_arr = np.linspace(1, 100, 100)
    Unmodified = Get_Error1(A)[0]
    Modified = Modified_error(A)
    print(Unmodified, Modified)
    plt.plot(x_arr, np.log10(Unmodified), label='MGS without augmented A')
    plt.plot(x_arr, np.log10(Modified), label='MGS with augmented A')
    plt.xlabel("Iteration index")
    plt.ylabel("$\log_{10}$(Error)")
    plt.legend(loc="center")
    plt.title('Comparison of error between Modified MGS and normal MGS')
    plt.show()

# Compare_Error(A2)

def Get_Error3(k):
    """
    Computes the error in least squares solution when b is outside of 
    the column space of matrix A using the three different methods.
    Also plots line graph for each method, with y-axis being the magnitude
    of the error, x-axis being the iteration index

    :param k: Number of iterations
    """
    Error, Error1, Error2 = [], [], []
    cla_utils.householder_qr(A2)
    for i in range(k):
        # Set a random seed
        np.random.seed(100+i)
        # Generate random vector to append to the end of A2
        v = np.random.randn(100)
        x = np.random.randn(20)
        A2_hat = np.column_stack((A2, v))
        # Apply QR factorisation in place
        cla_utils.GS_modified(A2_hat)
        # Use the last column of Q as r
        r = A2_hat[:,-1]
        # Generate b
        b = A2 @ x + r
        A1 = A2.copy()
        A3 = A2.copy()
        # Compute numerical solutions
        x_hat = MGS_solve_ls_modified(A1, b)
        x_hat1 = MGS_solve_ls(A1, b)
        x_hat2 = cla_utils.householder_ls(A3, b)
        # Calculate errors
        Error.append(np.linalg.norm(x - x_hat))
        Error1.append(np.linalg.norm(x - x_hat1))
        Error2.append(np.linalg.norm(x - x_hat2))
    x_arr = np.linspace(1, k, k)
    # Plot graphs showing the errors
    plt.plot(x_arr, np.log10(Error), label='Modified MGS')
    plt.plot(x_arr, np.log10(Error1), label='Unmodified MGS')
    plt.plot(x_arr, np.log10(Error2), label='Householder')
    plt.xlabel("Iteration index")
    plt.ylabel("$\log_{10}$(Error)")
    plt.legend(loc="center")
    plt.title('Comparison of errors with $b$ outside of column range')
    plt.show()

# Get_Error3(100)