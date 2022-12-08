import numpy as np
import cla_utils
import matplotlib.pyplot as plt
import Q2

A2 = np.loadtxt('A2.dat')

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
        x_hat = Q2.MGS_solve_ls_modified(A1, b)
        x_hat1 = Q2.MGS_solve_ls(A1, b)
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
Get_Error3(100)