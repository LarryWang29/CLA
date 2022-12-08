import numpy as np
import matplotlib.pyplot as plt
import Q2

A2 = np.loadtxt('A2.dat')

def Get_Error2(A):
    """
    Measures the error in both MGS and Modified MGS algorithm, provided 
    the matrix A

    :param A: an mxn-dimensional numpy array

    :return Error: an array of errors for MGS algorithm
    :return Error1: an array of errors for Modified MGS algorithm
    """
    N = np.shape(A)[1]
    Error, Error1 = [], []
    for i in range(100):
        # Generate x_star and b
        x_star = np.random.random(N)
        b = A @ x_star
        A1 = np.copy(A)
        # Calculate errors of each method
        x_hat = Q2.MGS_solve_ls_modified(A1, b)
        Error.append(np.linalg.norm(x_star - x_hat))
        x_hat1 = Q2.MGS_solve_ls(A1, b)
        Error1.append(np.linalg.norm(x_star - x_hat1))
    x_arr = np.linspace(1, 100, 100)
    # Plots the data
    plt.plot(x_arr, np.log10(Error1), label='MGS without augmented A')
    plt.plot(x_arr, np.log10(Error), label='MGS with augmented A')
    plt.xlabel("Iteration index")
    plt.ylabel("$\log_{10}$(Error)")
    plt.legend(loc="center")
    plt.title('Comparison of error between Modified MGS and normal MGS')
    plt.show()
Get_Error2(A2)