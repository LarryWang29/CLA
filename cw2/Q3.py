import numpy as np
import cla_utils
from numpy import random
import matplotlib.pyplot as plt
import time

def Mat_D(N, s):
    """
    Generates the matrix D for given values of N and s

    :param N: Dimension of vector u
    :param s: constant that governs the strength of correlation

    :return D: return the matrix D as defined in the question
    """
    N2 = N ** 2
    s2 = s ** 2
    # Generate Diagonal entries 
    D = np.eye(N2) * (1 + 4 * s2)
    # Generate superdiagonal and subdiagonal entries
    sn1 = -s2 * np.ones(N2-1)
    D += np.diag(sn1, 1) + np.diag(sn1, -1)
    # Generate nth diagonal entries
    sn2 = -s2 * np.ones(N2-N)
    D += np.diag(sn2, N) + np.diag(sn2, -N)
    # Setting specific entries on superdiagonal and subdiagonal to 0
    for i in range(1,N):
        D[i*N, i*N-1] = 0
        D[i*N-1, i*N] = 0
    return D

def sim(N, s, option, t=False):
    """
    Simulates the U-array and plots it.

    :param N: dimension of U-array
    :param s: a parameter indicating the strength of correlation between 
    adjacent points
    :param option: determines whether banded implementation or original
    implementation is used
    :param t: if True, returns the runtime of solver
    """
    N2 = N ** 2
    w = random.randn(N2)
    # Generate D
    D = Mat_D(N,s)
    # Choose option for solver
    if option=='Banded':
        # Getting start and finish times of the function
        if t:
            T1 = time.time()
        u_hat = cla_utils.solve_LU(D, w, N+1, N+1)
        if t:
            T2 = time.time()
    elif option=='Original':
        if t:
            T1 = time.time()
        u_hat = cla_utils.solve_LU(D, w)
        if t:
            T2 = time.time()
    else:
        raise NotImplementedError("Use valid option")
    u = np.reshape(u_hat, (N, N))
    if t:
        # Return runtime of solver if t=True
        return T2 - T1
    # else plots the heatmap
    plt.imshow(u, cmap='gray', vmin=-1, vmax=1)
    plt.show()



def matvec_prod_banded(A, b, bandwidth):
    """
    Computes the Matrix vector product given the bandwidth of the matrix

    :param A: Matrix involved in the multiplication
    :param b: vector involved in the multiplication
    :bandwidth: bandwidth of the matrix

    :return Prod: the product of A and b
    """
    m = len(b)
    y = np.zeros(m)
    for i in range(m):
        # finding indices i and j used for slicing
        j = min(i+bandwidth+1, m)
        k = max(0, i-bandwidth)
        y[i] = np.dot(A[i,k:j], b[k:j])
    return y

def iterative_solver(N, s, rho, v, tol, test=False):
    """
    Iteratively computes the solution u_hat for the system.

    :param N: Dimension of the U-array
    :param s: Corrolative strength of adjacent indices
    :param rho: Parameter used in the equation
    :param nu: Parameter used in the equation
    :param tol: Tolerance parameter used for checking stopping criteria
    :param test: If true, function will return w

    :return u_hat: The iterative solution
    :return iter1: Number of iterations
    :return w: Returns w to use for testing
    """
    N2 = N ** 2
    w = random.randn(N2)
    u_hat = np.zeros(N2)
    D = Mat_D(N,s)
    # Creating the modified block tridiagonal matrix A1 and the tridiagonal matrix A2
    A_block = 3*s**2 * np.eye(N) - s**2 * np.triu(np.tri(N, N, k=1), k=-1)
    IN = np.eye(N)
    A1_block, A2_block = (1 + rho) * IN + A_block, (1 + v) * IN + A_block
    A3_block, A4_block= rho * IN - A_block, v * IN - A_block
    iter1 = 0
    # Keep track of the permutation
    perm = np.arange(N2)
    # Permute the indices based on mod(N)
    perm = sorted(perm, key=lambda x: x%N)
    inv_perm = np.empty(N2, dtype=np.int32)
    # Generate the inverse permutation
    for i in np.arange(N2):
        inv_perm[perm[i]] = i
    # Check stopping condition and impose max interation count
    while np.linalg.norm(D @ u_hat - w) > tol * np.linalg.norm(w) and iter1 <= 1000:
        u_hat_star = np.zeros(N2)
        v1 = np.zeros(N2)
        for i in range(N):
            # Calculate Matrix vector product using banded algorithm
            v1[i*N:(i+1)*N] = matvec_prod_banded(A3_block, u_hat[i*N:(i+1)*N], 1) + w[i*N:(i+1)*N]
        v1 = v1[perm]
        for j in range(N):
            # Solve for u_hat_star block-wise
            u_hat_star[j*N:(j+1)*N] = cla_utils.solve_LU(A1_block, v1[j*N:(j+1)*N], 1, 1)
        v2 = np.zeros(N2)
        for k in range(N):
            # Calculate Matrix vector product using banded algorithm
            v2[k*N:(k+1)*N] = matvec_prod_banded(A4_block, u_hat_star[k*N:(k+1)*N], 1)
        # Reverse the permutation
        v2 = v2[inv_perm] + w
        for l in range(N):
            # Solve for u_hat block-wise
            u_hat[l*N:(l+1)*N] = cla_utils.solve_LU(A2_block, v2[l*N:(l+1)*N], 1, 1)
        iter1 += 1
    if test:
        return u_hat, iter1, w
    return u_hat, iter1