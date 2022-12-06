import numpy as np
import cla_utils
from numpy import random
import matplotlib.pyplot as plt
import cProfile
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
    D = np.eye(N2) * (1 + 4 * s2)
    sn1 = -s2 * np.ones(N2-1)
    D += np.diag(sn1, 1) + np.diag(sn1, -1)
    sn2 = -s2 * np.ones(N2-N)
    D += np.diag(sn2, N) + np.diag(sn2, -N)
    for i in range(1,N-1):
        D[i*N, i*[N+1]] = 0
        D[i*N, N+1] = 0
    return D

def sim(N, s, option):
    N2 = N ** 2
    np.random.seed(int(time.time()))
    w = np.random.randn(N2)
    print(w)
    D = Mat_D(N,s)
    if option=='Banded':
        u_hat = cla_utils.solve_LU(D, w, N+1, N+1)
    elif option=='Original':
        u_hat = cla_utils.solve_LU(D, w)
    return u_hat
    # u = np.reshape(u_hat, (N, N))
    # c = plt.imshow(u, cmap='hot', vmin=-4, vmax=4)
    # plt.colorbar(c)
    # plt.show()

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
        j = min(i+bandwidth+1, m)
        k = max(0, i-bandwidth)
        y[i] = np.dot(A[i,k:j], b[k:j])
    return y

def iterative_solver(N, s, rho, v, tol):
    """
    Iteratively computes the solution u_hat for the system
    """
    s2 = s ** 2
    N2 = N ** 2
    w = np.random.randn(N2)
    u_hat = np.zeros(N2)
    D = Mat_D(N,s)
    # Creating the modified block tridiagonal matrix A1 and the tridiagonal matrix A2
    A1 = np.kron(np.eye(N), (1 + rho + 3*s2) * np.eye(N) - s2 * np.triu(np.tri(N, N, k=1), k=-1))
    sn1 = s2 * np.ones(N2-1)
    # Generating the other matrices used in the system of equations
    A2 = (rho- 2*s2) * np.eye(N2) + np.diag(sn1, 1) + np.diag(sn1, -1)
    for i in range(1,N-1):
        A2[i*N, i*[N+1]] = 0
        A2[i*N, N+1] = 0
    A3 = -A2 + (1+rho+v) * np.eye(N2)
    A4 = -A1 + (1+rho+v) * np.eye(N2)
    iter1 = 0
    # Keep track of the permutation
    perm = np.arange(N2)
    # Permute the indices based on mod(N)
    perm = sorted(perm, key=lambda x: x%N)
    inv_perm = np.empty(N2, dtype=np.int32)
    for i in np.arange(N2):
        inv_perm[perm[i]] = i
    while np.linalg.norm(D @ u_hat - w) > tol * np.linalg.norm(w):
        v1 = matvec_prod_banded(A2, u_hat, 1) + w
        v1 = v1[perm]
        u_hat_star = np.linalg.solve(A1, v1)
        v2 = matvec_prod_banded(A4, u_hat_star, 1)
        v2 = v2[inv_perm] + w
        u_hat = np.linalg.solve(A3, v2) 
        iter1 += 1
    return u_hat, iter1

print(iterative_solver(50, 1, 1, 1, 1e-6))