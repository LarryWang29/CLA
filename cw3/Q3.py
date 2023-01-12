import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from cw3 import Q1efgh
import scipy as sp

# Question 3(c)
def compute_D(A, ret_H = False, rank_def=False):
    """
    Computes the matrix D as in the question, given a matrix A

    :param A: the matrix A
    
    :return D: the matrix D as described in the question
    """
    m = np.shape(A)[0]
    H = np.zeros((2*m, 2*m)) # Form the matrix H
    H[m:, :m] = A
    H[:m, m:] = A.T
    H1 = np.copy(H)
    mu = 1
    if rank_def:
        Ak = cla_utils.pure_QR(mu*np.eye(2*m) + H1, 2000, 1.0e-5)
        evals = np.array(Q1efgh.pure_QR_eig(Ak))
    else:
        # Calcualte shifted eigenvalues of H
        evals = cla_utils.pure_QR(mu*np.eye(2*m) + H1, 2000, 1.0e-5).diagonal()
    # Reverse the shift
    evals = np.copy(evals)
    evals -= mu
    evals = np.sort(evals)
    # Sort the eigenvalues  

    # Form D using the eigenvalues, ordered from largest to smallest
    D = np.diag(evals[-m:][::-1])
    if ret_H:
        return D, H
    return D

# Question 3(d)
def H_evec(D, H):
    """
    Computes the eigenvectors of the matrix H

    :param D: matrix D from the eigendecomposition of H
    :param H: matrix H as described in the question

    :return evecs: returns a matrix whose columns are eigenvectors of H
    """
    evals = D.diagonal() # Extract the positive eigenvalues of H from D
    evals = np.append(evals, -evals) # Add on their negative counterparts
    m = np.shape(H)[0]
    n = np.shape(D)[0]
    evecs = np.empty((m,0)) # Create array to store eigenvectors
    H1 = np.copy(H)
    for i in evals:
        # Apply inverse iteration to the eigenvalues
        evec = cla_utils.inverse_it(H1, 1.2 * np.ones(m), i, 1.0e-5, 1000)[0]
        evecs = np.append(evecs, np.array([evec]).T, axis=1)
    for i in range(n):
        # Make sure signs of obtained eigenvectors are correct
        if np.abs(evecs[0,i] - evecs[0,i+n]) <= 1.0e-03:
            continue
        else:
            evecs[:,i+n] *= -1
    return evecs

# Question 3(e)
def min_norm_sol(A, b):
    m = np.shape(A)[0]
    D, H = compute_D(A, True, True)
    H1 = H_evec(D, H) # Obtain the unitary matrix from eigendecomposition of H
    V = H1[:m,:m] * np.sqrt(2) # Extract V and U from H1
    U = H1[m:,:m] * np.sqrt(2)
    D1 = np.zeros((m,m), dtype='complex')
    # Compute the pseudoinverse of D
    for i in range(m):
        if D[i,i] > 1.0e-6:
            D1[i,i] = 1 / D[i,i]
    # Compute the minimum norm solution
    x = V @ D1 @ U.T @ b
    return x
