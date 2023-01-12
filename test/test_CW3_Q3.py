import pytest
import cla_utils
from numpy import random
import numpy as np
from cw3 import Q3
import scipy as sp


@pytest.mark.parametrize('m', [10, 15, 20, 25])
def test_Q3c(m):
    random.seed(323*m)
    A = np.random.randn(m,m) # Generate random matrix
    ATA = A.T @ A # Calculate A^TA
    evals = cla_utils.pure_QR(ATA, 2000, 1.0e-5).diagonal() # Use QR to compute eigenvalues of ATA
    evals = np.sort(evals) # Sort eigenvalues from largest to smallest
    evals = evals[::-1]
    K = np.zeros((m,m))
    for i in range(m):
        K[i,i] = np.sqrt(evals[i]) # Create matrix K which should be the same as D
    D = Q3.compute_D(A) # Compute D
    assert(np.linalg.norm(K-D) < 1.0e-3)

@pytest.mark.parametrize('m', [10, 15, 20, 25])
def test_Q3d(m):
    random.seed(931*m)
    A = np.random.randn(m,m)
    D, H = Q3.compute_D(A, True) # Obtain D and H from the given A
    K = Q3.H_evec(D,H) # Compute the unitary matrix K
    V = np.sqrt(2) * K[:m,:m] # Extract V and U from K
    U = np.sqrt(2) * K[m:,:m]
    assert(np.linalg.norm(V @ V.T - np.eye(m)) < 1.0e-3) # Check that V and U are unitary
    assert(np.linalg.norm(U @ U.T - np.eye(m)) < 1.0e-3)
    assert(np.linalg.norm(A - U @ D @ V.T) < 1.0e-3) # Check that UDV^T is the same as A

@pytest.mark.parametrize('m', [6, 8, 10])
def test_Q3e(m):
    random.seed(185*m)
    A = random.randn(m,m)
    A[m-1,:] = 2 # Set some columns of A to 2
    A[m-3,:] = 2
    A[m-5,:] = 2
    b = random.randn(m)
    x = Q3.min_norm_sol(A, b)
    Nsp = sp.linalg.null_space(A) # Obtain the nullspace of A
    for i in range(np.shape(Nsp)[1]): # Check that the solution is orthogonal to the nullspace
        assert(np.inner(x, Nsp[:,i]))