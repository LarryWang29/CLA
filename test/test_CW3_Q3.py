import pytest
import cla_utils
from numpy import random
import numpy as np
from cw3 import Q3


@pytest.mark.parametrize('m', [10, 15, 24, 28])
def test_Q3c(m):
    random.seed(323*m)
    B = np.random.randn(m,m)
    K = np.zeros((m,m))
    A = 1/2 * B @ B.T # Generate a symmetric matrix
    evals = cla_utils.pure_QR(A, 2000, 1.0e-5).diagonal() # Calculate eigenvalues using shifted QR
    evals = np.sort(evals)
    evals = evals[::-1]
    for i in range(m):
        K[i,i] = evals[i]
    D = Q3.compute_D(A)
    assert(np.linalg.norm(K-D) < 1.0e-3)

@pytest.mark.parametrize('m', [10, 15, 24, 28])
def test_Q3d(m):
    random.seed(931*m)
    B = np.random.randn(m,m)
    A = 1/2 * B @ B.T # Generate a symmetric matrix 
    D, H = Q3.compute_D(A, True)
    K = Q3.H_evec(D,H)
    V = np.sqrt(2) * K[:m,:m]
    U = np.sqrt(2) * K[m:,:m]
    assert(np.linalg.norm(V @ V.T - np.eye(m)) < 1.0e-3)
    assert(np.linalg.norm(U @ U.T - np.eye(m)) < 1.0e-3)
    assert(np.linalg.norm(A - U @ D @ V.T) < 1.0e-3)