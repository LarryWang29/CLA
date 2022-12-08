import numpy as np
from numpy import random
import cla_utils
import pytest
from cw2 import Q3

@pytest.mark.parametrize('m', [20, 204, 18])
def test_LU_inplace(m):
    random.seed(481*m)
    A = random.randn(m, m)
    # Generate random bandwidths
    n, p = random.randint(1, m-2, size=2)
    A = np.tril(A, p)
    # Setting entries outside of bandwith to 0
    A = np.triu(A, -n)
    A0 = 1.0*A
    cla_utils.LU_inplace(A, n, p)
    L = np.eye(m)
    i2 = np.tril_indices(m, k=-1)
    L[i2] = A[i2]
    U = np.triu(A)
    A1 = np.dot(L, U)
    # Check Error
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)

@pytest.mark.parametrize('m', [6, 12, 24, 30])
def test_solve_L_banded(m):
    random.seed(134*m)
    A = random.randn(m, m)
    B = np.tril(A)
    # Randomly select a bandwidth
    n = random.randint(1, m-2)
    # Setting entries outside of bandwith to 0
    B = np.triu(B, -n)
    x = random.randn(m)
    b = B @ x
    B1 = np.copy(B)
    x1 = cla_utils.solve_L(B1, b, n)
    #check normal equation residual
    assert(np.linalg.norm(x1-x)) < 1.0e-6

@pytest.mark.parametrize('m', [6, 12, 24, 30])
def test_solve_U_banded(m):
    random.seed(3144*m)
    A = random.randn(m, m)
    B = np.triu(A)
    # Randomly select a bandwidth
    n = random.randint(1, m-2)
    # Setting entries outside of bandwith to 0
    B = np.tril(B, n)
    x = random.randn(m)
    b = B @ x
    B1 = np.copy(B)
    x1 = cla_utils.solve_U(B1, b, n)
    #check normal equation residual
    assert(np.linalg.norm(x1-x)) < 1.0e-6

@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_LU_solve_doubly_banded(m):
    random.seed(84124*m)
    A = random.randn(m, m)
    # Randomly generates 2 bandwidths
    n, p = random.randint(1, m-2, size=2)
    A = np.triu(A, -n)
    # Setting entries outside of bandwith to 0
    A = np.tril(A, p)
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)
    x1 = cla_utils.solve_LU(A1, b, n, p)
    #check normal equation residual
    assert(np.linalg.norm(x1-x)) < 1.0e-6

@pytest.mark.parametrize('m', [5, 15, 20])
def test_LU_solve_doubly_banded(m):
    random.seed(84124*m)
    # Generate D
    D = Q3.Mat_D(m, 1)
    u_hat, iter, w = Q3.iterative_solver(m, 1, 1, 1, 1e-06, True)
    #check that u_hat satisfies the stopping condition
    assert(np.linalg.norm(D@u_hat-w)) < 1.0e-6 * np.linalg.norm(w)