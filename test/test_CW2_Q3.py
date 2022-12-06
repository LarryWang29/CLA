import numpy as np
from numpy import random
import cla_utils
import pytest

@pytest.mark.parametrize('m', [20, 204, 18])
def test_LU_inplace(m):
    random.seed(481*m)
    A = random.randn(m, m)
    n, p = random.randint(1, m-1, size=2)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    u1 = np.triu_indices(m, k=p)
    A[u1] = 0
    A0 = 1.0*A
    cla_utils.LU_inplace(A, n, p)
    L = np.eye(m)
    i2 = np.tril_indices(m, k=-1)
    L[i2] = A[i2]
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)

@pytest.mark.parametrize('m', [6, 12, 24, 30])
def test_solve_L_banded(m):
    random.seed(134*m)
    A = random.randn(m, m)
    B = np.tril(A)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    B[i1] = 0
    x = random.randn(m)
    b = B @ x
    B1 = np.copy(B)

    x1 = cla_utils.solve_L(B1, b, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6

@pytest.mark.parametrize('m', [6, 12, 24, 30])
def test_solve_U_banded(m):
    random.seed(3144*m)
    A = random.randn(m, m)
    B = np.triu(A)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    u1 = np.triu_indices(m, k=n)
    # Setting entries outside of bandwith to 0
    B[u1] = 0
    x = random.randn(m)
    b = B @ x
    B1 = np.copy(B)

    x1 = cla_utils.solve_U(B1, b, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6


@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_LU_solve_lower_banded(m):
    random.seed(5254*m)
    A = random.randn(m, m)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)

    x1 = cla_utils.solve_LU(A1, b, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6

@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_LU_solve_upper_banded(m):
    random.seed(1254*m)
    A = random.randn(m, m)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    i1 = np.triu_indices(m, k=n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)

    x1 = cla_utils.solve_LU(A1, b, None, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6


@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_LU_solve_doubly_banded(m):
    random.seed(84124*m)
    A = random.randn(m, m)
    # Randomly generates 2 bandwidths
    n, p = random.randint(1, m-1, size=2)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    u1 = np.triu_indices(m, k=p)
    A[u1] = 0
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)

    x1 = cla_utils.solve_LU(A1, b, n, p)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6