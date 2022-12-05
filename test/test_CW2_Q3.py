import numpy as np
from numpy import random
import cla_utils
import pytest

@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_lower_banded(m):
    random.seed(5254*m)
    A = random.randn(m, m)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    print(A)
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)

    x1 = cla_utils.solve_LU(A1, b, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6

@pytest.mark.parametrize('m', [6, 15, 35, 64])
def test_upper_banded(m):
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
def test_doubly_banded(m):
    random.seed(84124*m)
    A = random.randn(m, m)
    # Randomly select a bandwidth
    n = random.randint(1, m-1)
    i1 = np.tril_indices(m, k=-n)
    # Setting entries outside of bandwith to 0
    A[i1] = 0
    i2 = i1[::-1]
    A[i2] = 0
    x = random.randn(m)
    b = A @ x
    A1 = np.copy(A)

    x1 = cla_utils.solve_LU(A1, b, n, n)
    #!!!change test param to b

    #check normal equation residual
    assert(np.sqrt(np.inner(x1-x, x1-x))) < 1.0e-6