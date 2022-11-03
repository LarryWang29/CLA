'''Tests for the first exercise set.'''
import pytest
import cla_utils
from numpy import random
import numpy as np
from cw1 import Question1

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_Q1a_Householder(m, n):
    random.seed(8473*m + 9283*n)
    x = random.randn(m)
    b = random.randn(n)
    delx = 1 / (n-1)
    def phi(x, i):
        xi = i * delx
        return np.exp(- ((x - xi) ** 2) / (delx ** 2))
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            A[i][j] = phi(x[i], j)
    fx = A @ b
    b1 = Question1.GetCoeffs(x, fx, n, option='householder')
    assert(np.linalg.norm(b1-b) < 1.0e-6)

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_Q1a_GramSchmidt(m, n):
    random.seed(8473*m + 9283*n)
    x = random.randn(m)
    b = random.randn(n)
    delx = 1 / (n-1)
    def phi(x, i):
        xi = i * delx
        return np.exp(- ((x - xi) ** 2) / (delx ** 2))
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            A[i][j] = phi(x[i], j)
    fx = A @ b
    b1 = Question1.GetCoeffs(x, fx, n, option='GS')
    assert(np.linalg.norm(b1-b) < 1.0e-6)