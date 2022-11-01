import pytest
import cla_utils
from numpy import random
import numpy as np
from cw1 import Question1

A1 = np.loadtxt('A1.dat')
@pytest.mark.parametrize('m', [20, 40, 87])
def test_Q2cf(m):
    random.seed(1878*m)
    A = random.randn(m, m)
    B = A.copy()  # make a deep copy
    P, R = cla_utils.householder(B, kmax=None, swap=True)
    assert((np.linalg.norm(np.dot(R.T, R) - np.dot(P.T, P))) < 1.0e-7)

def test_Q2d():
    B = A1.copy()  # make a deep copy
    P, R = cla_utils.householder(B, kmax=None, swap=True, reduced_tol = 1.0e-07)
    assert(np.linalg.matrix_rank(R) == np.linalg.matrix_rank(A1))