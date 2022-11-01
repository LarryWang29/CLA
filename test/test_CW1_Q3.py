import pytest
import cla_utils
from numpy import random
import numpy as np
from cw1 import Question3

@pytest.mark.parametrize('m', [20, 5, 9])
def test_householder_qr(m):
    random.seed(4732*m)
    A = random.randn(m, m)
    A0 = 1*A
    R, Q = Question3.rq(A0)

    # check orthonormality
    assert(np.linalg.norm(np.dot(np.conj(Q.T), Q) - np.eye(m)) < 1.0e-6)
    # check upper triangular
    assert(np.allclose(R, np.triu(R)))
    # check RQ factorisation
    assert(np.linalg.norm(np.dot(R, Q) - A) < 1.0e-6)