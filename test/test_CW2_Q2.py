import pytest
from numpy import random
import numpy as np
from cw2 import Q2

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_MGS_ls(m, n):
    random.seed(8473*m + 9283*n)
    A = random.randn(m, n)
    b = random.randn(m)
    A0 = np.copy(A)
    # Apply MGS solve ls to A0
    x = Q2.MGS_solve_ls(A0, b)
    #check normal equation residual
    assert(np.linalg.norm(np.dot(A.T, np.dot(A, x) - b)) < 1.0e-6)

@pytest.mark.parametrize('m, n', [(3, 2), (20, 7), (40, 13), (87, 9)])
def test_modified_MGS_ls(m, n):
    random.seed(8473*m + 9283*n)
    A = random.randn(m, n)
    b = random.randn(m)
    A0 = np.copy(A)
    # Apply Modified MGS solve ls to A0
    x = Q2.MGS_solve_ls_modified(A0, b)
    #check normal equation residual
    assert(np.linalg.norm(np.dot(A.T, np.dot(A, x) - b)) < 1.0e-6)