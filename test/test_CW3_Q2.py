import pytest
import cla_utils
from numpy import random
import numpy as np
from cw3 import Q2

@pytest.mark.parametrize('m', [20, 40, 60])
def test_2a(m):
    A = random.randn(m, m) # Generate a matrix A
    b = random.randn(m) # Generate a vector b
    cla_utils.GMRES(A, b, 1000, 1.0e-05, Q2.get_callback(b))
    Errors = np.loadtxt('cw3/Errors.dat')
    assert(len(Errors) > 0) # Check that the .dat file is not empty
    assert(Errors[-1] < 1.0e-7) # Check that the final value is small