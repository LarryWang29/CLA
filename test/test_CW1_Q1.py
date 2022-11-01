'''Tests for the first exercise set.'''
import pytest
import cla_utils
from numpy import random
import numpy as np
from cw1 import Question1

@pytest.mark.parametrize('m', [1])
def test_Q1(m):
    def f(x):
        return np.exp(-x ** 2) + np.exp(-(x-1) ** 2)
    x = np.array([1, 2])
    fx = np.array([f(1), f(2)])
    b = Question1.GetCoeffs(x, fx, option='householder')
    assert (np.linalg.norm(np.array([1, 1]) - b) < 1.0e-6)