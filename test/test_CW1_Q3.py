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

@pytest.mark.parametrize('m, n, p', [(20, 15, 7), (40, 20, 13), (87, 40, 9)])
def test_factorisation(m, n, p):
    random.seed(3686*m + 1367*n + 5684*p)
    A = random.randn(n, m)
    B = random.randn(n, p)
    Q, S, R, U = Question3.sim_fac(A, B)
    # Obtain R11 and S11
    R11 = R[:,-n:]
    S11 = S[:p,:]
    # check orthonormality 
    assert(np.linalg.norm(np.dot(np.conj(Q.T), Q) - np.eye(n)) < 1.0e-6)
    assert(np.linalg.norm(np.dot(np.conj(U.T), U) - np.eye(m)) < 1.0e-6)
    # check upper triangular 
    assert(np.allclose(R11, np.triu(R11)))
    assert(np.allclose(S11, np.triu(S11)))
    # check simultaneous factorisation
    assert(np.linalg.norm(np.dot(Q.T, B) - S)  < 1.0e-6)
    assert(np.linalg.norm(np.dot(Q.T @ A, U) - R) < 1.0e-6)

@pytest.mark.parametrize('m, n, p', [(20, 15, 7), (40, 20, 13), (87, 40, 9)])
def test_sim_solver(m, n, p):
    random.seed(1236*m + 4224*n + 2368*p)
    A = random.randn(m, n)
    B = random.randn(p, n)
    b = random.randn(m)
    d = random.randn(p)
    x = Question3.constrained_ls(A, B, b, d)
    Q, S, R, U = Question3.sim_fac(A.T, B.T)
    S1 = S[:p, :p]
    y1 = np.linalg.inv(S1.T) @ d
    R2 = R[-(n-p):, -(n-p):]
    C = (U.T @ b)[-(n-p):]
    R1 = R[:p, -(n-p):]
    b_hat = C - R1.T @ y1
    y2 = (Q.T @ x)[-(n-p):]
    # check solution to the modified least squares problem
    assert(np.linalg.norm(np.dot(R2, np.dot(R2.T, y2) - b_hat)) < 1.0e-6)
    # check Bx = d
    assert(np.linalg.norm(np.dot(B, x) - d) < 1.0e-06)