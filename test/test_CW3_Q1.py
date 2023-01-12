import pytest
import cla_utils
from numpy import random
import numpy as np
from cw3 import Q1efgh

@pytest.mark.parametrize('m', [10, 15, 24, 28])
def test_Q1e(m):
    random.seed(8473*m)
    Z = random.randn(m,m)
    Z = np.triu(Z) # Extract the upper triangular part
    inc = random.randint(2, 4) # Generate a random integer
    for i in range(0,m-1,inc):
        Z[i+1,i] = random.randn(1) # Assign the integer to subdiagonal entries
    random.seed(823*m)
    B = np.random.randn(m,m)
    Q = cla_utils.householder_qr(B)[0] # Generate an unitary matrix
    A = Q.T @ Z @ Q
    evals = Q1efgh.pure_QR_eig(A)
    evecs = np.zeros((m,m), dtype='complex')
    for i in range(m):
        ev = evals[i]
        if np.imag(ev) == 0:
            evecs[:, i] = cla_utils.inverse_it(A, np.ones(m), ev, 1.0e-5, 1000)[0]
        else:
            mur, mui = np.real(ev), np.imag(ev)
            K = np.zeros((2*m, 2*m)) # Construct the matrix B accordingly
            K[:m,m:] = mui * np.eye(m)
            K[m:,:m] = -mui * np.eye(m)
            K[m:,m:] = A
            K[:m,:m] = A
            v_dot = cla_utils.inverse_it(K, np.ones(2*m), mur, 1.0e-5, 1000)[0] # Calculate auxilary vector
            vr, vi = v_dot[:m], v_dot[m:] # Extract corresponding parts to retrieve eigenvector
            evecs[:,i] = vr + 1j*vi # Obtain complex eigenvectors
    Errors = np.zeros(m)
    for i in range(m):
        Errors[i] = np.linalg.norm(A @ evecs[:,i] - evals[i] * evecs[:,i])
    assert(np.linalg.norm(Errors) < 1.0e-4)

@pytest.mark.parametrize('m', [10, 15, 24, 28])
def test_Q1h(m):
    random.seed(123*m)
    B = np.random.randn(m,m)
    A = 1/2 * B @ B.T # Generate a symmetric matrix
    evals = Q1efgh.shifted_QR(A, 1000, 1.0e-5) # Calculate eigenvalues using shifted QR
    error = np.zeros(m) # Create array to store the errors
    for i in range(m):
        evec = cla_utils.inverse_it(A, np.ones(m), evals[i], 1.0e-5, 1000)[0] # Calculate the eigenvector
        error[i] = np.linalg.norm(A @ evec - evals[i] * evec) # Calculate the norm of error
    assert(np.linalg.norm(error) < 1.0e-5)