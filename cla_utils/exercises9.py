import numpy as np
import numpy.random as random
import cla_utils

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.76505141, -0.03865876,  0.42107996],
                     [-0.03865876,  0.20264378, -0.02824925],
                     [ 0.42107996, -0.02824925,  0.23330481]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.76861909,  0.01464606,  0.42118629],
                     [ 0.01464606,  0.99907192, -0.02666057],
                     [ 0.42118629, -0.02666057,  0.23330798]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    iter = 0
    if store_iterations:
        x = np.copy(x0)
        x = x / np.sqrt(np.dot(x, x))
    else:
        x = x0 / np.sqrt(np.dot(x0, x0)) 
    
    while iter < maxit:
        # prev = np.dot(x, A @ x)
        w = A @ x
        x = w / np.sqrt(np.dot(w, w))
        if store_iterations:
            x0 = np.vstack((x0, x))
        lambda0 = np.dot(x, A @ x)
        iter += 1
        r = A @ x - lambda0 * x
        if np.sqrt(np.inner(r, r)) < tol:
            break

    if store_iterations:
        return x0.T, lambda0
    else:
        return x, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """

    x = x0
    m = np.shape(A)[0]
    iter = 0
    if store_iterations:
        x0 = np.array([x0])
        x0 = np.transpose(x0)
        lambda_list = []
    while iter < maxit:
        B = A - mu * np.eye(m)
        w = cla_utils.householder_solve(B, x)
        # print(np.linalg.norm(x - (A - mu * np.eye(m)) @ w1))
        # w = np.linalg.solve(A - mu * np.eye(m), x)
        # print(np.linalg.norm(x - (A - mu * np.eye(m)) @ w1))
        x = w / np.sqrt(np.dot(w, w))
        # print(np.linalg.norm(x))
        l = np.dot(x, A @ x)
        if store_iterations:
            np.hstack((x0, x))
            lambda_list.append(l)
        iter += 1
        r = A @ x - l * x
        if np.linalg.norm(r) < tol:
            break
    
    if store_iterations:
        return x0, np.array(lambda_list)
    else:
        return x, l


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    lambda0 = np.dot(x0, A @ x0)
    m = np.shape(A)[0]
    iter = 0
    while iter < maxit:
        w = np.linalg.inv(A - lambda0 * np.eye(m)) @ x0
        x0 = w / np.sqrt(np.inner(w, w))
        lambda0 = np.dot(x0, A @ x0)
        r = A @ x0 - lambda0 * x0
        iter += 1
        if np.linalg.norm(r) < tol:
            break
    
    return x0, lambda0


def pure_QR(A, maxit, tol):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    Ak = A
    count = 0
    while count < maxit:
        Q, R = cla_utils.householder_qr(Ak)
        Ak_star = R @ Q
        count += 1
        if np.linalg.norm(Ak_star - Ak) < tol:
            break
        Ak = Ak_star

    return Ak

# print(pow_it(get_A3(), np.ones(3), 0.001, 10000, True))
# print(pow_it(get_B3(), np.ones(3), 0.001, 10000))
