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
        x = w / np.linalg.norm(w)
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
    x = x0
    l = np.dot(x0, A @ x0)
    m = np.shape(A)[0]
    iter = 0
    if store_iterations:
        x0 = np.array([x0])
        x0 = np.transpose(x0)
        lambda_list = [] 
    while iter < maxit:
        w = np.linalg.inv(A - l * np.eye(m)) @ x
        x = w / np.sqrt(np.inner(w, w))
        l = np.dot(x, A @ x)
        r = A @ x0 - l * x
        if store_iterations:
            np.hstack((x0, x))
            lambda_list.append(l)
        iter += 1
        if np.linalg.norm(r) < tol:
            break
    
    if store_iterations:
        return x0, np.array(lambda_list)
    else:
        return x, l


def pure_QR(A, maxit, tol, store_AS_norm=False, store_Ak_diag=False, non_sym=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    m = np.shape(A)[0]
    Ak = A
    count = 0
    if store_AS_norm:
        AS_norm = []
    if store_Ak_diag:
        diags = np.empty((m, 0))
    while count < maxit:
        Q, R = cla_utils.householder_qr(Ak)
        Ak_star = R @ Q
        count += 1
        if store_Ak_diag:
            Ak_diags = Ak_star.diagonal()
            diags = np.append(diags, np.array([Ak_diags]).T, axis=1)
        if store_AS_norm:
            AS_norm.append(np.linalg.norm(np.tril(Ak_star, k=-1)))
        if non_sym: # For general non symmetric matrices
            Tol_list = [] # Create list to store specific subdiagonal entries
            for i in range(m-1): # Iterate through all subdiagonal entries
                if i != m-2: # Check conditions for all but last entry in the subdiagonal
                    if np.abs(Ak_star[i+1, i]) < tol: # Check if current entry is zero
                        Tol_list.append(Ak_star[i+1, i])
                        continue
                    else:
                        if np.abs(Ak_star[i+2,i+1]) < tol: # Check if next entry is zero
                            continue # Continue if next entry is zero
                        else:
                            break # break out of loop if successive entries are non-zero
                else:
                    if np.abs(Ak_star[i+1, i]) < tol: # append the last entry; no need to check for next entry
                        Tol_list.append(Ak_star[i+1, i])                         
                    if np.linalg.norm(Tol_list) < tol: # Check norm of list consisting of "zero" terms are below tolerance
                        if store_Ak_diag and store_AS_norm: # Different return options
                            return Ak_star, diags, AS_norm
                        if store_Ak_diag:
                            return Ak_star, diags
                        if store_AS_norm:
                            return Ak_star, AS_norm
                        return Ak_star
        else:
            if (np.linalg.norm(Ak_star[np.tril_indices(m, -1)])/m**2 < tol):
                break
        Ak = Ak_star
    if store_Ak_diag and store_AS_norm:
        return Ak_star, diags, AS_norm
    if store_Ak_diag:
        return Ak_star, diags
    if store_AS_norm:
        return Ak_star, AS_norm
    return Ak_star

# print(pow_it(get_A3(), np.ones(3), 0.001, 10000, True))
# print(pow_it(get_B3(), np.ones(3), 0.001, 10000))
