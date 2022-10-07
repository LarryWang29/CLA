from re import A
import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    dim = np.shape(A)
    m = dim[0]
    b = np.zeros(m)
    for i in range(m):
        for j in range(dim[1]):
            b[i] += A[i][j] * x[j]
    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    dim = np.shape(A)
    m = dim[0]
    b = np.zeros(m, dtype='complex_')
    for i in range(len(x)):
        b += x[i] * A[:,i]
    return b



def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v1^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    m = len(u1)
    n = len(v1)
    B = np.zeros((m, 2), dtype = 'complex_')
    for i in range(m):
        B[:,0][i] = u1[i]
        B[:,1][i] = u2[i]
    v1c = np.conjugate(v1)
    v2c = np.conjugate(v2)
    C = np.zeros((2, n), dtype = 'complex_')
    for j in range(n):
        C[0,:][j] = v1c[j]
        C[1,:][j] = v2c[j]
    A = B.dot(C)
    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    n = len(u)
    I = np.identity(n, dtype = 'complex_')
    V = np.outer(u, np.conjugate(v))
    alpha = -1 / (1 + np.dot(u,np.conjugate(v)))
    Ainv = I + alpha * V

    return Ainv


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    m = len(xr)
    B = Ahat.copy()
    C = Ahat.copy()
    np.fill_diagonal(C, 0)
    for i in range(m):
        B[i,:][i+1:m] = Ahat[:,i][i+1:m]
        C[:,i][i+1:m] = -1 * Ahat[i,:][i+1:m]
    zr = column_matvec(B, xr) - column_matvec(C, xi)
    zi = column_matvec(C, xr) + column_matvec(B, xi)

    return zr, zi
