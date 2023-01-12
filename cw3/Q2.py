import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from cw3 import Q1efgh

# Part 2(a)
def get_callback(x_sol):
    def callback(x):
        error = np.linalg.norm(x-x_sol) # Compute error
        file = open('cw3/Errors.dat', 'a')
        file.write(str(error))
        file.write('\n')
        file.close
    return callback

# Part 2(c)
def mat_A(m):
    """
    Forms a matrix of the form A as defined in the question.

    :param m: dimension of the matrix

    :return A: matrix in the required form
    """
    A = -3 * np.eye(m) +  np.triu(np.tri(m, m, k=1), k=-1)
    return A

plt.figure(0)
sizes = np.linspace(5, 100, 20, dtype='int32') # Array of different sizes
Error = np.zeros(20) # Create an array to store the errors
eval_func = lambda t: -2 + 2 * np.cos(t) # Create function that generates the eigenvalues
vfunc = np.vectorize(eval_func)
for i in range(20):
    m = sizes[i]
    Am = mat_A(m)
    evalues = Q1efgh.shifted_QR(Am, 1000, 1.0e-05) # Compute eigenvalues using QR
    idx = np.linspace(1, m, m) * (np.pi/(m+1))
    evalues1 = vfunc(idx) # Compute eigenvalues using formula
    evalues, evalues1 = np.sort(evalues), np.sort(evalues1)
    Error[i] = np.linalg.norm(evalues1 - evalues) # Calculate error
plt.plot(sizes, Error)
plt.title('Plot of norm of differences array against m')
plt.xlabel('Dimension of A')
plt.ylabel('Norm of the differences array')


# Part 2(d)
fig, ax = plt.subplots(nrows=2, ncols=2)
sizes2 = np.array([50, 100, 150, 200])
for i in range(4):
    m = sizes2[i]
    idx = np.linspace(1, m, m) * (np.pi/(m+1))
    evalues1 = vfunc(idx)
    ax[i // 2, i % 2].hist(evalues1, bins=20)
    ax[i // 2, i % 2].set_title("m = {}".format(m))
plt.show()

# Part 2(e)
fig, ax = plt.subplots(nrows=2, ncols=2)
dims = np.array([50, 100, 150, 200]) # Check for large m
for i in range(4):
    m = dims[i]
    A = mat_A(m)
    b = np.random.randn(m)
    c = A @ b
    # r0 = np.linalg.norm(A @ b - d)
    cla_utils.GMRES(A, c, 1000, 1.0e-5, callback = get_callback(b))
    Errors = np.loadtxt('cw3/Errors.dat')
    n = len(Errors)
    vals = np.linspace(1, n, n)
    ax[i // 2, i % 2].plot(vals, Errors)
    ax[i // 2, i % 2].set_title('Plot of $R_{n}$ against iteration index, ' + "m = {}".format(m))
    ax[i // 2, i % 2].set_xlabel('Iteration index')
    ax[i // 2, i % 2].set_ylabel('Magnitude of Residual')
plt.tight_layout()
plt.show()