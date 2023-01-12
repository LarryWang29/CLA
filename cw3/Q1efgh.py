import cla_utils
import numpy as np
import matplotlib.pyplot as plt
# A3 = np.loadtxt('A3.dat')

# Part 1(e)
def pure_QR_eig(A):
    """
    :param A: the m*m matrix to find the eigenvalues for

    :return evalues: a list of the eigenvalues
    """
    A_dot = cla_utils.pure_QR(A, 1000, 1.0e-05, False, False, True)
    m = np.shape(A)[0]
    evalues = []
    idx = np.argwhere(np.abs(np.diag(A_dot, -1)) > 1.0e-5).flatten()
    for i in range(m):
        if i in idx:
            a, b, c, d = A_dot[i,i], A_dot[i,i+1], A_dot[i+1,i], A_dot[i+1,i+1]
            quad = np.sqrt((a+d)**2 - 4*(a*d-b*c)+0j)
            e1 = (a + d + quad) / 2
            e2 = (a + d - quad) / 2
            evalues.append(e1)
            evalues.append(e2)
        else:
            if i-1 in idx:
                continue
            else:
                evalues.append(A_dot[i,i])
    return evalues

def complex_eigvec(A, ev, tol, maxit):
    """
    Computes the inverse iteration for a complex shift
    :param A: an mxm numpy array
    :param ev: the eigenvalue to compute eigenvector for
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations

    :return v: the eigenvector for the corresponding eigenvalue
    """
    mur, mui = np.real(ev), np.imag(ev)
    n = np.shape(A)[0]
    B = np.zeros((2*n, 2*n)) # Construct the matrix B accordingly
    B[:n,n:] = mui * np.eye(n)
    B[n:,:n] = -mui * np.eye(n)
    B[n:,n:] = A
    B[:n,:n] = A
    v_dot = cla_utils.inverse_it(B, np.ones(2*n), mur, tol, maxit)[0] # Calculate auxilary vector
    vr, vi = v_dot[:n], v_dot[n:] # Extract corresponding parts to retrieve eigenvector
    v = vr + 1j*vi # Obtain complex eigenvectors
    return v

sizes = np.linspace(3, 15, 5, dtype='int32')
Error_arr = np.zeros(5)
np.random.seed(1283)
for i in range(5):
    n = sizes[i]
    A = np.random.randn(n, n) # Generate random non-symmetric matrices
    evals = pure_QR_eig(A) # Apply general pure QR to obtain eigenvalues
    evectors = np.zeros((n,n), dtype='complex')
    for p in range(n):
        ev = evals[p]
        if np.imag(ev) == 0: # Check if the eigenvector has imaginary componenet
            evectors[:,p] = cla_utils.inverse_it(A, np.ones(n), ev, 1.0e-5, 1000)[0] # Obtain the eigenvectors
        else:
            v = complex_eigvec(A, ev, 1.0e-5, 1000)
            evectors[:,p] = v # Obtain complex eigenvectors
    Errors = np.zeros(n)
    for k in range(n):
        Errors[k] = np.linalg.norm(A @ evectors[:,k] - evals[k] * evectors[:,k])
    Error_arr[i] = np.linalg.norm(Errors) # Keep track of error at each iteration
plt.semilogy(sizes, Error_arr)
plt.title('Convergence of non-symmetric matrices')
plt.xlabel('Dimension of the matrix')
plt.ylabel('Norm of Error array')
plt.show()


# Part 1(h)
def shifted_QR(A, maxit, tol, store_diags=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    Ak = A
    cla_utils.hessenberg(Ak)
    count = 0
    evalues = []
    m = np.shape(Ak)[0]
    if store_diags:
        diags = np.empty((m, 0))
    while count < maxit and m > 1:
        mu = Ak[m-1,m-1]
        Q, R = cla_utils.householder_qr(Ak[:m, :m] - mu * np.eye(m))
        Ak[:m,:m] = R @ Q + mu * np.eye(m)
        count += 1
        if np.abs(Ak[m-2,m-1]) < tol:
            evalues.append(Ak[m-1,m-1])
            m -= 1
        if store_diags:
            Ak_diags = Ak.diagonal()
            diags = np.append(diags, np.array([Ak_diags]).T, axis=1)
    evalues.append(Ak[0,0])
    if store_diags:
        return evalues, diags
    return evalues

