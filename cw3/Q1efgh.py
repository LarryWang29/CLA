import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import cw3
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

A = np.array([[3, 0, 0], [0, 1, -1], [0, 5, -1]])
C = np.random.randn(3,3)
Q = cla_utils.householder_qr(C)[0]
B = Q @ A @ Q.T
evals = np.sort(pure_QR_eig(B))
print(evals)

D = np.array([[2, 0, 0, 0, 0, 0], 
[0, 3, -3, 0, 0, 0], [0, 6, -3, 0, 0, 0],
[0, 0, 0, 7, 0, 0], [0, 0, 0, 0, 1, 1],
[0, 0, 0, 0, -1, 1]])
P = np.random.randn(6,6)
Q1 = cla_utils.householder_qr(P)[0]
K = Q1 @ D @ Q1.T
evals = np.sort(pure_QR_eig(K))
print(evals)

# Part 1(h)
def shifted_QR(A, maxit, tol, store_diags=False, store_iter=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param store_diags: if True, store the diagonal values of A at each iteration
    and return them at the end
    :param store_iter: if True, store the iteration number at which an eigenvalue 
    converges

    :return evalues: a list of the eigenvalues
    """
    Ak = A
    cla_utils.hessenberg(Ak)
    count = 0
    evalues = []
    m = np.shape(Ak)[0]
    if store_diags:
        diags = np.empty((m, 0))
    if store_iter:
        iter_num = []
    while count < maxit and m > 1:
        mu = Ak[m-1,m-1]
        Q, R = cla_utils.householder_qr(Ak[:m, :m] - mu * np.eye(m))
        Ak[:m,:m] = R @ Q + mu * np.eye(m)
        count += 1
        if store_diags:
            Ak_diags = Ak.diagonal()
            diags = np.append(diags, np.array([Ak_diags]).T, axis=1)
        if np.abs(Ak[m-2,m-1]) < tol:
            if store_iter:
                iter_num.append(count)
            evalues.append(Ak[m-1,m-1])
            m -= 1
    evalues.append(Ak[0,0])
    if store_iter:
        iter_num.append(count)
        return evalues, iter_num
    if store_diags:
        return evalues, diags
    return evalues

# Calculate the eigenvalues using shifted QR and keep note of iteration number for convergence 
# of each eigenvector
evals, iters = shifted_QR(A3, 1000, 1.0e-5, False, True) 
print(iters) # Print out number of iterations
# Calculate eigenvalues using pure QR
evals1 = cla_utils.pure_QR(A3, 1000, 1.0e-05).diagonal()
# Calculate the norm of differences between the eigenvalues
print(np.linalg.norm(np.sort(evals) - np.sort(evals1)))