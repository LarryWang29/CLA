import cla_utils
import numpy as np
A3 = np.loadtxt('A3.dat')

# Part 1(e)
def pure_QR_eig(A):
    """
    :param A: the m*m matrix to find the eigenvalues for

    :return evalues: a list of the eigenvalues
    """
    A_dot = cla_utils.pure_QR(A, 1000, 1.0e-05)
    m = np.shape(A)[0]
    evalues = []
    i = 0
    while i < m:
        if A_dot[i, i-1] and A_dot[i,i+1] == 0:
            evalues.append(A_dot[i,i])
            i += 1 
        else:
            a, b, c, d = A_dot[i,i], A_dot[i,i+1], A_dot[i+1,i], A_dot[i+1,i+1]
            quad = np.sqrt((a+d)**2 - 4*(a*d-b*c)+0j)
            e1 = (a + d + quad) / 2
            e2 = (a + d - quad) / 2
            evalues.append(e1)
            evalues.append(e2)
            i += 2 # Skip an index for 2 by 2 blocks
    return evalues

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
    # k = m
    while count < maxit and m > 1:
        mu = Ak[m-1,m-1]
        Q, R = cla_utils.householder_qr(Ak[:m, :m] - mu * np.eye(m))
        Ak_star = R @ Q + mu * np.eye(m)
        count += 1
        if np.abs(Ak_star[m-2,m-1]) < tol:
            evalues.append(Ak_star[m-1,m-1])
            m -= 1
        if store_diags:
            Ak_diags = Ak_star.diagonal()
            diags = np.append(diags, np.array([Ak_diags]).T, axis=1)
        Ak = Ak_star
    evalues.append(Ak[0,0])
    if store_diags:
        return evalues, diags
    return evalues

# print(shifted_QR(A3, 1000, 1.0e-05))
# print(np.linalg.eigvals(A3))
