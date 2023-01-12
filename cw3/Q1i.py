import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from cw3 import Q1efgh
A4 = np.loadtxt('A4.dat')

# Part 1(i)
random.seed(91)
size = np.linspace(50, 100, 11, dtype='int32')
Conv_iter1 = []
Conv_iter2 = []
for i in range(11):
    A = np.random.randn(size[i], size[i])
    B = 1/2 * A @ A.T # Generate a symmetric matrix of a given size
    # Obtain the diagonal entries at each iteration for pure QR
    Ak_list = cla_utils.pure_QR(B, 1000, 1.0e-5, False, True)[1]
    its = np.shape(Ak_list)[1]
    for j in range(its-1):
        # Compare the magnitude of the differences between diagonal entries
        # in successive iterations
        if np.linalg.norm(Ak_list[:,j] - Ak_list[:,j+1]) < 1.0e-3:
            Conv_iter1.append(j+2) # Adding 2 because we are comparing with the next iteration
            break
        else:
            continue
    # Obtain the diagonal entries at each iteration for shifted QR
    Ak_list2 = Q1efgh.shifted_QR(B, 1000, 1.0e-5, True)[1]
    its2 = np.shape(Ak_list2)[1]
    for k in range(its2-1):
        # Compare the magnitude of the differences between diagonal entries
        # in successive iterations
        if np.linalg.norm(Ak_list2[:,k] - Ak_list2[:,k+1]) < 1.0e-3:
            Conv_iter2.append(k+2)
            break
        else:
            continue

# Plot to demonstrate convergence rate of Pure QR and Shifted QR 
plt.title('Iteration counts against size of matrix')
plt.plot(size, Conv_iter1, label='Pure QR algorithm')
plt.plot(size, Conv_iter2, label='Shifted QR algorithm')
plt.xlabel('Dimension of the matrix')
plt.ylabel('Iteration counts')
plt.legend()
plt.show()