import cla_utils
import numpy as np
import matplotlib.pyplot as plt
A3 = np.loadtxt('A3.dat')

# Part 1(a)
A_star, Ak_list, AS_norms = cla_utils.pure_QR(A3, 1000, 1.0e-5, True, True)
iter_len = len(AS_norms)
iter_num = np.linspace(1, iter_len, iter_len)
plt.figure(0)
plt.semilogy(iter_num, AS_norms)
plt.xlabel('iteration number')
plt.ylabel('log of $||A_{S}||$')
# plt.show()

Evalues = A_star.diagonal() # Extract the eigenvalues
print(Evalues)
Evectors = [] # Create empty list to store the eigenvectors
for i in Evalues:
    Evectors.append(cla_utils.inverse_it(A3, np.ones(6), i, 1.0e-5, 1000)[0]) # Obtain the eigenvectors
Errors = []
for i in range(len(Evectors)):
    Errors.append(np.linalg.norm(A3 @ Evectors[i] - Evalues[i]  * Evectors[i]))

# Part 1(b)
plt.figure(1)
for i in range(len(Evalues)):
    Ak_list[i,:] -= Evalues[i]
    Ak_list[i,:] += 1.0e-17 # Adding a tiny increment to avoid taking log of 0
    plt.semilogy(iter_num, np.abs((Ak_list[i,:])), label = "N = {}".format(i))
plt.legend()
# plt.show()
