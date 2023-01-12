import cla_utils
import numpy as np
import matplotlib.pyplot as plt
A3 = np.loadtxt('A3.dat')

# Part 1(a)
# A_star is the converged matrix, AS_norms is a list consisting of the matrix norm of A_S at each iteration
# Ak_list is used in part (b)
A_star, Ak_list, AS_norms = cla_utils.pure_QR(A3, 1000, 1.0e-5, True, True)
iter_len = len(AS_norms)
iter_num = np.linspace(1, iter_len, iter_len) # Array of iteration numbers
plt.figure(0)
plt.semilogy(iter_num, AS_norms)
plt.title('log of $||A_{S}||$ against iteration number')
plt.xlabel('iteration number')
plt.ylabel('log of $||A_{S}||$') # Plot norm of A_S against iteration number


Evalues = A_star.diagonal() # Extract the eigenvalues
n = len(Evalues) 
Evectors = np.zeros((n,n)) # Create array to store the eigenvectors
for i in range(n):
    ev = Evalues[i]
    Evectors[:,i] = cla_utils.inverse_it(A3, np.ones(6), ev, 1.0e-5, 1000)[0] # Obtain the eigenvectors
Errors = np.zeros(n)
for j in range(n):
    Errors[j] = (np.linalg.norm(A3 @ Evectors[:,j] - Evalues[j]  * Evectors[:,j])) # Obtain the error
plt.figure(1)
plt.plot(np.linspace(1, n, n), Errors)
plt.title('Value of $||A_{3} v - \lambda v||$')
plt.xlabel('Number of eigenvalue')
plt.ylabel('$||A_{3} v - \lambda v||$')
# plt.show()

# Part 1(b)
# Ak_list is a list that stores the diagonal entries of Ak at each iteration
plt.figure(2)
for i in range(n):
    Ak_list[i,:] -= Evalues[i] # Calculate the difference between diagonal entries and eigenvalues
    Ak_list[i,:] += 1.0e-17 # Adding a tiny increment to avoid taking log of 0
    plt.semilogy(iter_num, np.abs((Ak_list[i,:])), label = "N = {}".format(i+1))
plt.title('Convergence of diagonal entries to eigenvalues')
plt.xlabel('Iteration number')
plt.ylabel('Absolute difference between diagonal entry and eigenvalue')
plt.legend()
plt.show()
