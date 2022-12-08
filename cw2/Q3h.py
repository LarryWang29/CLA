from cw2 import Q3
import numpy as np
import matplotlib.pyplot as plt

# Keep s fixed
s = 1

rho = np.linspace(0.5, 2, 4)
nu = np.linspace(0.5, 2, 4)
N = np.linspace(10, 35, 6, dtype='int32')
LN = len(N)


def fix_nu(nu):
    """
    Calculate iteration counts for different rho and N values while keeping 
    nu fixed

    :param nu: nu value to fix

    :returns: a list of lists. Each list consists of iteration counts for 
    different rho values while keeping N fixed
    """
    iter_lists = [[] for _ in range(LN)]
    for i in rho:
        for j in range(LN):
            iter_lists[j].append(Q3.iterative_solver(N[j], 1, i, nu, 1.0e-6)[1])
    return iter_lists

# Plotting two plots side by side 
fig = plt.figure(figsize=(10, 5))

# ax1 = plt.subplot(1, 2, 1)
Y1 = fix_nu(nu[0])
for k in range(LN):
    Ni = N[k]
    plt.plot(rho, Y1[k], label = "N = {}".format(Ni))
    plt.title('$nu = 0.1$')
    plt.legend()
    plt.xlabel('value of $rho$')
    plt.ylabel('Number of iterations')

ax2 = plt.subplot(1, 2, 2)
Y2 = fix_nu(nu[1])
for k in range(LN):
    Ni = N[k]
    plt.plot(rho, Y2[k], label = "N = {}".format(Ni))
    plt.title('$nu = 1$')
    plt.legend()
    plt.xlabel('value of $rho$')
    plt.ylabel('Number of iterations')
plt.show()