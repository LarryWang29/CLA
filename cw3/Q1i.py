import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from cw3 import Q1efgh
A4 = np.loadtxt('A4.dat')

# Part 1(i)
# size = 30
# for i in range(1):
#     fig = plt.figure(figsize=(10, 5))
#     A = np.random.randn(size, size)
#     B = 1/2 * A @ A.T
#     A_star, Ak_list = cla_utils.pure_QR(B, 1000, 1.0e-5, False, True)
#     ax1 = plt.subplot(1, 2, 1)
#     Evalues = A_star.diagonal()
#     num = np.shape(Ak_list)[1]
#     iter_num = np.linspace(1, num, num)
#     for j in range(size):
#         Ak_list[j,:] -= Evalues[j]
#         Ak_list[j,:] += 1.0e-17 # Adding a tiny increment to avoid taking log of 0
#         plt.semilogy(iter_num, np.abs((Ak_list[j,:])))
#     ax2 = plt.subplot(1, 2, 2)
#     evalues, Ak_list2 = Q1efgh.shifted_QR(B, 1000, 1.0e-8, True)
#     evalues.reverse()
#     num1 = np.shape(Ak_list2)[1]
#     iter_num1 = np.linspace(1, num1, num1)
#     for k in range(size):
#         Ak_list2[k,:] -= evalues[k]
#         print(Ak_list2)
#         Ak_list2[k,:] += 1.0e-17 # Adding a tiny increment to avoid taking log of 0
#         plt.semilogy(iter_num1, np.abs((Ak_list2[k,:])))
#     plt.show()

# Part 1(h)
print(Q1efgh.shifted_QR(A4, 1000, 1.0e-5))