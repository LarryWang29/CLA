import cla_utils
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
A1 = np.loadtxt('A1.dat')


m = 100 
n = 50
k = 20
random.seed(392*m + 2332*n)
A = random.randn(m, n) # Initialise A
Q, R = cla_utils.householder_qr(A)
x = random.randint(1,m)
zero_id = set(random.randint(1, m, x))
R = [([0] * n if i in zero_id else R[i]) for i in range(m)]
B = Q @ R
results = []
for i in range(k):
    s = 10 ** (-i+2)
    C = B.copy()
    R_hat = cla_utils.householder(C, kmax=None, swap=True, reduced_tol=s)[1]
    results.append(np.shape(R_hat)[0])

x_arr = np.linspace(-2, k-2, 20, endpoint=False)
y_arr = results
plt.title('Computed Matrix Rank against tolerance for m=100, n=50')
plt.xlabel('Negative power of 10')
plt.ylabel('Computed Rank')
plt.plot(x_arr, y_arr)