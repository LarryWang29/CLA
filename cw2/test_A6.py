import cla_utils
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def An(n):
    A = -1 * np.tri(n) + 2 * np.eye(n)
    A[:,-1] = np.ones(n)
    cla_utils.LUP_inplace(A)
    L = np.eye(n)
    i1 = np.tril_indices(n, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    return A

print(An(6))
n = 60
B = -1 * np.tri(n, dtype=np.float64) + 2 * np.eye(n,dtype=np.float64)
B[:,-1] = np.ones(60,dtype=np.float64)
A60 = An(n)
L = np.eye(n)
i1 = np.tril_indices(n, k=-1)
L[i1] = A60[i1]
U = np.triu(A60)
print(np.linalg.norm(B - L @ U) / np.linalg.norm(B))

Error_array = []
for i in range(20):
    # Set a random seed 
    random.seed(random.randint(1, 1000))
    # Generate a random vector x
    x = random.randn(n)
    # Compute Bx
    y = B @ x
    B1 = np.copy(B)
    # Apply solve_LUP to generate x1
    x1 = cla_utils.solve_LUP(B1, y)
    # Compare the forward error and append the value to an array
    Error_array.append(np.linalg.norm(B @ x1 - y) / np.linalg.norm(y))

# Get the mean of the error array
print(np.mean(Error_array))
# Plot a graph of the errors
plt.plot(np.linspace(1,20,20), Error_array)
plt.ylabel('Relative Error')
plt.xlabel('Iteration index')
plt.show()
