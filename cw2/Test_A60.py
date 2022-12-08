import cla_utils
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def An(n):
    """
    Generates the n*n matrix An as described by the question

    :param n: dimension of the matrix 

    :return A: the required matrix
    """
    # setting lower triangular part
    A = -1 * np.tri(n) + 2 * np.eye(n)
    A[:,-1] = np.ones(n)
    return A

# LUP on A6 (for part a)
A6 = An(6)
cla_utils.LUP_inplace(A6)
print(A6)

# Calculates the error of LUP in place
n = 60
B = -1 * np.tri(n, dtype=np.float64) + 2 * np.eye(n,dtype=np.float64)
B[:,-1] = np.ones(60,dtype=np.float64)
A60 = An(n)
cla_utils.LUP_inplace(A60)
L = np.eye(n)
i1 = np.tril_indices(n, k=-1)
# Getting lower triangular and upper triangular entries
L[i1] = A60[i1]
U = np.triu(A60)
print(np.linalg.norm(B - L @ U) / np.linalg.norm(B))

# Calculates the error of solve LUP
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
