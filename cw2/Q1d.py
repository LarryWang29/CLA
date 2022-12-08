import numpy as np
import cla_utils
from numpy import random
import matplotlib.pyplot as plt


def Error_N(N, k):
    """
    This function calculates the average of relative error for the solve_LUP algorithm 
    on a N*N matrix, whose entries are independently distributed random variables 
    drawn from the uniform distribution on the interval [-1/N, 1/N].


    :param N: Size of the matrix
    :k: Number of iterations

    :returns Avg_Error: Average of relative error after k iterations
    """
    Growth_factor = []
    for i in range(k):
        # Generate a random seed to avoid same values being generated
        random.seed(random.randint(1, 1000))
        # Generate n*n matrix following uniform distribution on [-1/n, 1/n]
        A = random.uniform(-1/N, 1/N, (N,N))
        # Get maximum of a_ij
        a = np.amax(np.abs(A))
        A1 = np.copy(A)
        cla_utils.LUP_inplace(A1)
        U = np.triu(A1)
        # Get maximum of u_ij
        u = np.amax(np.abs(U))
        Growth_factor.append(u/a)
    return np.mean(Growth_factor)

# Generate the array x_arr = (5, 10 ... 95, 100)
x_arr = np.linspace(10, 1000, 50, dtype='int32')
Err_arr = []
# Calculate the average relative error for each element in x_arr
for i in x_arr:
    Err_arr.append(Error_N(i, 2))
# Plot loglog graph for the error
plt.loglog(x_arr, Err_arr)
plt.title('Plot of growth factor against the dimension of matrix')
plt.ylabel('Growth Factor')
plt.xlabel('Dimension of Matrix')
plt.show()
