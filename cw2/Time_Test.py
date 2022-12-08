import numpy as np
from cw2 import Q3
import matplotlib.pyplot as plt

# Generate different N
x_arr = np.linspace(2, 40, 20, dtype='int32')
y_arr, y_arr1 = [], []
for i in x_arr:
    # Calculate the runtime of both algorithms
    y_arr.append(Q3.sim(i, 1, 'Banded', True))
    y_arr1.append(Q3.sim(i, 1, 'Original', True))
# Plot Runtime against N on the same graph
plt.plot(x_arr, y_arr, label='Banded solve LU')
plt.plot(x_arr, y_arr1, label='solve LU')
plt.xlabel("N (dimension of array u)")
plt.ylabel("Runtime in seconds")
plt.title('Run time for both banded and original solve LU with varying N')
plt.legend()
plt.show()