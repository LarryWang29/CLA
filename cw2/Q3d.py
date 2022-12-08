import numpy as np
from cw2 import Q3
import matplotlib.pyplot as plt

# Plotting iteration times for banded and original algorithm against N
x_arr = np.linspace(2, 40, 20, dtype='int32')
y_arr, y_arr1 = [], []
for i in x_arr:
    y_arr.append(Q3.sim(i, 1, 'Banded', True))
    y_arr1.append(Q3.sim(i, 1, 'Original', True))
plt.plot(x_arr, y_arr)
plt.plot(x_arr, y_arr1)
plt.show()