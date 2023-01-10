import cla_utils
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import Q1efgh

# Part 2(c)
def mat_A(m):
    A = -3 * np.eye(m) +  np.triu(np.tri(m, m, k=1), k=-1)
    return A

sizes = np.linspace(10, 30, 11, dtype='int32')
Error = np.zeros(11)
eval_func = lambda t: -2 + 2 * np.cos(t)
vfunc = np.vectorize(eval_func)
for i in range(11):
    m = sizes[i]
    Am = mat_A(m)
    evalues = cla_utils.pure_QR(Am, 1000, 1.0e-05).diagonal()
    idx = np.linspace(1, m, m) * (np.pi/(m+1))
    evalues1 = vfunc(idx)
    evalues, evalues1 = np.sort(evalues), np.sort(evalues1)
    Error[i] = np.linalg.norm(evalues1 - evalues)
plt.plot(sizes, Error)

# Part 2(d)
fig, ax = plt.subplots(nrows=2, ncols=2)
sizes2 = np.array([25, 50, 75, 100])
for i in range(4):
    m = sizes2[i]
    idx = np.linspace(1, m, m) * (np.pi/(m+1))
    evalues1 = vfunc(idx)
    ax[i // 2, i % 2].hist(evalues1, bins=15)
plt.show()