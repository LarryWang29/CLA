import cla_utils
import numpy as np
A4 = np.loadtxt('A4.dat')

# Part 1(c)
A4_dot = cla_utils.pure_QR(A4, 1000, 1.0e-5, False, False, True)
# A4_dot = cla_utils.pure_QR(A4, 1000, 1.0e-5)
# print(A4_dot - A4_dot1)

# Part 1(d)
m = np.shape(A4)[0]
evalues = []
i = 0
while i < m:
    if A4_dot[i, i-1] and A4[i,i+1] == 0:
        evalues.append(A4_dot[i,i])
        i += 1 
    else:
        a, b, c, d = A4_dot[i,i], A4_dot[i,i+1], A4_dot[i+1,i], A4_dot[i+1,i+1]
        quad = np.sqrt((a+d)**2 - 4*(a*d-b*c)+0j)
        e1 = (a + d + quad) / 2
        e2 = (a + d - quad) / 2
        evalues.append(e1)
        evalues.append(e2)
        i += 2 # Skip an index for 2 by 2 blocks
evectors = []
for i in evalues:
    evectors.append(cla_utils.inverse_it(A4, np.ones(6), i, 1.0e-5, 1000)[0]) # Obtain the eigenvectors
Errors = []
for i in range(len(evectors)):
    Errors.append(np.linalg.norm(A4 @ evectors[i] - evalues[i]  * evectors[i]))
# print(np.linalg.norm(Errors))
print(evalues)
print(np.linalg.eigvals(A4))
# Part 1(e)