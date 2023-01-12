import cla_utils
import numpy as np
import matplotlib.pyplot as plt
A4 = np.loadtxt('A4.dat')

# Part 1(c)
A4_star = cla_utils.pure_QR(A4, 1000, 1.0e-5)
print(A4_star)

# Part 1(d)
A4_dot = cla_utils.pure_QR(A4, 1000, 1.0e-5, False, False, True)
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
evectors = np.zeros((m,m), dtype='complex')
for i in range(m):
    ev = evalues[i]
    if np.imag(ev) == 0: # Check if the eigenvector has imaginary componenet
        evectors[:,i] = cla_utils.inverse_it(A4, np.ones(6), ev, 1.0e-5, 1000)[0] # Obtain the eigenvectors
    else:
        mur, mui = np.real(ev), np.imag(ev)
        B = np.zeros((2*m, 2*m)) # Construct the matrix B accordingly
        B[:m,m:] = mui * np.eye(m)
        B[m:,:m] = -mui * np.eye(m)
        B[m:,m:] = A4
        B[:m,:m] = A4
        v_dot = cla_utils.inverse_it(B, np.ones(2*m), mur, 1.0e-5, 1000)[0] # Calculate auxilary vector
        vr, vi = v_dot[:m], v_dot[m:] # Extract corresponding parts to retrieve eigenvector
        evectors[:,i] = vr + 1j*vi # Obtain complex eigenvectors
Errors = np.zeros(m)
for i in range(m):
    Errors[i] = np.linalg.norm(A4 @ evectors[:,i] - evalues[i]  * evectors[:,i])
plt.plot(np.linspace(1, m, m), Errors)
plt.title('Value of $||A_{4} v - \lambda v||$')
plt.xlabel('Number of eigenvalue')
plt.ylabel('$||A_{4} v - \lambda v||$')
plt.show()