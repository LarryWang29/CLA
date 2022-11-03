from http.client import NotConnected
from tkinter import Y
import cla_utils
import math
import numpy as np
from numpy import random

def rq(A):
    A_hat = np.flip(A, 0) # Flips rows of A
    Q_hat, R_hat = cla_utils.householder_qr(np.transpose(A_hat)) # QR Factorisation of A_hat
    Q = np.flip(Q_hat.T, 0) # Obtain Q
    R = np.flip(np.transpose(np.flip(R_hat, 0)), 0) # Obtain R

    return R, Q

def sim_fac(A, B):
    Q, S = cla_utils.householder_qr(B) # QR Factorisation of B
    R, U = rq(Q.T @ A) # RQ Factorisation of QtA
    return Q, S, R, U.T

def constrained_ls(A, B, b, d):
    m, n, p = np.shape(A)[0], np.shape(A)[1], np.shape(B)[0]
    Q, S, R, U = sim_fac(A.T, B.T)
    S1 = S[:p, :p] # Obtain the S11 matrix used in the proof
    y1 = np.linalg.inv(S1.T) @ d # Obtain vector y
    R2 = R[-(n-p):, -(n-p):] # Truncate R to obtain submatrix
    C = (U.T @ b)[-(n-p):] # Calculate vector c
    R1 = R[:p, -(n-p):] # Truncate R to obtain the other submatrix
    b_hat = C - R1.T @ y1 # Calculating the vector used in least squared problem
    y2 = cla_utils.householder_ls(R2.T, b_hat) # Use least squared to solve for y2
    y = np.concatenate([y1, y2])
    x = Q @ y # Obtain x
    return x

