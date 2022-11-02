from http.client import NotConnected
from tkinter import Y
import cla_utils
import math
import numpy as np
from numpy import random

def rq(A):
    A_hat = np.flip(A, 0)
    Q_hat, R_hat = cla_utils.householder_qr(np.transpose(A_hat))
    Q = np.flip(Q_hat.T, 0)
    R = np.flip(np.transpose(np.flip(R_hat, 0)), 0)

    return R, Q

def sim_fac(A, B):
    Q, S = cla_utils.householder_qr(B)
    R, U = rq(Q.T @ A)
    return Q, S, R, U.T

def constrained_ls(A, B, b, d):
    m, n, p = np.shape(A)[0], np.shape(A)[1], np.shape(B)[0] 
    Q, S, R, U = sim_fac(A.T, B.T)
    S1 = S[:p, :p]
    y1 = np.linalg.inv(S1.T) @ d
    R2 = R[-(n-p):, -(n-p):]
    C = (U.T @ b)[-(n-p):]
    R1 = R[:p, -(n-p):]
    b_hat = C - R1.T @ y1
    y2 = cla_utils.householder_ls(R2.T, b_hat)
    y = np.concatenate([y1, y2])
    x = Q @ y
    return x

