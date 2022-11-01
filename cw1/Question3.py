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
