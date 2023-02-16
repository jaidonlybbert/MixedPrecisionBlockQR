import numpy as np

def wy_representation(V, B) -> np.ndarray:
    """For the factored form of Q = {Q1}{Q2}{...}{Qn} where Qn = {Im} - {beta_j}{v_j}{v_j}^T
        and the factors {v_j, b_j} are stored as V = [v0, v1, ..., vn], B = [beta_1, beta_2, ..., beta_n],
        the W and Y factors such that Q = Im - {W}{Y}^T can be calculated from V, and B.

        Since Y = V, only W is returned.

        Reference:
             Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins University Press. Pg. 239. Algorithm 5.1.2

    Args:
        V (np.ndarray): matrix V containing n householder vectors [v0, v1, ..., vn]
        B (np.ndarray): matrix B containing n coefficients [beta0, beta1, ..., betan] derived from factored forms 
    """

    m,r = len(V[0]),len(V)
    Y = np.array(V[0]).reshape(-1, 1)
    W = np.array(B[0]*V[0]).reshape(-1, 1)

    for i in range(1, r):
        z = B[i] * (np.identity(m) - W.dot(Y.T)).dot(V[i])
        W = np.c_[W,z]
        Y = np.c_[Y,V[i]]
    
    return W, Y

