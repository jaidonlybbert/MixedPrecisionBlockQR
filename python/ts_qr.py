import numpy as np
from qr import householder_qr 


def arrange_to_diag(A,B):
    m1,n1 = A.shape
    m2,n2 = B.shape
    down_left = np.zeros((m2, n1))
    top_right = np.zeros((m1, n2))
    return np.c_[np.r_[A,down_left], np.r_[top_right, B]]

def arrange_matrixs_to_diag(*matrixs):
    current, i, n = matrixs[0],1,len(matrixs)
    while i < n:
       current = arrange_to_diag(current, matrixs[i])
       i += 1
    return current


def ts_qr(A):
    m, n = A.shape
    h = int(m / 4)
    A1 = A[0:h,:]
    A2 = A[h:h*2,:]
    A3 = A[2*h:3*h,:]
    A4 = A[3*h:4*h,:]
    U1,R1 = householder_qr(A1)
    U2,R2 = householder_qr(A2)
    U3,R3 = householder_qr(A3)
    U4,R4 = householder_qr(A4)
    U12,R12 = householder_qr(np.r_[R1, R2])
    U34,R34 = householder_qr(np.r_[R3, R4])
    U1234,R1234 = householder_qr(np.r_[R12,R34])
    L1 = arrange_matrixs_to_diag(U1, U2, U3, U4) 
    L2 = arrange_matrixs_to_diag(U12, U34)
    Q = L1.dot(L2).dot(U1234)
    return Q,R1234


    
# def tiled_qr(A):
#     B1 = A[:,0:3]
#     return ts_qr(B1)



A = np.random.random((4 * 6, 3 * 3))
B1 = A[:,0:3]
Q, R = ts_qr(B1)
_Q,_R = np.linalg.qr(B1)
print(np.allclose(_R, R))
print(np.allclose(Q, _Q))