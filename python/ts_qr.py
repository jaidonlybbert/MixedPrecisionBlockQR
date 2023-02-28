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
    reduction_tree = [U1234, U12, U34, U1,U2,U3,U4]
    return Q,R1234, reduction_tree
    
def tiled_qr(A):
    B1 = A[:,0:3]
    Q1,R1, reduction_tree = ts_qr(B1)
    U1234, U12, U34, U1,U2,U3,U4 = reduction_tree
    # Apply QT horizontally across trailing matrix
    A[0:6,3:6] = U1.T.dot(A[0:6,3:6])
    A[6:12,3:6] = U2.T.dot(A[6:12,3:6])
    A[12:18,3:6] = U3.T.dot(A[12:18,3:6])
    A[18:24,3:6] = U4.T.dot(A[18:24,3:6])

    # Apply QT vertically
    A0 = np.r_[A[0:3, 3:6], A[6:9, 3:6]]
    A[0:3, 3:6] = U12.T @ A0
    A1 = np.r_[A[12:15, 3:6], A[18:21, 3:6]]
    A[12:15, 3:6] = U34.T @ A1

    A3 = np.r_[A[0:3, 3:6], A[12:15, 3:6]]
    A[0:3, 3:6] = U1234.T @ A3

    B2 = A[:,3:6]
    Q2,R2, reduction_tree = ts_qr(B2)
    Q = np.c_[Q1,Q2]
    R = np.c_[R1, R2]

    down = np.c_[np.zeros((R2.shape[0], R1.shape[1])), R2]
    R = np.r_[R, down]

    return Q, R


A = np.random.random((4 * 6, 3 * 2))

tiled_qr(A)
# B1 = A[:,0:3]
# Q, R = ts_qr(B1)
# _Q,_R = np.linalg.qr(B1)
# print(np.allclose(_R, R))
# print(np.allclose(Q, _Q))