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


def get_full_Q_and_reducted_R(A):
    m,n = A.shape
    Q,R = householder_qr(A, mode='complete')
    return Q[:,:n],R[:n],Q

def ts_qr(A):
    m, n = A.shape
    h = int(m / 4)
    A1 = A[0:h,:]
    A2 = A[h:h*2,:]
    A3 = A[2*h:3*h,:]
    A4 = A[3*h:4*h,:]
    Q1,R1,U1 = get_full_Q_and_reducted_R(A1)
    Q2,R2,U2 = get_full_Q_and_reducted_R(A2)
    Q3,R3,U3 = get_full_Q_and_reducted_R(A3)
    Q4,R4,U4 = get_full_Q_and_reducted_R(A4)
    Q12,R12,U12 = get_full_Q_and_reducted_R(np.r_[R1, R2])
    Q34,R34,U34 = get_full_Q_and_reducted_R(np.r_[R3, R4])
    Q1234,R1234,U1234 = get_full_Q_and_reducted_R(np.r_[R12,R34])
    L1 = arrange_matrixs_to_diag(Q1, Q2, Q3, Q4) 
    L2 = arrange_matrixs_to_diag(Q12, Q34)
    Q = L1.dot(L2).dot(Q1234)
    reduction_tree = [U1234, U12, U34, U1,U2,U3,U4]
    return Q,R1234, reduction_tree


    
def tiled_qr(A):
    B1 = A[:,0:3]
    Q1,R11, reduction_tree = ts_qr(B1)
    U1234, U12, U34, U1,U2,U3,U4 = reduction_tree
    # Apply QT horizontally across trailing matrix

    A[0:6,3:6] = U1.T.dot(A[0:6,3:6])
    A[6:12,3:6] = U2.T.dot(A[6:12,3:6])
    A[12:18,3:6] = U3.T.dot(A[12:18,3:6])
    A[18:24,3:6] = U4.T.dot(A[18:24,3:6])

    # Apply QT vertically
    _A12 = U12.T.dot(np.r_[A[0:3, 3:6], A[6:9, 3:6]]) # 6 * 3 3 *
    A[0:3, 3:6] = _A12[0:3,:]
    A[6:9, 3:6] = _A12[3:6,:]

    _A34 = U34.T.dot(np.r_[A[12:15, 3:6], A[18:21, 3:6]])
    A[12:15, 3:6] = _A34[0:3,:]
    A[18:21, 3:6] = _A34[3:6,:]

    _A1234 = U1234.T.dot(np.r_[A[0:3, 3:6], A[12:15, 3:6]])
    A[0:3, 3:6] = _A1234[0:3,:]
    A[12:15, 3:6] = _A1234[3:6,:]

    B2 = A[3:,3:6]
    Q2, R22 = householder_qr(B2)
    Q3 = Q2.dot(A[0:3, 3:6])
    # dont' know how to get Q
    Q = Q1
    Q = np.zeros(1)
    R12 = A[0:3, 3:6]
    up = np.c_[R11, R12]

    

    down = np.c_[np.zeros((R22.shape[0], R11.shape[1])), R22]
    R = np.r_[up, down]

    return Q, R

def test_ts_qr():
    A = np.random.random((4 * 6, 3))
    Q,R, tree = ts_qr(A)
    _Q,_R = np.linalg.qr(A)
    assert np.allclose(Q, _Q), 'ts qr Q not match'
    assert np.allclose(R, _R), 'ts qr R not match'
    print('ts qr test passed')

np.random.seed(0)
test_ts_qr()
A = np.random.random((4 * 6, 3 * 2))
_Q,_R = np.linalg.qr(A)
Q,R = tiled_qr(A)

assert np.allclose(R, _R), 'tiled qr R not match'
assert np.allclose(Q, _Q), 'tiled qr Q not match'
