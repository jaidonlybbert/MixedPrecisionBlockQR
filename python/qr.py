import numpy as np
from numpy import linalg

def householder(x, sign=-1):
    sigma = np.linalg.norm(x)
    e = np.zeros_like(x)
    e[0] = 1
    u = x - sign * sigma * e
    return u / np.linalg.norm(u)

def qr(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    m,n = A.shape
    Q,H = np.identity(m), np.identity(m)
    
    for i in range(n):
        # skip last transformation if is sqare matrix
        if m == n and i == n - 1:
            break
        Ai = A[i:, i:]
        # to match numpy implementation
        sign = 1 if i == n - 1 else -1
        v = householder(Ai[:, 0], sign)
        H_hat = np.identity(m - i) - 2 * np.outer(v, v)
        H = np.pad(H_hat, (i, 0), 'constant')
        H[np.diag_indices(i)] = 1
        Q = Q.dot(H)
        A = H.dot(A)
    return Q, A

def testHouseholder():
    raw = np.array([0, 0, 2])
    v = householder(raw)
    assert np.allclose(v, np.array(
        [1, 0, 1]) / (2 ** (1/2))), "v is not correct"
    H = np.identity(3) - 2 * np.outer(v, v)
    reflected = H.dot(raw)
    assert np.allclose(reflected, np.array(
        [-2, 0, 0])), 'reflected is not correct'

def testQR():

    tests = [
        [
            [1,2,3],
            [4,5,6],
            [7,8,7],
            [4,2,3],
            [4,2,2]
        ],
        [
            [0, 3, 1],
            [0, 4, -2],
            [2, 1, 1]
        ],
        [
            [12,-51,4],
            [6,167,-68],
            [-4,24,-41]
        ],
    ]

    for A in tests:
        Q, R = qr(np.array(A))
        _Q, _R = linalg.qr(A, mode='complete')
        assert np.allclose(
            Q, _Q), f"Q should be :\n{np.array(_Q)},\n got:\n {Q}"
        assert np.allclose(
            R, _R), f"Q should be :\n{np.array(_R)},\n got:\n {R}"


testHouseholder()
testQR()