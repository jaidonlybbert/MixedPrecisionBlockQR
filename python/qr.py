import numpy as np
from numpy import linalg

def householder(x):
    sigma = np.linalg.norm(x)
    sign = -1
    e = np.zeros_like(x)
    e[0] = 1
    u = x - sign * sigma * e
    return u / np.linalg.norm(u)

def qr(A):
    n = len(A)
    Q,H = np.identity(n), np.identity(n)
    
    for i in range(n - 1):
        Ai = A[i:, i:]
        v = householder(Ai[:, 0])
        H_hat = np.identity(n - i) - 2 * np.outer(v, v)
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
        _Q, _R = linalg.qr(A)
        assert np.allclose(
            Q, _Q), f"Q should be :\n{np.array(_Q)},\n got:\n {Q}"
        assert np.allclose(
            R, _R), f"Q should be :\n{np.array(_R)},\n got:\n {R}"


testHouseholder()
testQR()