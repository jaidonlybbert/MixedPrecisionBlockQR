import numpy as np

def householder(x):
    sigma = np.linalg.norm(x)
    sign = 1 if x[0] >= 0 else -1
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
        [-1, 0, 1]) / (2 ** (1/2))), "v is not correct"
    H = np.identity(3) - 2 * np.outer(v, v)
    reflected = H.dot(raw)
    assert np.allclose(reflected, np.array(
        [2, 0, 0])), 'reflected is not correct'

def testQR():

    # [(A,Q,R)]
    tests = [
        (
            [
                [0, 3, 1],
                [0, 4, -2],
                [2, 1, 1]
            ],
            [
                [0, 0.6, -0.8],
                [0, 0.8, 0.6],
                [1, 0, 0]
            ],
            [
                [2, 1, 1],
                [0, 5, -1],
                [0, 0, -2]
            ]
        ),
        (
            [[12,-51,4],[6,167,-68],[-4,24,-41]],
            [[-0.85714286,0.39428571,0.33142857],[-0.42857143,-0.90285714,-0.03428571],[0.28571429,-0.17142857,0.94285714]],
            [[-14.,-21.,14.], [0.,-175.,70.], [0.,0.,-35.]]
        )
    ]

    for A, _Q, _R in tests:
        Q, R = qr(np.array(A))
        assert np.allclose(
            Q, _Q), f"Q should be :\n{np.array(_Q)},\n got:\n {Q}"
        assert np.allclose(
            R, _R), f"Q should be :\n{np.array(_R)},\n got:\n {R}"


testHouseholder()
testQR()