import numpy as np
from numpy import linalg
import time

np.set_printoptions(suppress = True)

def householder(x):
    sigma = np.linalg.norm(x)
    sign = -1 if x[0] >= 0 else 1
    e = np.zeros_like(x)
    e[0] = 1
    u = x - sign * sigma * e
    return u / np.linalg.norm(u)

def qr(A, dtype=np.float64, mode='reduced'):
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=dtype)
    m,n = A.shape
    Q,H = np.identity(m, dtype=dtype), np.identity(m, dtype=dtype)
    
    for i in range(n):
        # skip last transformation if is sqare matrix
        if m == n and i == n - 1:
            break
        Ai = A[i:, i:]
        v = householder(Ai[:, 0])
        H_hat = np.identity(m - i, dtype=dtype) - 2 * np.outer(v, v)
        H = np.pad(H_hat, (i, 0), 'constant')
        H[np.diag_indices(i)] = 1
        Q = Q.dot(H)
        A = H.dot(A)
    if mode == 'reduced':
        return Q[:,:n], A[:n]
    else:
        return Q,A

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
        # [
        #     [1,2,3],
        #     [1,2,3],
        #     [1,2,3]
        # ],
        np.random.random((10, 10)),
        np.random.random((100, 100)),
        np.random.random((200, 100)),
    ]
    for mode in ('complete', 'reduced'):
        for A in tests:
            Q, R = qr(A, mode=mode)
            _Q, _R = linalg.qr(A, mode=mode)

            error = np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)
            expectError = 1e-8
            assert error < expectError, f"error is {error}, expect small than {expectError}"
            assert np.allclose(R, _R), f"Q should be :\n{_R},\n got:\n {R}"
            assert np.allclose(Q, _Q), f"Q should be :\n{_Q},\n got:\n {Q}"



    
        
def testPerformance():
    for dtype in np.float64, np.float32:
        for n in 100,1000:
            A = np.random.random((n, n)).astype(dtype)
            start_time = time.time()
            Q, R = qr(A, dtype=dtype)
            print(f'{n}x{n} {dtype} matrix cost time: {time.time() - start_time} seconds')

            A = np.random.random((n, n)).astype(dtype)
            start_time = time.time()
            Q, R = linalg.qr(A, mode='complete')
            print(f'{n}x{n} {dtype} matrix numpy cost time: {time.time() - start_time} seconds')


# testHouseholder()
testQR()
# testPerformance()
