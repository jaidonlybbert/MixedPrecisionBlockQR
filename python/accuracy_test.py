import numpy as np
from numpy import linalg
from qr import householder_qr, block_qr
from utils import get_error, generate_matrix

def test(qr_func=householder_qr):
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
        np.random.random((10, 10)),
        np.random.random((100, 100)),
        np.random.random((200, 100)),
        generate_matrix(100, 100),
    ]

    for mode in ('complete', 'reduced'):
        for A in tests:
            Q, R = qr_func(A, mode=mode)
            _Q, _R = linalg.qr(A, mode=mode)

            error = get_error(A, Q, R)
            expectError = 1e-8
            assert error < expectError, f"error is {error}, expect small than {expectError}"
            assert np.allclose(R, _R), f"R should be :\n{_R},\n got:\n {R}"
            assert np.allclose(Q, _Q), f"Q should be :\n{_Q},\n got:\n {Q}"
    
def edge_case_test(qr_func=householder_qr):
    edge_cases = [
        [
            [1,2,3],
            [1,2,3],
            [1,2,3]
        ],
        [
            [1,0,0],
            [0,2,0],
            [0,0,3]
        ],
        [
            [1,2,3],
            [0,0,0],
            [0,0,0]
        ],
    ]
    for A in edge_cases:
        Q, R = qr_func(A)
        error = get_error(A, Q, R)
        expectError = 1e-8
        assert error < expectError, f"error is {error}, expect small than {expectError}"

try:
    test()
    edge_case_test()
    print("All householder QR tests passed.")
except AssertionError as e:
    print(f"Householder test failed: {e}")

try:
    test(qr_func=block_qr)
    print("All block QR tests passed.")
except AssertionError as e:
    print(f"Block QR test failed: {e}")


