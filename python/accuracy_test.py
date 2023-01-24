import numpy as np
from numpy import linalg
from qr import qr
from utils import get_error, generate_matrix
from prettytable import PrettyTable

def test():
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
            Q, R = qr(A, mode=mode)
            _Q, _R = linalg.qr(A, mode=mode)

            error = get_error(A, Q, R)
            expectError = 1e-8
            assert error < expectError, f"error is {error}, expect small than {expectError}"
            assert np.allclose(R, _R), f"R should be :\n{_R},\n got:\n {R}"
            assert np.allclose(Q, _Q), f"Q should be :\n{_Q},\n got:\n {Q}"
    
def edge_case_test():
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
        Q, R = qr(A)
        error = get_error(A, Q, R)
        expectError = 1e-8
        assert error < expectError, f"error is {error}, expect small than {expectError}"

def ill_conditioned_test():
    sizes = [10, 100]
    condition_numbers = [3,4,5,6,7]
    
    rows = []
    for n in sizes:
        for con in condition_numbers:
            A = generate_matrix(n, 10 ** con)

            Q, R = qr(A)
            Q1, R1 = qr(A, dtype=np.float32)
            _Q,_R = linalg.qr(A)

            rows.append(
                [(n,f'10^{con}'), get_error(A,Q,R), get_error(A,Q1,R1), get_error(A,_Q,_R)]
            )
    table = PrettyTable()
    table.field_names = ["(n, condition_num)", "qr float64", "qr float32", "numpy(lapack) qr float64"]
    table.add_rows(rows)
    print(table)

test()
edge_case_test()
ill_conditioned_test()