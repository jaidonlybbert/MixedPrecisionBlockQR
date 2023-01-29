from utils import generate_matrix
import numpy as np
from qr import householder_qr

def linear_least_square(A, y):
    Q,R = householder_qr(A)
    n = R.shape[0]
    X = [0] * n
    # get generalized inverse for rectangle matrix, for square maxtix, can use Q.T
    b = np.dot(np.linalg.pinv(Q), y)

    # r4 * x4 = b4
    # r3*x3 + r4 * x4 = b3
    # r2*x2 + r3 * x3 + r4 * x4 = b2
    # ...
    # xi = (bi - (rn * xn + ... ri+1 * xi+1)) / ri
    for i in reversed(range(n)):
        total = 0
        for k in  range(i + 1,n):
            total += X[k] * R[i][k]
        X[i] = (b[i] - total) / R[i][i]
    return np.array(X)


def test():
    tests = [
        [
            [1,2,3,4],
            [6,5,7,10]
        ],
        [
            [1,2,3],
            [4,5,6],
            [7,8,7],
            [4,2,3],
            [4,2,2],
            [10,20,30]
        ],
        np.random.random((100, 100)),
        generate_matrix(100, 10 ** 5)
    ]
    for dataset in tests:
        dataset = np.array(dataset)
        y = dataset[-1]
        x = dataset[0:-1].T
        A = np.c_[np.ones(x.shape[0]), x]
        
        X = linear_least_square(A, y)
        _X = np.linalg.lstsq(A, y, rcond=None)[0]
        assert np.allclose(X, X), f"X should be :\n{_X},\n got:\n {X}"

def test_ill_condition():
    # x + αy = 1
    # αx + y = 0
    # α = 0.999
    A = np.array([
        [1, 0.999],
        [0.999, 1]
    ])
    y = np.array([1.0, 0.0])
    delta_x = 0.0001
    X = np.linalg.lstsq(A, y,rcond=None)[0]
    A[0][1] += delta_x
    A[1][0] += delta_x
    _X = np.linalg.lstsq(A, y,rcond=None)[0]
    print(np.linalg.norm(X - _X) / delta_x)

def test_error():
    delta_x = 0.001
    dataset = generate_matrix(100, 10 ** 5)
    y = dataset[-1]
    x = dataset[0:-1].T

    A = np.c_[np.ones(x.shape[0]), x]
    print(np.linalg.cond(A)) # 200000
    X = np.linalg.lstsq(A, y,rcond=None)[0]
    
    y[0] += delta_x
    _X = np.linalg.lstsq(A, y,rcond=None)[0]

    print(np.linalg.norm(X - _X) / delta_x) # 45

    

test()
test_error()