import numpy as np
from qr import compute_householder_normal, get_householder_factors, householder_qr, block_qr
from wy import wy_representation
from test_data import get_general_matrix, get_strange_matrix
from utils import get_error

red = '\033[31m'
green = '\033[32m'

def test_householder():
    # Vector to be reflected
    raw = np.array([0, 0, 2])
    # Unit normal vector for reflection plane
    v = compute_householder_normal(raw)

    try:
        assert np.allclose(v, np.array([1, 0, 1]) / (2 ** (1/2))), "Unit vector v is not correct"
        H = np.identity(3) - 2 * np.outer(v, v)
        reflected = H.dot(raw)
        assert np.allclose(reflected, np.array([-2, 0, 0])), 'Reflected vector is not correct'
        print(f"{green}householder tests passed.")
    except AssertionError as e:
        print(f"{red}householder test failed: {e}")

def test_qr(qr_func=householder_qr):
    for mode in ('complete', 'reduced'):
        for A in get_general_matrix():
            try:
                Q, R = qr_func(A, mode=mode)
                _Q, _R = np.linalg.qr(A, mode=mode)

                error = get_error(A, Q, R)
                expectError = 1e-8
                assert error < expectError, f"error is {error}, expect small than {expectError}"
                # assert np.allclose(R, _R), f"R should be :\n{_R},\n got:\n {R}"
                # assert np.allclose(Q, _Q), f"Q should be :\n{_Q},\n got:\n {Q}"
                print(f"{green}{qr_func.__name__} QR tests passed.")
            except AssertionError as e:
                print(f"{red}{qr_func.__name__} QR tests failed: {e}")
            except Exception as e:
                print(f"{red}{qr_func.__name__} QR implementation failed: {e}")


def test_qr_edgecases(qr_func=householder_qr):
    for A in get_strange_matrix():
        try:
            Q, R = qr_func(A)
            error = get_error(A, Q, R)
            expectError = 1e-8
            assert error < expectError, f"error is {error}, expect small than {expectError}"
            print(f"{green}{qr_func.__name__} QR edgecase tests passed.")
        except AssertionError as e:
            print(f"{red}{qr_func.__name__} QR edgecase tests failed: {e}")
        except Exception as e:
            print(f"{red}{qr_func.__name__} QR implementation failed: {e}")


def test_wy_representation():
    # Verify Q = (Im - {W}{Y}^T) for the QR decomposition of A = QR
    for A in get_general_matrix():
        V, B = get_householder_factors(A)
        m, n = A.shape
        _Q, _R = np.linalg.qr(A)

        W, Y = wy_representation(V, B)

        Q = np.identity(m) - np.matmul(W, np.transpose(Y))
        Q = Q[:, :n]
        error = np.linalg.norm(Q - _Q) / np.linalg.norm(_Q)
        expectError = 1e-8
        try:
            assert error < expectError, f"error is {error}, expect small than {expectError}"
            assert np.allclose(Q, _Q), f"Q should be :\n{_Q},\n got:\n {Q}"
            print(f"{green}WY representation tests passed.")
        except AssertionError as e:
            print(f"{red}WY test failed: {e}")
        except Exception as e:
            print(f"{red}WY implementation failed: {e}")


test_householder()
test_qr(qr_func=householder_qr)
test_qr_edgecases(qr_func=householder_qr)

test_wy_representation()
test_qr(qr_func=block_qr)

