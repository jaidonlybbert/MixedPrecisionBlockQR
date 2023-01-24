import numpy as np
from qr import householder

def test_householder():
    raw = np.array([0, 0, 2])
    v = householder(raw)
    assert np.allclose(v, np.array([1, 0, 1]) / (2 ** (1/2))), "Unit vector v is not correct"

    H = np.identity(3) - 2 * np.outer(v, v)
    reflected = H.dot(raw)
    assert np.allclose(reflected, np.array([-2, 0, 0])), 'Reflected vector is not correct'

test_householder()