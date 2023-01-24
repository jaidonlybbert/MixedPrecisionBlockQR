import numpy as np
from numpy import linalg
import time
from qr import qr

def test_perfomance():
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

test_perfomance()