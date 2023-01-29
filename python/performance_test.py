import numpy as np
from numpy import linalg
from qr import householder_qr
from utils import get_error, generate_matrix
from prettytable import PrettyTable, MARKDOWN
import time
import os

def get_error_and_duration(A, qr_fn):
    start_time = time.time()
    Q, R = qr_fn(A)
    end_time = time.time()
    return format(get_error(A,Q,R), '.2e'), round(end_time - start_time, 2)

def qr_float32(A):
    return householder_qr(A, dtype=np.float32)

def qr_float16(A):
    return householder_qr(A, dtype=np.float16)

def test():
    sizes = [10, 100, 500]
    condition_numbers = [3,4,5,6,7]
    
    duration_rows = []
    error_rows = []
    for n in sizes:
        for con in condition_numbers:
            A = generate_matrix(n, 10 ** con)
            duration_row, error_row = [(n,f'10^{con}')], [(n,f'10^{con}')]

            for fn in qr_float16,qr_float32,householder_qr, linalg.qr:
                error, duration = get_error_and_duration(A, fn)
                duration_row.append(f'{duration}s')
                error_row.append(error)

            duration_rows.append(duration_row)
            error_rows.append(error_row)

    error_table, duration_table = PrettyTable(), PrettyTable()
    error_table.set_style(MARKDOWN)
    duration_table.set_style(MARKDOWN)
    error_table.field_names = ["(n, condition_num)", "qr float16", "qr float32", "qr float64", "numpy(lapack) qr float64"]
    duration_table.field_names = ["(n, condition_num)", "qr float16", "qr float32", "qr float64", "numpy(lapack) qr float64"]
    error_table.add_rows(error_rows)
    duration_table.add_rows(duration_rows)
    with open(os.path.join(os.path.dirname(__file__), './performance_test_result/error.md'), 'w') as f:
        f.write(error_table.get_string())
    with open(os.path.join(os.path.dirname(__file__), './performance_test_result/duration.md'), 'w') as f:
        f.write(duration_table.get_string())

test()

