import numpy as np
from numpy import linalg
from qr import householder_qr
from utils import get_error, generate_matrix
from prettytable import PrettyTable, MARKDOWN
import time

def get_error_and_duration(A, qr_fn):
    start_time = time.time()
    Q, R = qr_fn(A)
    end_time = time.time()
    return format(get_error(A,Q,R), '.3g'), round(end_time - start_time, 2)
    # row.append(f'error:{get_error(A,Q,R)}, duration:{round(end_time - start_time, 2)}')

def qr_float32(A):
    return householder_qr(A, dtype=np.float32)

def ill_conditioned_test():
    sizes = [10, 100, 1000]
    condition_numbers = [3,4,5,6,7]
    
    rows = []
    for n in sizes:
        for con in condition_numbers:
            A = generate_matrix(n, 10 ** con)
            row = [(n,f'10^{con}')]

            error, duration = get_error_and_duration(A, qr_float32)
            row.append(f'e:{error}, d:{duration}s')

            error, duration = get_error_and_duration(A, householder_qr)
            row.append(f'e:{error}, d:{duration}s')

            error, duration = get_error_and_duration(A, linalg.qr)
            row.append(f'e:{error}, d:{duration}s')

            rows.append(row)

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ["(n, condition_num)", "qr float32", "qr float64", "numpy(lapack) qr float64"]
    table.add_rows(rows)
    print(table)

ill_conditioned_test()