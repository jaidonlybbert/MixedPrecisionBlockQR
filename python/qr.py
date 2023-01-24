import numpy as np

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
    if A.dtype != dtype:
        A = A.astype(dtype)
    m,n = A.shape
    Q,H = np.identity(m, dtype=dtype), np.identity(m, dtype=dtype)
    
    for i in range(n):
        # skip last transformation if is sqare matrix
        if m == n and i == n - 1:
            break
        column_need_transform = A[i:,i]
        # skip this transformation if this vector is all zero 
        if np.allclose(column_need_transform, np.zeros_like(column_need_transform)):
            continue
        v = householder(column_need_transform)
        H_hat = np.identity(m - i, dtype=dtype) - 2 * np.outer(v, v)
        H = np.pad(H_hat, (i, 0), 'constant')
        H[np.diag_indices(i)] = 1
        Q = Q.dot(H)
        A = H.dot(A)
    if mode == 'reduced':
        return Q[:,:n], A[:n]
    else:
        return Q,A