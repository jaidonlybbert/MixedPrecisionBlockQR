import numpy as np

# remember to run accuracy_test and ensure it pass if you change this file

def compute_householder_normal(u):
    """Computes unit normal vector for bisecting reflection plane
    of the Householder reflection

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    sigma = np.linalg.norm(u)
    sign = -1 if u[0] >= 0 else 1
    e = np.zeros_like(u)
    e[0] = 1
    v = sign * sigma * e
    u = u - v
    w = u / np.linalg.norm(u)
    return w

def householder_qr(A, dtype=np.float64, mode='reduced'):
    """Implements QR decomposition using Householder reflections for NxM matrix A
    
    Bhaskar Dasgupta. Applied Mathematical Methods. Pearson, 1986.

    Args:
        A (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to np.float64.
        mode (str, optional): _description_. Defaults to 'reduced'.

    Returns:
        _type_: _description_
    """
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
        # select i-th coloumn of A
        column_need_transform = A[i:,i]
        # skip this transformation if this vector is all zero 
        if np.allclose(column_need_transform, np.zeros_like(column_need_transform)):
            continue
        # compute normal vector of bisecting plane
        w = compute_householder_normal(column_need_transform)
        # compute householder reflection matrix
        H_hat = np.identity(m - i, dtype=dtype) - 2 * np.outer(w, w)
        H = np.pad(H_hat, (i, 0), 'constant')
        H[np.diag_indices(i)] = 1
        Q = Q.dot(H)
        A = H.dot(A)
    if mode == 'reduced':
        return Q[:,:n], A[:n]
    else:
        return Q,A

def block_qr(A, dtype=np.float64, mode='reduced'):
    """Implements Block QR decomposition such that A = QR using Householder 
    reflections and WY representation.

    Reference:
        Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins 
        University Press. Pg. 239. Algorithm 5.2.3

    Args:
        A (np.ndarray): mxn rectangular matrix
    """

    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=dtype)
    if A.dtype != dtype:
        A = A.astype(dtype)

    m, n = A.shape
    Q = np.zeros(shape=(m, n))
    R = np.zeros(shape=(n, n))

    if mode == 'reduced':
        return Q[:,:n], R[:n]
    else:
        return Q,R