import numpy as np
from wy import wy_representation

# remember to run test_all to ensure the previous function is not broken if you change this file

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
    V,B = [], []

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
        V.append(np.pad(w, (i, 0), 'constant'))
        B.append(2.0)
        # compute householder reflection matrix
        H_hat = np.identity(m - i, dtype=dtype) - 2 * np.outer(w, w)
        H = np.pad(H_hat, (i, 0), 'constant')
        H[np.diag_indices(i)] = 1
        Q = Q.dot(H)
        A = H.dot(A)
    if mode == 'reduced':
        return Q[:,:n], A[:n]
    elif mode == 'complete':
        return Q,A
    else:
        return V,B

def get_householder_factors(tile, dtype=np.float64):
    """Find the group of householder transformation matrix factors V, and B such
     that H_i = I - {B[i]}{V[i]}{V[i]}^T for each matrix in the group of householder
     matrices H_group = [H_i, H_(i+1), ..., H_j] 
     such that the matrix product Q = {H_i}{H_(i+1)}{...}{H_j} has the properties
     of Q in the QR decomposition of tile = QR

    Args:
        tile (np.ndarray): A matrix of size m x (i-j+1)

    Returns:
        V (List[np.ndarray]): A list of Householder transformation matrices
            [H_i, H_(i+j), ..., H_j]
        B (List[np.float64]): A list of coefficients {beta}
    """
    return householder_qr(tile, mode='raw')


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

    Q = np.identity(m)
    R = np.zeros(shape=(m, n))

    lambda_ = 0
    k = 0
    r = 3

    while lambda_ <= n:
        tau = min(lambda_ + r, n+1)
        k += 1

        A_tile = A[lambda_:(m+1), lambda_:tau]
        V, B = get_householder_factors(A_tile)

        W, Y = wy_representation(V, B)
        tile_h, tile_w = A_tile.shape
        wy_mtx = np.identity(tile_h) - np.matmul(W, np.transpose(Y))

        A[lambda_:(m+1), (tau+1):(n+1)] = np.matmul(np.transpose(wy_mtx), A[lambda_:(m+1), (tau+1):(n+1)])
        Q[:, lambda_:(m+1)] = Q[:, lambda_:(m+1)] @ wy_mtx
        
        lambda_ = tau + 1

    R = np.linalg.inv(Q) @ A

    if mode == 'reduced':
        return Q[:,:n], R[:n]
    else:
        return Q,R