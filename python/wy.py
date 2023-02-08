import numpy as np

def wy_representation(V, B) -> np.ndarray:
    """For the factored form of Q = {Q1}{Q2}{...}{Qn} where Qn = {Im} - {beta_j}{v_j}{v_j}^T
        and the factors {v_j, b_j} are stored as V = [v0, v1, ..., vn], B = [beta_1, beta_2, ..., beta_n],
        the W and Y factors such that Q = Im - {W}{Y}^T can be calculated from V, and B.

        Since Y = V, only W is returned.

        Reference:
             Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins University Press. Pg. 239. Algorithm 5.1.2

    Args:
        V (np.ndarray): matrix V containing n householder vectors [v0, v1, ..., vn]
        B (np.ndarray): matrix B containing n coefficients [beta0, beta1, ..., betan] derived from factored forms 
    """

    m = len(V[0])
    r = len(V)

    Y = np.array(V[0]).reshape(m, 1)
    W = np.array(B[0]*V[0]).reshape(m, 1)

    for i in range(1, r):
        z = B[i] * np.dot((np.identity(m) - np.matmul(W, np.transpose(Y))), V[i])
        z = z.reshape(m, 1)
        W = np.concatenate((W, z), axis=1)
        Y = np.concatenate((Y, V[i].reshape(m, 1)), axis=1)
    
    return W, Y

def test_wy_representation(A, V, B):
    # Verify Q = (Im - {W}{Y}^T) for the QR decomposition of A = QR
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
        print(f"WY representation tests passed.")
    except AssertionError as e:
        print(f"WY test failed: {e}")