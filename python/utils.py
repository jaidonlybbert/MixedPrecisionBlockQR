
import numpy as np

def get_error(A, Q, R):
    return np.linalg.norm(A - np.dot(Q, R)) / np.linalg.norm(A)

# Construct random matrix P with specified condition number
#
#  Bierlaire, M., Toint, P., and Tuyttens, D. (1991). 
#  On iterative algorithms for linear ls problems with bound constraints. 
#  Linear Algebra and Its Applications, 143, 111–143.
#
def generate_matrix(n, condition_number=100):
    cond_P = float(condition_number) 
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n + 1)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    exp_vec = exp_vec[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P


