import numpy as np
from utils import generate_matrix

def test_generate_matrix():
    for i in 100,1000,10000:
        cond = np.linalg.cond(generate_matrix(100, i))
        assert np.isclose(cond,i)

test_generate_matrix()