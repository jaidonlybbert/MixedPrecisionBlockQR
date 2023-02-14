import numpy as np
from utils import generate_matrix

def get_general_matrix():
    return map(np.array, [
    [
        [1,2,3],
        [4,5,6],
        [7,8,7],
        [4,2,3],
        [4,2,2]
    ],
    [
        [0, 3, 1],
        [0, 4, -2],
        [2, 1, 1]
    ],
    [
        [12,-51,4],
        [6,167,-68],
        [-4,24,-41]
    ],
    np.random.random((10, 10)),
    np.random.random((100, 100)),
    np.random.random((200, 100)),
    generate_matrix(100, 100),
])

def get_strange_matrix():
    return (
        map(np.array,[
            [
                [1,2,3],
                [1,2,3],
                [1,2,3]
            ],
            [
                [1,0,0],
                [0,2,0],
                [0,0,3]
            ],
            [
                [1,2,3],
                [0,0,0],
                [0,0,0]
            ],
        ])
    )
