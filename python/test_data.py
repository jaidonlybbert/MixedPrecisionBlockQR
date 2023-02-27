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
    [
        [10,20,30,40,50,60],
        [32,32,44,55,66,35],
        [23,66,74,64,45,65],
        [67,28,46,26,46,42],
        [95,95,52,88,65,11],
        [75,53,96,47,32,32],
    ],
    np.random.random((10, 10)),
    np.random.random((100, 100)),
    np.random.random((200, 100)),
    np.random.random((300, 100)),
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
