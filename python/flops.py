'''
For visualization of FLOP count of QR decomposition algorithms
'''

def average_square(m, n):
    avg = 0
    for i in range(n):
        avg += (m-i)*(n-i)
    avg /= n
    return avg


def householder_qr_flops(m, n):
    '''
    See QR decomposition algorithms section 3.1.1
    '''

    line1_flops = n * m / 2
    line2_flops = 3 * n * m / 2
    line3_flops = average_square(m, n)

    print(f'{line3_flops / (n*m)} * N * M')

householder_qr_flops(1000, 1000)