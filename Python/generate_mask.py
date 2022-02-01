import numpy as np

def generate_mask(n1, n2, rank, p):
    """
    Generate a mask with probability p for each entry
    :param int n1: number of rows
    :param int n2: number of columns
    :param int rank: number of minimal observed entries in each row and column
    :param float p: probability of observing an entry
    """
    omega = np.round(0.5 * (np.random.random((n1, n2)) + p))
    return omega