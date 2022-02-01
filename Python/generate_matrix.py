import numpy as np

def generate_matrix(n1, n2, d1, d2, singular_values):
    """
    Generate an n1 x n2 matrix X with specific singular values
    whose row and column spcaes are spanned by matrices A,B of sizes n1 x d1 and n2 x d2, respectively
    :param int n1: number of rows
    :param int n2: number of columns
    :param int d1: number of columns of A
    :param int d2: number of columns of B
    :param list singular_values: required singular values
    """

    rank = len(singular_values)

    A = np.random.randn(n1, d1)
    B = np.random.randn(n2, d2)
    A = np.linalg.qr(A)[0]
    B = np.linalg.qr(B)[0]

    U = np.random.randn(d1, rank)
    V = np.random.randn(d2, rank)
    U = np.linalg.qr(U)[0]
    V = np.linalg.qr(V)[0]

    D = np.diag(singular_values)

    return A @ U @ D @ V.T @ B.T, A, B
