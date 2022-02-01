import numpy as np
from generate_matrix import generate_matrix
from generate_mask import generate_mask
from algorithms.init_options import INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED
from algorithms.GNIMC import GNIMC
from algorithms.AltMin import AltMin
from algorithms.RGD import RGD

def run_demo():
    # experiment configurations
    n1 = 300
    n2 = 350
    rank = 5
    d1 = 2*rank
    d2 = 3*rank
    condition_number = 1e1
    oversampling_ratio = 2
    singular_values = np.linspace(1, condition_number, rank)
    print("n1, n2:", n1, n2)
    print("d1, d2:", n1, n2)
    print("oversampling ratio:", oversampling_ratio)
    print("singular values:", singular_values)

    # options (see more options in the algorithms)
    options = {
        # general
        'verbose' : True,
        # early stopping criteria (-1 to disable a criterion)
        'stop_relRes':  5e-14,
        'stop_relDiff': 5e-14,
    }

    # calculate full matrix, mask and corresponding observed matrix
    p = oversampling_ratio * rank * (d1 + d2 - rank) / (n1 * n2)
    X0, A, B = generate_matrix(n1, n2, d1, d2, singular_values)
    omega = generate_mask(n1, n2, p)
    X = X0 * omega
    
    # run GNIMC
    X_hat, _, _, _ = RGD(X, omega, rank, A, B, **options)

    # report
    true_error = np.linalg.norm(X_hat - X0, ord='fro') / np.linalg.norm(X0, ord='fro')
    observed_error = np.linalg.norm((X_hat - X0) * omega, ord='fro') / np.linalg.norm(X0, ord='fro')
    print('true error: {}, observed error: {}'.format(true_error, observed_error))


if __name__ == '__main__':
    run_demo()
