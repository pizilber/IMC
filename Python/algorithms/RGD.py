### Gradient descent algorithm for inductive matrix completion ###
### with option for balance regularization in the form of lambda * || U.T @ U - V.T @ V ||_F^2
### Written by Pini Zilber and Boaz Nadler, 2022 ###

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from algorithms.init_options import INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED

def RGD(X, omega, rank, A, B, verbose=True, eta=0.05, reg_lambda=0.5, max_outer_iter=10000,
    init_option=INIT_WITH_SVD, init_U=None, init_V=None,
    stop_relRes=1e-14, stop_relDiff = -1, stop_relResDiff = -1):
    """
    Run GNIMC algorithm for inductive matrix completion
    :param ndarray X: Input matrix (n1,n2) whose row and column spaces are spanned by A (n1,d1) and B (n2,d2).
      Unobserved entries should be zero
    :param ndarray omega: Mask matrix (n1,n2). 1 on observed entries, 0 on unobserved
    :param int rank: Underlying rank matrix
    :param ndarray A: side information matrix (n1,d1), whose columns span the row space of the underlying matrix
    :param ndarray B: side information matrix (n2,d2), whose columns span the column space of the underlying matrix
    :param bool verbose: if True, display intermediate results
    :param int eta: step size
    :param bool reg_lambda: balance regularization coefficient
    :param int max_outer_iter: Maximal number of outer iterations
    :param int init_option: how to initialize U and V (INIT_WITH_SVD, INIT_WITH_RAND, or INIT_WITH_USER_DEFINED)
    :param ndarray init_U: U initialization (n1,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param ndarray init_V: V initialization (n2,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param float stop_relRes: relRes threshold for ealy stopping (relevant to noise-free case), -1 to disable
    :param float stop_relDiff: relative X_hat difference threshold for ealy stopping, -1 to disable
    :param float stop_relResDiff: relRes difference difference threshold for early stopping, -1 to disable
    :return: GNIMC's estimate, final iteration number, convergence flag and all relRes
    """
    n1, n2 = X.shape
    d1 = A.shape[1]
    d2 = B.shape[1]
    m = np.count_nonzero(omega)
    p = m / (n1*n2)
    I, J, _ = sparse.find(omega)

    # initial estimate
    if init_option == INIT_WITH_SVD:
      L, S, R = sp_linalg.svds(X/p, k=rank, tol=1e-16)
      U = A.T @ L @ np.diag(np.sqrt(S))
      V = B.T @ R.T @ np.diag(np.sqrt(S))
    elif init_option == INIT_WITH_RANDOM:
      U = np.random.randn(d1, rank)
      V = np.random.randn(d2, rank)
      U = np.linalg.qr(U)[0]
      V = np.linalg.qr(V)[0]
    else:
      U = init_U
      V = init_V

    # before iterations
    X_sparse = sparse.csr_matrix(X)
    x = X[I,J]
    X_norm = np.linalg.norm(x)
    early_stopping_flag = False
    relRes = float("inf")
    all_relRes = [relRes]
    best_relRes = float("inf")
    U_best = U
    V_best = V
    x_hat_prev = sparse.csr_matrix(np.zeros_like(x))

    # iterations
    iter_num = 0
    while iter_num < max_outer_iter and not early_stopping_flag:
        iter_num += 1

        x_hat = np.sum((A[I,:] @ U) * (B[J,:] @ V), 1)
        error = x_hat - x
        At_error_B = A.T @ sparse.csr_matrix((error, (I,J)), shape=X.shape) @ B
        nabla_U = (1/p) * At_error_B @ V + reg_lambda * U @ (U.T@U - V.T@V)
        nabla_V = (1/p) * At_error_B.T @ U + reg_lambda * V @ (V.T@V - U.T@U)
        U = U - eta * nabla_U
        V = V - eta * nabla_V
        
        # calculate error
        relRes = np.linalg.norm(x_hat - x) / X_norm
        all_relRes.append(relRes)
        if relRes < best_relRes:
          best_relRes = relRes
          U_best = U
          V_best = V
        x_hat_diff = np.linalg.norm(x_hat - x_hat_prev) / np.linalg.norm(x_hat)

        # report
        if verbose:
          print("[INSIDE RGD] iter: " + str(iter_num) + ", relRes: " + str(relRes))

        # check early stopping criteria
        if stop_relRes > 0:
          early_stopping_flag |= relRes < stop_relRes
        if stop_relDiff > 0:
          early_stopping_flag |= x_hat_diff < stop_relDiff
        if stop_relResDiff > 0:
          early_stopping_flag |= np.abs(relRes / all_relRes[-2] - 1) < stop_relResDiff
        if verbose and early_stopping_flag:
          print("[INSIDE RGD] early stopping")

    # return
    convergence_flag = iter_num < max_outer_iter
    X_hat = A @ U_best @ V_best.T @ B.T 
    return X_hat, iter_num, convergence_flag, all_relRes


def generate_product_matrix(A, B):
  """
  Returns M such that M @ vec(C) = vec(A @ C @ B)
  """
  assert((A.shape[0] == B.shape[1]), 'error: dimension mismatch')

  m = A.shape[0]
  M = np.zeros((m, A.shape[1] * B.shape[0]))
  for i in range(m):
    AB = np.outer(A[i,:], B[:,i])
    M[i,:] = AB.flatten()
  return M