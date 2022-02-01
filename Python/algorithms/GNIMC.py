### Gauss-Newton based algorithm for inductive matrix completion ###
### Written by Pini Zilber and Boaz Nadler, 2022 ###

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from algorithms.init_options import INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED

def GNIMC(X, omega, rank, A, B, verbose=True, alpha=-1, perform_qr=True, max_outer_iter=100,
    max_inner_iter_init=10, max_inner_iter_final=1000, lsqr_inner_init_tol=1e-15, lsqr_smart_tol=True, lsqr_smart_obj_min=1e-5,
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
    :param int alpha: corresponds to the choice of the minimal norm solution to the lsqr problem. -1 by default
    :param bool perform_qr: work with QR decomposition of the factor matrices to enhance performance
    :param int max_outer_iter: Maximal number of outer iterations
    :param int max_inner_iter_init: Maximal number of inner iterations
    :param int max_inner_iter_final: use this if opts.lsqr_smart_tol && relRes < opts.lsqr_smart_obj_min
    :param float lsqr_init_tol: initial tolerance of the LSQR solver
    :param bool lsqr_smart_tol: if True, when relRes <= lsqr_smart_obj_min, use lsqr_tol=objective**2
    :param float lsqr_smart_obj_min: relRes threshold to begin smart tolerance from
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
    AU_omega_rows = A[I,:] @ U
    BV_omega_cols = B[J,:] @ V

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
    x_hat = np.sum(AU_omega_rows * BV_omega_cols, 1)
    x_hat_prev = x_hat

    # iterations
    iter_num = 0
    while iter_num < max_outer_iter and not early_stopping_flag:
        iter_num += 1

        # construct variables for lsqr
        if perform_qr:
          U_Q, U_R = np.linalg.qr(U)
          V_Q, V_R = np.linalg.qr(V)
          AU_for_use = A[I,:] @ U_Q
          BV_for_use = B[J,:] @ V_Q
        else:
          AU_for_use = AU_omega_rows
          BV_for_use = BV_omega_cols
        L1 = generate_product_matrix(A[I,:], BV_for_use.T)
        L2 = generate_product_matrix(AU_for_use, B[J,:].T)
        L = sparse.csr_matrix(np.concatenate((L1, L2), axis=1))
        
        # update b of lsqr (as in ||Ax-b||^2)
        b = np.zeros((m,))
        update = alpha * np.sum(AU_omega_rows * BV_omega_cols, 1)
        b = x + update

        # determine LSQR tolerance and #iterations
        lsqr_tol = lsqr_inner_init_tol
        lsqr_iters = max_inner_iter_init
        if lsqr_smart_tol and relRes < lsqr_smart_obj_min:
          lsqr_tol = min(lsqr_tol, relRes**2)
          lsqr_iters = max_inner_iter_final

        # solve the least squares problem
        z = sp_linalg.lsqr(L, b, atol=lsqr_tol, btol=lsqr_tol, iter_lim=lsqr_iters)[0]
        #relRes = res / np.linalg.norm(b)

        # construct Utilde and Vtilde from the solution z
        U_tilde = np.reshape(z[:d1*rank], (d1, rank))
        V_tilde = np.reshape(z[d1*rank:], (rank, d2)).T
        if perform_qr:
          U_tilde = U_tilde @ np.linalg.inv(V_R).T
          V_tilde = V_tilde @ np.linalg.inv(U_R).T

        # calculate new U, V
        U = 0.5 * (1 - alpha) * U + U_tilde
        V = 0.5 * (1 - alpha) * V + V_tilde
        AU_omega_rows = A[I,:] @ U
        BV_omega_cols = B[J,:] @ V
        
        # get new estimate and calculate corresponding error
        x_hat = np.sum(AU_omega_rows * BV_omega_cols, 1)
        #X_hat = sparse.csr_matrix((x_hat, (I, J)), shape=(n1,n2))

        relRes = np.linalg.norm(x_hat - x) / X_norm
        all_relRes.append(relRes)
        if relRes < best_relRes:
          best_relRes = relRes
          U_best = U
          V_best = V
        x_hat_diff = np.linalg.norm(x_hat - x_hat_prev) / np.linalg.norm(x_hat)

        # report
        if verbose:
          print("[INSIDE GNIMC] iter: " + str(iter_num) + ", relRes: " + str(relRes))

        # check early stopping criteria
        if stop_relRes > 0:
          early_stopping_flag |= relRes < stop_relRes
        if stop_relDiff > 0:
          early_stopping_flag |= x_hat_diff < stop_relDiff
        if stop_relResDiff > 0:
          early_stopping_flag |= np.abs(relRes / all_relRes[-2] - 1) < stop_relResDiff
        if verbose and early_stopping_flag:
          print("[INSIDE GNIMC] early stopping")

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