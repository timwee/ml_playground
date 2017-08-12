#! /usr/bin/env python

from scipy import linalg
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np


def randomized_krylov_svd(A, n_components, n_iter=8):
    m, n = A.shape

    # oversample a bit
    num_sample = n_components + 12

    # Random block initialization
    G = np.random.normal(size=(A.shape[1], num_sample))

    # Make the Krylov subspaces, and orthonormalize
    K_new_dim, _ = linalg.qr(safe_sparse_dot(A, G), mode='economic')
    del G
    K = K_new_dim
    for it in range(1, n_iter):
        # Krylov, keep multiplying with AAT
        K_new_dim, _ = linalg.qr(safe_sparse_dot(A, safe_sparse_dot(A.T, K_new_dim)),
                             mode='economic')
        K = np.hstack((K,K_new_dim))
    # orthonormal basis for Krylov subspace
    Q, _ = linalg.qr(K, mode='economic')
    del K

    # postprocessing rayleigh ritz
    B = safe_sparse_dot(Q.T, A)
    U_prime, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, U_prime)
    return U[:, :n_components], s[:n_components], V[:n_components, :]