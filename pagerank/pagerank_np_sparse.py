#! /usr/bin/env python

# Reference http://www.ams.org/samplings/feature-column/fcarc-pagerank

from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

def pagerank_sparse(H, alpha=0.85, check_iter=5, max_iter=30, convergence=1e-4):
    """
    H - Takes in the adjacency matrix as a sparse matrix.
    The adjacency matrix is out->in.
    """
    N = H.shape[0]
    I = np.ones((N,), np.float64) / N
    I_old = np.ones((N,), np.float64) / N
    A = np.zeros(N, np.float64)
    leaf_nodes = np.where(H.sum(axis=0) == 0)[1]

    # A1 flattens into 1-d array
    # from binary/incidence matrix to normalized by number of outgoing edges
    H.data /= np.take(H.sum(axis=0).A1, H.indices)

    done = False

    num_iter = 0
    while not done:
        I /= sum(I)
        for i in range(check_iter):
            num_iter += 1
            I_old, I = I, I_old
            T = (1 - alpha) * sum(I_old) / N
            A = alpha * sum(I_old.take(leaf_nodes, axis = 0)) / N
            I = (alpha * (H @ I_old)) + A + T
        diff = sum(abs(I - I_old))
        done = (diff < convergence) or num_iter > max_iter
        
    return I, num_iter

    

