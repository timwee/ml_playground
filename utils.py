#! /usr/bin/env python

from scipy.sparse import csr_matrix


def read_sparse_h5(f, k):
    """ 
    Saves the following metadata:
    indptr, data, indices
    """
    indptr = f[k + "_indptr"]
    data = f[k + "_data"]
    indices = f[k + "_indices"]
    shape_arr = f[k + "_shape"]
    return csr_matrix((data, indices, indptr), shape=(shape_arr[0], shape_arr[1]))


