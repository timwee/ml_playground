# For testing, compare sparse implementation with naive
# copied naive version from https://github.com/louridas/pagerank/blob/master/python/pageRank.py


# PageRank Python implementation by Vincent Kraeutler, at:
# http://kraeutler.net/vincent/essays/google%20page%20rank%20in%20python
#
# The code has the following changes from the original:
#
# * The data types were converted from 32 bits to 64 bits and precision
#
# * The convergence criterion was changed from the average deviation
# to the Euclidean 1-norm distance, which is tighter
# 
# The original code was released by Vincent Kraeutler under a Creative
# Commons Attribution 2.5 License
#

#!/usr/bin/env python

from numpy import *
from numba import jit, vectorize

@jit()
def pageRankGenerator(At = [array((), int64)], 
                      numLinks = array((), int64),  
                      ln = array((), int64),
                      alpha = 0.85, 
                      convergence = 0.0001, 
                      checkSteps = 10
                      ):
    """
    Compute an approximate page rank vector of N pages to within
    some convergence factor.

    @param At a sparse square matrix with N rows. At[ii] contains
    the indices of pages jj linking to ii.

    @param numLinks iNumLinks[ii] is the number of links going out
    from ii.

    @param ln contains the indices of pages without links

    @param alpha a value between 0 and 1. Determines the relative
    importance of "stochastic" links.

    @param convergence a relative convergence criterion. Smaller
    means better, but more expensive.

    @param checkSteps check for convergence after so many steps
    """

    # the number of "pages"
    N = len(At)

    # the number of "pages without links"
    M = ln.shape[0]

    # initialize: single-precision should be good enough
    iNew = ones((N,), float64) / N
    iOld = ones((N,), float64) / N

    done = False
    num_iter = 0
    while not done:

        # normalize every now and then for numerical stability
        iNew /= sum(iNew)

        for step in range(checkSteps):
            num_iter += 1
            # swap arrays
            iOld, iNew = iNew, iOld

            # an element in the 1 x I vector. 
            # all elements are identical.
            # teleportation factor
            oneIv = (1 - alpha) * sum(iOld) / N

            # an element of the A x I vector.
            # all elements are identical.
            # leaf/dangling node matrix
            oneAv = 0.0
            if M > 0:
                # only need to update if there are dangling nodes (M)
                oneAv = alpha * sum(iOld.take(ln, axis = 0)) / N
                #print(oneAv)
            # the actual adjacency matrix
            # the elements of the H x I multiplication
            ii = 0 
            # update each row separately
            while ii < N:
                page = At[ii]
                h = 0
                if page.shape[0]:
                    h = alpha * dot(
                            iOld.take(page, axis = 0),
                            1. / numLinks.take(page, axis = 0)
                            )
                # update formula incorporating all 3 elements
                iNew[ii] = h + oneAv + oneIv
                ii += 1

        diff = sum(abs(iNew - iOld))
        done = (diff < convergence)

        yield iNew, num_iter


def transposeLinkMatrix(
        outGoingLinks = [[]]
        ):
    """
    Transpose the link matrix. The link matrix contains the pages
    each page points to. But what we want is to know which pages
    point to a given page, while retaining information about how
    many links each page contains (so store that in a separate
    array), as well as which pages contain no links at all (leaf
    nodes).

    @param outGoingLinks outGoingLinks[ii] contains the indices of
    pages pointed to by page ii

    @return a tuple of (incomingLinks, numOutGoingLinks, leafNodes)
    """

    nPages = len(outGoingLinks)
    # incomingLinks[ii] will contain the indices jj of the pages
    # linking to page ii
    incomingLinks = [[] for ii in range(nPages)]
    # the number of links in each page
    numLinks = zeros(nPages, int64)
    # the indices of the leaf nodes
    leafNodes = []

    for ii in range(nPages):
        if len(outGoingLinks[ii]) == 0:
            leafNodes.append(ii)
        else:
            numLinks[ii] = len(outGoingLinks[ii])
            # transpose the link matrix
            for jj in outGoingLinks[ii]:
                incomingLinks[jj].append(ii)

    incomingLinks = [array(ii) for ii in incomingLinks]
    numLinks = array(numLinks)
    leafNodes = array(leafNodes)

    return incomingLinks, numLinks, leafNodes


def pageRank_naive_numba(
        linkMatrix = [[]],
        alpha = 0.85, 
        convergence = 0.0001, 
        checkSteps = 10
        ):
    """
    Convenience wrap for the link matrix transpose and the generator.

    @see pageRankGenerator for parameter description
    """
    incomingLinks, numLinks, leafNodes = transposeLinkMatrix(linkMatrix)
    num_iter = 0
    for gr, i in pageRankGenerator(incomingLinks, numLinks, leafNodes,
                                alpha = alpha, convergence = convergence,
                                checkSteps = checkSteps):
        final = gr
        num_iter = i
    return final, num_iter


###########################################################


from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

@jit()
def pagerank_sparse_numba(H, alpha=0.85, check_iter=5, max_iter=30, convergence=1e-4):
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

