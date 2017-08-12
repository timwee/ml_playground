#! /usr/bin/env python

from scipy import linalg
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

dtype = np.float64
def rand_block_power_iter(A, n_components, n_iter=8):
	m, n = A.shape
	# bec. this is a sketching method, oversample (see Halko or Tygert paper)
	oversample_rank = max(int(n_components * 1.5), 12)

	# Step 1: Find an orthogonal basis with oversample_rank columns (smaller than n) to project A to.
	# shape (m, oversample_rank)
	S = np.random.normal(0.0, 1.0, (n, oversample_rank)).astype(dtype)
	Y = safe_sparse_dot(A, S)

	Q, _ = linalg.qr(Y, mode='economic') # shape (m, oversample_rank)
	print("pre-power Q shape=", Q.shape)

	for power_iter in range(n_iter):
		Q = safe_sparse_dot(Q.T, A).T
		Q, _ = linalg.lu(Q, permute_l=True)
		Q = safe_sparse_dot(A, Q)
		# fbpca switches to LU sometimes
		Q, _ = linalg.qr(Q, mode="economic")

		#Q = A @ A.T @ Q #safe_sparse_dot(A.T, safe_sparse_dot(A, q))
		#print(Q.shape, A.shape)
		#Q, _ = linalg.qr(Q)

	print("post-power Q shape=", Q.shape)

	# step 2 - q at this point is the orthonormal projection of A's range.
	B = Q.T @ A

	# step 3 - do SVD on the smaller matrix B (oversample_rank, n)
	u, s, vt = linalg.svd(B, full_matrices=False)

	# step 4
	# since A ~= QQ_tA = QB = QSEV_t
	# If U = Q@S
	# then A = Q @ S @ E @ V.T 
	#   = U @ E @V.T
	u = safe_sparse_dot(Q, u)
	#vt = safe_sparse_dot(vt, Q.conj().T)
	#return u.astype(dtype), s.astype(dtype), vt.astype(dtype)

	return u[:,:n_components].astype(dtype), s[:n_components].astype(dtype), vt[:n_components,:].astype(dtype)