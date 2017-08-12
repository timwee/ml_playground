#! /usr/bin/env python

from scipy import linalg
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

dtype = np.float64
def streaming_block_power_iter(A, n_components, block_size=100, n_iter=8):
	m, n = A.shape
	# bec. this is a sketching method, oversample (see Halko or Tygert paper)
	oversample_rank = max(int(n_components * 1.5), 12)

	y = np.zeros(dtype=dtype, shape=(n, oversample_rank))
	for band in A[0:-1:block_size,:]:
		a = band.shape[0]
		#assert b==n
		o = np.random.normal(0.0, 1.0, (a, oversample_rank)).astype(dtype)
		y = y +	safe_sparse_dot(band.T, o)

	q, _ = linalg.qr(y)

	for power_iter in range(n_iter):
		q_old = q.copy()
		q[:] = 0.0
		for band in A[0:-1:block_size,:]:
			q += band.T * band * q_old
		q, _ = linalg.qr(q)
	# shape - (oversample_rank, n)
	qt = q.T.copy()
	del q
	print("qt.shape=", qt.shape)

	# Step2: project A onto orthonormal basis Q
	# Normally, B= Q.T @ A
	B = np.zeros(shape=(qt.shape[0], n), dtype=dtype)
	for band in A[0:-1:block_size,:]:
		B += safe_sparse_dot(qt, band)
		#B += safe_sparse_dot(b, b.T)
	print("B.shape= ", B.shape)
	u, s, vt = linalg.svd(B)
	s = np.sqrt(s)

	q = qt.T.copy()
	del qt
	u = safe_sparse_dot(q, u)
	vt = vt.dot(q.conj().T)
	print(u.shape, s.shape, vt.shape)
	return u.astype(dtype), s.astype(dtype), vt.astype(dtype)
