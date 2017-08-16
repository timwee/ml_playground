#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from ..line_search import interpolating_line_search, CG_C2

def fletcher_reeves(dfval_old, dfval):
	return (dfval.T * dfval) / (dfval_old.T * dfval_old)

def polak_ribiere(dfval_old, dfval):
	return (dfval.T * (dfval- dfval_old)) / (norm(dfval_old) ** 2)

def fmin_cg(f, x0, df, tol=1e-4, maxiter=1000, restart_enabled=False):
	"""
	Minimize a function using non-linear conjugate gradient
	"""
	def should_restart(dfval, dfval_old, restart_tol=0.1):
		return np.abs(np.dot(dfval.T, dfval_old)) / (np.linalg.norm(dfval) ** 2.0) >= restart_tol


	fval = f(x0)
	dfval = df(x0)
	dfval_old = dfval
	pval = -dfval
	x = x0
	x_old = x
	k = 1
	default_alpha = 1.
	while k <= maxiter:
		(alpha, _, _), _, _ = interpolating_line_search(f, df, x, pval, c2=CG_C2)
		if not alpha:
			alpha = default_alpha
		x_old = x
		x  = x + alpha * pval
		dfval_old = dfval
		dfval = df(x)
		beta = polak_ribiere(dfval_old, dfval)
		if restart_enabled and should_restart(dfval, dfval_old):
			beta = 0.0 # set pval in next line to be just steepest descent
		pval = -dfval + beta * pval

		if np.abs(np.sum(x - x_old)) < tol:
			break
		k += 1
	return x