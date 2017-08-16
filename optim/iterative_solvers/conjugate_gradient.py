#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from ..line_search import interpolating_line_search, CG_C2

EPS = 1e-8

def fletcher_reeves(dfval_old, dfval, pval):
	return (dfval.T * dfval) / ((dfval_old.T * dfval_old) + EPS)

def polak_ribiere(dfval_old, dfval, pval):
	return (dfval.T * (dfval- dfval_old)) / ((norm(dfval_old) ** 2) + EPS)

def hestenes_stiefel(dfval_old, dfval, pval):
	delta_dfval = dfval- dfval_old
	return -1. * (dfval.T * delta_dfval) / ((pval.T * delta_dfval) + EPS)

def dai_yuan(dfval_old, dfval, pval):
	delta_dfval = dfval- dfval_old
	return (dfval.T * dfval) / ( (pval.T * delta_dfval) + EPS)

DIR_STEPSIZE = {"FR" : fletcher_reeves, \
			    "PR" : polak_ribiere, \
			    "HS" : hestenes_stiefel, \
			    "DY" : dai_yuan}

def fmin_cg(f, x0, df, tol=1e-7, mode="PR", maxiter=1000, restart_enabled=False):
	"""
	Minimize a function using non-linear conjugate gradient.
	direction step_size (parameterized by "mode") defaults to polak_ribiere

	Parameters
    ----------
    f : callable
        function to evaluate
    x0 : array_like
        current point
    df : callable
        derivative function of f
    tol : float, optional
        testing for solution's convergence
    mode : str, optional
        for computing step size of new direction vector. Defaults to polak ribiere
    max_iter : float, optional
    	max iterations we should run the algorithm for
    restart_enabled : bool, optional
    	whether we should run a restart when 2 consecutive gradients are far from orthogonal to each other.
	"""
	def should_restart(dfval, dfval_old, restart_tol=0.1):
		return np.abs(np.dot(dfval.T, dfval_old)) / (np.linalg.norm(dfval) ** 2.0) >= restart_tol

	if mode not in DIR_STEPSIZE:
		print("couldn't find %s, defaulting to 'PR'" % mode)
		mode = "PR"
	compute_dir_stepsize = DIR_STEPSIZE[mode]
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
		beta = compute_dir_stepsize(dfval_old, dfval, pval)
		if restart_enabled and should_restart(dfval, dfval_old):
			beta = 0.0 # set pval in next line to be just steepest descent
		pval = -dfval + beta * pval

		if np.abs(np.sum(x - x_old)) < tol:
			break
		k += 1
	return x