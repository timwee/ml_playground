#!/usr/bin/env python

import numpy as np

DEFAULT_C1 = 1e-4
EPS = 1e-7

NEWTON_C2 = 0.9
CG_C2 = 0.1 # better c2 for conjugate gradient according to Nocedal/Wright
DEFAULT_C2 = NEWTON_C2

def backtracking_linesearch(f, df, x, p, **kwargs):
    """
    Convenience wrapper for _backtracking_linesearch. Takes care of creating the phi function:
    phi(alpha) = f(x + alpha * p)

    Parameters
    ----------
    alpha_init : initial value for factor
    f : callable
        function to evaluate, this should be f(alpha) = g(x + alpha * pk), where pk is the direction vector/array
        f is expected to only take alpha, since that's the only thing that changes in this algorithm
    df : callable
        derivative function of f
    x : array_like
        current point at which we are doing line search
    p : array_like
        direction vector
    c1 : float, optional
        typically 1e-4
        factor for the sufficient decrease/Armijo condition
    c2 : float, optional
        typically 0.1 for nonlinear conjugate gradient, 0.9 for newton and quasi-newton
        factor for the curvature condition
    alpha_init: float, optional
        initial point for alpha, the value we are optimizing for

    Returns
    ----------
    (alpha, fval, dfval) : tuple
        alpha : float
            how far to step in search direction - xk + alpha * pk. None if line search is unsuccessful
        fval : float
            function value at alpha
        dfval : float
            derivative value at alpha
    func_calls : float
        number of times phi was called, mainly for testing/debugging purposes
    dfunc_calls : float
        number of times dphi was called, mainly for testing/debugging purposes

    """
    func_calls = [0]
    dfunc_calls = [0]
    def phi(alphak):
        func_calls[0] += 1
        return f(x + alphak * p)
    def dphi(alphak):
        dfunc_calls[0] += 1
        return np.dot(df(x + alphak * p), p)
    return _backtracking_linesearch(phi, dphi, **kwargs), func_calls[0], dfunc_calls[0]




def _backtracking_linesearch(f, df, alpha_init=1, max_iter=40, shrink_factor=0.5, grow_factor=2.1, c1=DEFAULT_C1, \
                            c2=DEFAULT_C2, min_step_size=1e-10, max_step_size=100, \
                            use_wolfe=True, use_strong_wolfe=True, decay=None, decay_iter_start=5):
    """
    Parameters
    ----------
    alpha_init : initial value for factor
    f : callable
        function to evaluate, this should be f(alpha) = g(x + alpha * pk), where pk is the direction vector/array
        f is expected to only take alpha, since that's the only thing that changes in this algorithm
    df : callable
        derivative function of f, projected onto direction pk.
    c1 : float, optional
        typically 1e-4
        factor for the sufficient decrease/Armijo condition
    c2 : float, optional
        typically 0.1 for nonlinear conjugate gradient, 0.9 for newton and quasi-newton
        factor for the curvature condition

    Returns
    ----------
    alpha : float
        how far to step in search direction - xk + alpha * pk
    fk : float
        function value at current alpha
    dfk : float
        derivative value at current alpha

    Conditions:
    0 < c1 < c2 < 1
    """
    assert shrink_factor * grow_factor != 1.0
    assert c1 < 0.5
    assert c1 > 0.0
    assert c2 > c1
    assert c2 < 1.0

    f0, df0 = f(0.0), df(0.0)
    fk, dfk = f(alpha_init), df(alpha_init)
    alpha = alpha_init
    #print("k, alpha, fk, f0, c1, df0, f0 + c1 * alpha * df0")
    for k in range(max_iter):
        if decay and k > decay_iter_start:
            it = k - decay_iter_start
            grow_factor *= max(1.0, grow_factor * 1. / (1. + (decay * it)))
            shrink_factor *= min(1.0, shrink_factor * (1. + (decay * it)/ 1.))
        fk = f(alpha)
        dfk = df(alpha)
        # pheta in nocedal wright, for changing alpha
        alpha_mult = 1.0
        # sufficient decrease condition
        #print(k, alpha, fk, f0, c1, df0, f0 + c1 * alpha * df0, "before sufficient decrease")
        if fk > (f0 + c1 * alpha * df0):
            alpha_mult = shrink_factor
        # curvature condition
        elif use_wolfe and dfk < c2 * df0:
            alpha_mult = grow_factor
        elif use_strong_wolfe and dfk > -c2 * df0:
            alpha_mult = shrink_factor
        else:
            # converged
            #print("converged")
            break
        alpha *= alpha_mult
        if alpha < min_step_size:
            #print("Step size got too small in backtracking line search")
            return None, fk, dfk
        elif alpha > max_step_size:
            #print("Step size got too big in backtracking line search")
            return None, fk, dfk
    else:
        #print("line search didn't converge")
        return None, fk, dfk
    return alpha, fk, dfk


#########################################################################################################

def interpolating_line_search(f, df, x, p, **kwargs):
    """
    from Nocedal and Wright book.
    This does not do curvature and strong wolfe checks, just armajilo/sufficient decrease

    Parameters
    ----------
    alpha_init : initial value for factor
    f : callable
        function to evaluate, this should be f(alpha) = g(x + alpha * pk), where pk is the direction vector/array
        f is expected to only take alpha, since that's the only thing that changes in this algorithm
    df : callable
        derivative function of f
    x : array_like
        current point at which we are doing line search
    p : array_like
        direction vector
    c1 : float, optional
        typically 1e-4
        factor for the sufficient decrease/Armijo condition
    """
    func_calls = [0]
    dfunc_calls = [0]
    def phi(alphak):
        func_calls[0] += 1
        return f(x + alphak * p)
    def dphi(alphak):
        dfunc_calls[0] += 1
        return np.dot(df(x + alphak * p), p)
    return _interpolating_line_search(phi, dphi, **kwargs), func_calls[0], dfunc_calls[0]

def _interpolating_line_search(f, df, alpha_init=1, max_iter=40, c1=DEFAULT_C1, \
                                c2=DEFAULT_C2, min_step_size=1e-10, max_step_size=100, **kwargs):
    assert c1 < 0.5
    assert c1 > 0.0
    assert c2 > c1
    assert c2 < 1.0

    f0, df0 = f(0.0), df(0.0)
    fk, dfk = f(alpha_init), df(alpha_init)
    alpha0 = alpha_init
    if fk <= (f0 + c1 * alpha0 * df0):
        # satisfied condition
        return alpha0, fk, dfk

    # quadratic interpolation
    alpha_quad = - (df0 * (alpha0 ** 2.0)) / \
                    (2.0 * (fk - f0 - (df0 * alpha0)))
    fquad = f(alpha_quad)
    dfquad = df(alpha_quad)
    if fquad <= (f0 + c1 * alpha0 * df0):
        return alpha_quad, fquad, dfquad


    while alpha_quad > min_step_size:
        # do cubic interpolation
        denom = alpha0 ** 2.0 * alpha_quad ** 2.0 * (alpha_quad - alpha0)
        row1 = (fquad - f0 - (df0 * alpha_quad))
        row2 = (fk - f0 - df0 * alpha0)
        a = (((alpha0 ** 2.0) * row1) + \
            (-(alpha_quad ** 2.0) * row2)) / denom
        b = ((-(alpha0 ** 3.0) * row1) + \
            ((alpha_quad ** 3.0) * row2)) / denom
        alpha_cubic = (-b + np.sqrt(abs(b**2 - 3 * a * df0))) / (3.0*a + EPS)

        fcubic = f(alpha_cubic)
        dfcubic = df(alpha_cubic)

        if fcubic <= f0 + c1 * alpha_cubic * df0:
            return alpha_cubic, fcubic, dfcubic

        # From Nocedal and Wright:
        # If the new alpha is too close to its predecessor or else too much smaller than
        #   the predecessor, we reset alpha to be predecessor/2.0
        # This safeguard procedures ensures that we make reasonable progress on each iteration
        #    and that the final alpha is not too small
        if (alpha_quad - alpha_cubic) > (alpha_quad) / 2.0 or \
            (1 - (alpha_cubic / alpha_quad)) < 0.96:
            alpha_cubic = alpha_quad / 2.0

        # replace predecessor estimates with new
        # cubic interpolation works by keeping last 2 estimates updated.
        alpha0 = alpha_quad
        alpha_quad = alpha_cubic
        fk = fquad
        fquad = fcubic
        dfk = dfquad
        dfquad = dfcubic
    return None, fquad, dfquad


