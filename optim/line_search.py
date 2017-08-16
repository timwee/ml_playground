#!/usr/bin/env python

import numpy as np
from math import sqrt

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

#########################################################################################################

def strong_wolfe_with_zoom(f, df, x, p, **kwargs):
    func_calls = [0]
    dfunc_calls = [0]
    def phi(alphak):
        func_calls[0] += 1
        return f(x + alphak * p)
    def dphi(alphak):
        dfunc_calls[0] += 1
        return np.dot(df(x + alphak * p), p)
    return _strong_wolfe_with_zoom(phi, dphi, **kwargs), func_calls[0], dfunc_calls[0]

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found return None
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _cubic_interpolate(alpha_0, alpha_1, f_0, f_1, df_0, df_1):
    """ 
    pg. 59 from Nocedal/Wright 
    we do a subscript of 0 for i, and 1 for i+1
    """
    d1 = df_0 + (df_1 - (3 * (f_0 - f_1) / (alpha_0 - alpha_1)))
    # in the book, they have a sign of (alpha_i - alpha_i-1), but we ensure that this is always positive
    d2 = sqrt((d1 ** 2.0) - (df_0 * df_1))
    alpha_j = alpha_1 - (alpha_1 - alpha_0) * \
        ((df_1 + d2 - d1) / (df_1 - df_0 + 2 * d2))

    dalpha = (alpha_1 - alpha_0)
    min_allowed = alpha_0 + 0.05 * dalpha
    max_allowed = alpha_0 + 0.95 * dalpha
    if alpha_j < min_allowed:
        return min_allowed
    elif alpha_j > max_allowed:
        return max_allowed
    return alpha_j

def _zoom_interpolate(i, alpha_lo, alpha_hi, alpha_0, f_lo, f_hi, f_0, df_lo, df_hi, df_0, alpha_rec, f_rec):
    #return _cubic_interpolate(alpha_lo, alpha_hi, f_lo, f_hi, df_lo, df_hi)
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    
    dalpha = alpha_hi - alpha_lo
    a, b = alpha_lo, alpha_hi
    if dalpha < 0:
        a, b = alpha_hi, alpha_lo

    a_j = None
    if (i > 0):
        cchk = delta1 * dalpha
        a_j = _cubicmin(alpha_lo, f_lo, df_lo, alpha_hi, f_hi,
                        alpha_rec, f_rec)
    if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
        qchk = delta2 * dalpha
        a_j = _quadmin(alpha_lo, f_lo, df_lo, alpha_hi, f_hi)
        if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
            a_j = alpha_lo + 0.5*dalpha
    return a_j

def _zoom(f, df, alpha_lo, alpha_hi, alpha_0, f_lo, f_hi, f_0, df_lo, df_hi, df_0, c1, c2, max_zoom_iter=40):
    """
    From Nocedal/Wright:
    We now specify the function zoom, which requires a little explanation. The order of its 
    input arguments is such that each call has the form zoom(αlo,αhi), where
    (a) the interval bounded by αlo and αhi contains step lengths that satisfy the strong Wolfe conditions;
    (b) αlo is, among all step lengths generated so far and satisfying the sufficient decrease condition, 
        the one giving the smallest function value; and
    (c) αhi is chosen so that φ (αlo)(αhi − αlo) < 0.
    Each iteration of zoom generates an iterate αj between αlo and αhi, and then replaces one
    of these end points by αj in such a way that the properties(a), (b), and (c)continue to hold.
    Algorithm 3.6 (zoom). repeat
        Interpolate (using quadratic, cubic, or bisection) to find 
            a trial step length αj between αlo and αhi;
        Evaluate φ(αj );
        if φ(αj ) > φ(0) + c1 * αj * φ'(0) or φ(αj ) ≥ φ(αlo)
             αhi ←αj; 
        else
            Evaluate φ'(αj );
            if |φ'(αj )| ≤ −c2φ'(0)
                Set α∗ ← αj and stop; 
            if φ'(αj)(αhi −αlo) ≥ 0
                αhi ← αlo;
            αlo ←αj;
    end (repeat)
    """
    
    alpha_rec = 0
    f_rec = f_0
    for i in range(max_zoom_iter):
        if alpha_lo > alpha_hi:
            alpha_j = _zoom_interpolate(i, alpha_hi, alpha_lo, alpha_0, f_hi, f_lo, f_0, df_hi, df_lo, df_0, alpha_rec, f_rec)
        else:
            alpha_j = _zoom_interpolate(i, alpha_lo, alpha_hi, alpha_0, f_lo, f_hi, f_0, df_lo, df_hi, df_0, alpha_rec, f_rec)
        f_j = f(alpha_j)
        df_j = df(alpha_j)
        if f_j > (f_0 + c1 * alpha_j * df_0) or f_j >= f_lo:
            alpha_rec = alpha_hi
            f_rec = f_hi
            alpha_hi = alpha_j
            f_hi = f_j
            df_hi = df_j
        else:
            if abs(df_j) <= -c2 * df_0:
                return alpha_j
            if df_j * (alpha_hi - alpha_lo) >= 0:
                f_rec = f_hi
                alpha_rec = alpha_hi
                alpha_hi = alpha_lo
                f_hi = f_lo
                df_hi = df_lo
            else:
                alpha_rec = alpha_lo
                f_rec = f_lo
            alpha_lo = alpha_j
            f_lo = f_j
            df_lo = df_j
    print("Didn't find appropriate value in zoom")
    return None


def _strong_wolfe_with_zoom(f, df, alpha_init=0, max_iter=100, c1=DEFAULT_C1, \
                            c2=DEFAULT_C2, min_step_size=1e-10, max_step_size=1e10, **kwargs):
    f0, df0 = f(0.0), df(0.0)
    alphaprev = 0.0

    # In Nocedal, Wright, this was left unspecified.
    alphak = 1.0
    while np.isnan(f(alphak)):
        # in case we set this to be too large
        alphak /= 2.

    if alphak <= min_step_size:
        print("alphak is <= min_step_size before loop!")
        return None, f0, df0
    fprev, dfprev = f0, df0
    fk, dfk = f(alphak), df(alphak)
    for i in range(max_iter):
        if fk > (f0 + c1 * alphak * df0) or fk >= fprev and i > 0:
            alpha_zoom = _zoom(f, df, alphaprev, alphak, alpha_init, fprev, fk, f0, \
                dfprev, dfk, df0, c1, c2)
            if not alpha_zoom:
                alpha_zoom = alphak
            return alpha_zoom, f(alpha_zoom), df(alpha_zoom)
        if np.abs(dfk) <= -c2 * df0:
            return alphak, fk, dfk
        if dfk >= 0:
            alpha_zoom = _zoom(f, df, alphak, alphaprev, alpha_init, fk, fprev, f0, \
                dfk, dfprev, df0, c1, c2)
            if not alpha_zoom:
                alpha_zoom = alphak
            return alpha_zoom, f(alpha_zoom), df(alpha_zoom)
        alphaprev = alphak 
        alphak *= 1.25
        fprev = fk
        fk = f(alphak)
        dfprev = dfk
        dfk = df(alphak)
        if alphak > max_step_size:
            print("reached max step size")
            return max_step_size, fk, dfk
    else:
        print("didn't converge")
        return None, fk, dfk
    return alphak, fk, dfk




















