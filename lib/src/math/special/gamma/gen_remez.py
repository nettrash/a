"""A small Remez exchange for minimax polynomials in mpmath.

fit(f, a, b, n, w=None, basis=None) minimises max |w(x) (sum c_k phi_k(x) - f(x))| over [a, b]
with phi_k(x) = x^k (k = 0..n) unless `basis` gives callables. Returns (coeffs, maxerr).
f and w take and return mpmath numbers.
"""

import mpmath
from mpmath import mp, mpf


def _solve(f, w, basis, xs):
    n = len(basis)
    A = mpmath.matrix(n + 1, n + 1)
    rhs = mpmath.matrix(n + 1, 1)
    for i, x in enumerate(xs):
        for j, ph in enumerate(basis):
            A[i, j] = ph(x)
        A[i, n] = (-1) ** i / w(x)
        rhs[i] = f(x)
    sol = mpmath.lu_solve(A, rhs)
    return [sol[j] for j in range(n)], sol[n]


def _err(f, w, basis, c, x):
    return w(x) * (sum(ck * ph(x) for ck, ph in zip(c, basis)) - f(x))


def fit(f, a, b, n=None, w=None, basis=None, iters=30, grid=4000, verbose=False, fvals=None):
    a, b = mpf(a), mpf(b)
    if basis is None:
        basis = [(lambda k: (lambda x: x ** k))(k) for k in range(n + 1)]
    if w is None:
        w = lambda x: mpf(1)
    m = len(basis)
    # Chebyshev extrema as the first reference.
    xs = [(a + b) / 2 - (b - a) / 2 * mpmath.cos(mpmath.pi * i / m) for i in range(m + 1)]
    # Cache f on a grid (Chebyshev-like density at the ends).
    gx = [(a + b) / 2 - (b - a) / 2 * mpmath.cos(mpmath.pi * i / grid) for i in range(grid + 1)]
    fg = fvals if fvals is not None else [f(x) for x in gx]
    wg = [w(x) for x in gx]
    cache = dict(zip(gx, fg))
    fc = lambda x: cache[x] if x in cache else f(x)
    best = None
    for it in range(iters):
        c, E = _solve(fc, w, basis, xs)
        e = [wg[i] * (sum(ck * ph(x) for ck, ph in zip(c, basis)) - fg[i]) for i, x in enumerate(gx)]
        # Local extrema of e on the grid, alternating signs.
        ext = []
        for i in range(len(gx)):
            l = e[i - 1] if i > 0 else None
            r = e[i + 1] if i + 1 < len(gx) else None
            v = e[i]
            if (l is None or abs(v) >= abs(l)) and (r is None or abs(v) >= abs(r)):
                ext.append(i)
        # Merge same-sign runs keeping the largest.
        merged = []
        for i in ext:
            if merged and (e[merged[-1]] >= 0) == (e[i] >= 0):
                if abs(e[i]) > abs(e[merged[-1]]):
                    merged[-1] = i
            else:
                merged.append(i)
        # Keep m+1 points: drop the smallest from the ends while too many.
        while len(merged) > m + 1:
            if abs(e[merged[0]]) < abs(e[merged[-1]]):
                merged.pop(0)
            else:
                merged.pop()
        mx = max(abs(v) for v in e)
        if verbose:
            print(it, mpmath.nstr(abs(E), 6), mpmath.nstr(mx, 6), len(merged))
        if best is None or mx < best[1]:
            best = (c, mx)
        if len(merged) < m + 1:
            break
        newxs = [gx[i] for i in merged]
        if mx <= abs(E) * (1 + mpf("1e-4")):
            break
        xs = newxs
    return best


def to_f64(c):
    return [float(x) for x in c]


def check(f, a, b, coeffs, w=None, basis=None, n=20000, ev=None):
    """Max weighted error of float coefficients evaluated exactly (in mpmath)."""
    a, b = mpf(a), mpf(b)
    if basis is None:
        basis = [(lambda k: (lambda x: x ** k))(k) for k in range(len(coeffs))]
    if w is None:
        w = lambda x: mpf(1)
    mx = 0
    for i in range(n + 1):
        x = a + (b - a) * i / n
        v = sum(mpf(ck) * ph(x) for ck, ph in zip(coeffs, basis))
        mx = max(mx, abs(w(x) * (v - f(x))))
    return mx
