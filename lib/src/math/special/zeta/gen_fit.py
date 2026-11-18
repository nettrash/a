"""The fitting tools of the zeta area's generators (gen_*.py in this directory): polynomial
minimax by Remez's exchange and rational near-minimax by Lawson's iteration on a linearised
least-squares fit, with mpmath. Run the generators from this directory (`python3 gen_lambert.py`);
they print the Rust tables. Python's byte-code cache goes to the temporary directory, never here.
"""

import os
import sys
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "a-special-zeta-pycache")


import mpmath
from mpmath import mpf

mp = mpmath.mp


def _extrema(err, a, b, n, npts):
    xs = [a + (b - a) * (1 - mpmath.cos(mpmath.pi * i / (npts - 1))) / 2 for i in range(npts)]
    es = [err(x) for x in xs]
    # local extrema of |e| with sign changes: split into runs of the same sign, take the max of each
    runs = []
    cur = None
    for x, e in zip(xs, es):
        s = 1 if e >= 0 else -1
        if cur is None or s != cur[0]:
            cur = [s, x, e]
            runs.append(cur)
        elif abs(e) > abs(cur[2]):
            cur[1], cur[2] = x, e
    # refine each by golden section on a small bracket
    out = []
    for s, x, e in runs:
        h = (b - a) / npts * 2
        lo, hi = max(a, x - h), min(b, x + h)
        g = lambda t: -abs(err(t))
        for _ in range(30):
            m1 = lo + (hi - lo) * mpf("0.381966")
            m2 = lo + (hi - lo) * mpf("0.618034")
            if g(m1) < g(m2):
                hi = m2
            else:
                lo = m1
        xm = (lo + hi) / 2
        em = err(xm)
        if abs(em) < abs(e):
            xm, em = x, e
        out.append((xm, em))
    return out


def remez(f, a, b, deg, rel=True, x0=0, fixed=(), iters=30, npts=600, verbose=False, w=None):
    a, b, x0 = mpf(a), mpf(b), mpf(x0)
    nf = len(fixed)
    nfree = deg + 1 - nf
    fv = {}

    def F(x):
        k = x
        if k not in fv:
            fv[k] = f(x)
        return fv[k]

    def W(x):
        if w is not None:
            return w(x)
        return abs(F(x)) if rel else mpf(1)

    def base(x):
        u = x - x0
        return sum(mpf(c) * u**k for k, c in enumerate(fixed))

    # initial nodes: Chebyshev extrema
    m = nfree + 1
    nodes = [a + (b - a) * (1 - mpmath.cos(mpmath.pi * i / (m - 1))) / 2 for i in range(m)]
    coeffs = None
    best = None
    for it in range(iters):
        A = mpmath.matrix(m, m)
        rhs = mpmath.matrix(m, 1)
        for i, x in enumerate(nodes):
            u = x - x0
            for j in range(nfree):
                A[i, j] = u ** (nf + j)
            A[i, nfree] = (-1) ** i * W(x)
            rhs[i] = F(x) - base(x)
        sol = mpmath.lu_solve(A, rhs)
        c = list(fixed) + [sol[j] for j in range(nfree)]

        def p(x):
            u = x - x0
            return mpmath.polyval(c[::-1], u)

        err = lambda x: (p(x) - F(x)) / W(x)
        ex = _extrema(err, a, b, nfree, npts)
        emax = max(abs(e) for _, e in ex)
        if best is None or emax < best[1]:
            best = (c, emax)
        if verbose:
            print(it, mpmath.nstr(emax, 5), mpmath.nstr(abs(sol[nfree]), 5), len(ex))
        if len(ex) < m:
            break
        # choose m alternating points with the largest errors (keep the global max)
        while len(ex) > m:
            # drop the smaller end or merge
            if abs(ex[0][1]) < abs(ex[-1][1]):
                ex.pop(0)
            else:
                ex.pop()
        nodes = [x for x, _ in ex]
        if abs(emax - abs(sol[nfree])) < abs(sol[nfree]) * mpf("1e-4"):
            break
    return best


def to_f64(c):
    return [float(x) for x in c]


def check_f64(f, coeffs, a, b, x0=0, n=4000, rel=True):
    """Max error of the f64-rounded polynomial (evaluated in mp) against f."""
    cs = [mpf(float(x)) for x in coeffs]
    mx = 0
    for i in range(n + 1):
        x = mpf(a) + (mpf(b) - mpf(a)) * i / n
        v = mpmath.polyval(cs[::-1], x - mpf(x0))
        t = f(x)
        e = abs(v - t) / (abs(t) if rel else 1)
        mx = max(mx, e)
    return mx


def rs(c):
    """Rust literals."""
    return ", ".join(repr(float(x)) for x in c)


def remez_rat(f, a, b, m, n, rel=True, x0=0, iters=40, npts=800, verbose=False, init_q=None):
    """Minimax rational p/q (deg m, n; q[0] = 1) in u = x - x0. Returns (p, q, err)."""
    a, b, x0 = mpf(a), mpf(b), mpf(x0)
    N = m + n + 2
    nodes = [a + (b - a) * (1 - mpmath.cos(mpmath.pi * i / (N - 1))) / 2 for i in range(N)]
    cache = {}

    def F(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]

    def W(x):
        return abs(F(x)) if rel else mpf(1)

    best = None
    qc = init_q[:] if init_q else [mpf(1)] + [mpf(0)] * n
    E = mpf(0)
    for it in range(iters):
        for inner in range(12):
            A = mpmath.matrix(N, N)
            rhs = mpmath.matrix(N, 1)
            for i, x in enumerate(nodes):
                u = x - x0
                qold = mpmath.polyval(qc[::-1], u)
                for j in range(m + 1):
                    A[i, j] = u**j
                for j in range(1, n + 1):
                    A[i, m + j] = -F(x) * u**j
                A[i, m + n + 1] = -((-1) ** i) * W(x) * qold
                rhs[i] = F(x)
            try:
                sol = mpmath.lu_solve(A, rhs)
            except ZeroDivisionError:
                return best
            pc = [sol[j] for j in range(m + 1)]
            qn = [mpf(1)] + [sol[m + j] for j in range(1, n + 1)]
            En = sol[m + n + 1]
            done = abs(En - E) <= abs(En) * mpf("1e-10")
            qc, E = qn, En
            if done:
                break

        def r(x):
            u = x - x0
            return mpmath.polyval(pc[::-1], u) / mpmath.polyval(qc[::-1], u)

        err = lambda x: (r(x) - F(x)) / W(x)
        ex = _extrema(err, a, b, N - 1, npts)
        emax = max(abs(e) for _, e in ex)
        # q must not vanish on [a, b]
        ok = all(mpmath.polyval(qc[::-1], a + (b - a) * k / 200 - x0) > 0 for k in range(201))
        if ok and (best is None or emax < best[2]):
            best = (pc[:], qc[:], emax)
        if verbose:
            print(it, mpmath.nstr(emax, 5), mpmath.nstr(abs(E), 5), len(ex), ok)
        if len(ex) < N:
            break
        while len(ex) > N:
            if abs(ex[0][1]) < abs(ex[-1][1]):
                ex.pop(0)
            else:
                ex.pop()
        nodes = [x for x, _ in ex]
        if abs(emax - abs(E)) < abs(E) * mpf("1e-3"):
            break
    return best


def check_rat_f64(f, p, q, a, b, x0=0, n=3000, rel=True):
    ps = [mpf(float(x)) for x in p]
    qs = [mpf(float(x)) for x in q]
    mx = 0
    for i in range(n + 1):
        x = mpf(a) + (mpf(b) - mpf(a)) * i / n
        u = x - mpf(x0)
        v = mpmath.polyval(ps[::-1], u) / mpmath.polyval(qs[::-1], u)
        t = f(x)
        mx = max(mx, abs(v - t) / (abs(t) if rel else 1))
    return mx


def lawson_rat(f, a, b, m, n, rel=True, x0=0, npts=300, iters=60, verbose=False, w=None):
    """Rational p/q by Sanathanan–Koerner linearisation with Lawson's weights (near-minimax)."""
    a, b, x0 = mpf(a), mpf(b), mpf(x0)
    xs = [a + (b - a) * (1 - mpmath.cos(mpmath.pi * (i + mpf(1) / 2) / npts)) / 2 for i in range(npts)]
    xs = [a] + xs + [b]
    fs = [f(x) for x in xs]
    ws = [w(x) for x in xs] if w is not None else [abs(v) if rel else mpf(1) for v in fs]
    us = [x - x0 for x in xs]
    lw = [mpf(1)] * len(xs)
    qold = [mpf(1)] * len(xs)
    best = None
    for it in range(iters):
        A = mpmath.matrix(len(xs), m + 1 + n)
        rhs = mpmath.matrix(len(xs), 1)
        for i, u in enumerate(us):
            s = mpmath.sqrt(lw[i]) / (ws[i] * qold[i])
            for j in range(m + 1):
                A[i, j] = s * u**j
            for j in range(1, n + 1):
                A[i, m + j] = -s * fs[i] * u**j
            rhs[i] = s * fs[i]
        sol, res = mpmath.qr_solve(A, rhs)
        p = [sol[j] for j in range(m + 1)]
        q = [mpf(1)] + [sol[m + j] for j in range(1, n + 1)]
        qv = [mpmath.polyval(q[::-1], u) for u in us]
        errs = [(mpmath.polyval(p[::-1], u) / qv[i] - fs[i]) / ws[i] for i, u in enumerate(us)]
        emax = max(abs(e) for e in errs)
        ok = all(v > 0 for v in qv)
        if ok and (best is None or emax < best[2]):
            best = (p, q, emax)
        if verbose:
            print(it, mpmath.nstr(emax, 4), ok)
        qold = qv
        # Lawson update
        tot = 0
        for i in range(len(xs)):
            lw[i] = lw[i] * abs(errs[i])
            tot += lw[i]
        lw = [v / tot for v in lw]
    # true error on a finer grid
    fine = [a + (b - a) * i / 3000 for i in range(3001)]
    p, q, _ = best
    W = w if w is not None else (lambda x: abs(f(x)) if rel else mpf(1))
    e = max(abs((mpmath.polyval(p[::-1], x - x0) / mpmath.polyval(q[::-1], x - x0) - f(x)) / W(x)) for x in fine)
    return (p, q, e)
