"""Minimax polynomial fits (relative or absolute error) with mpmath, for the Bessel/Airy tables.

fit(f, a, b, deg, center=None, rel=True, grid=1500) -> (coeffs as floats, max error measured with
the f64 coefficients evaluated exactly in mp).

The polynomial is in the variable u = x - center (center defaults to a), monomial basis.
Optionally `even=True` or `odd=True` fits a polynomial in u**2 (times u for odd) and `skip`
fixes the low coefficients to given values.
"""

import mpmath
from mpmath import mpf

mp = mpmath.mp


def _solve(A, b):
    return mpmath.lu_solve(mpmath.matrix(A), mpmath.matrix(b))


def fit(f, a, b, deg, center=None, rel=True, grid=1500, iters=12, basis=None, weight=None,
        fixed=None, verbose=False, exact=0, want_mp=False):
    """Remez fit of `f` on [a, b] by sum_k c_k * phi_k(u), u = x - center.

    basis: list of exponents (default 0..deg). weight(x): error weight (default 1/|f| if rel).
    fixed: dict exponent -> value: terms subtracted from f before fitting (not refit).
    """
    a, b = mpf(a), mpf(b)
    c = a if center is None else mpf(center)
    exps = list(range(deg + 1)) if basis is None else list(basis)
    n = len(exps)
    fixed = fixed or {}
    xs = [a + (b - a) * (1 - mpmath.cos(mpmath.pi * i / (grid - 1))) / 2 for i in range(grid)]
    fv = [f(x) for x in xs]

    hs = max(abs(a - c), abs(b - c))

    def fixed_part(u):
        return sum((mpf(v) * u ** e for e, v in fixed.items()), mpf(0))

    if weight is None:
        wv = [1 / abs(v) if rel else mpf(1) for v in fv]
    else:
        wv = [weight(x) for x in xs]
    gv = [fv[i] - fixed_part(xs[i] - c) for i in range(grid)]
    # Initial reference: Chebyshev extrema, n + 1 points.
    m = n + 1
    ref = [int(round((grid - 1) * (1 - mpmath.cos(mpmath.pi * i / (m - 1))) / 2)) for i in range(m)]
    ref = sorted(set(ref))
    while len(ref) < m:
        for j in range(grid):
            if j not in ref:
                ref.append(j)
                break
        ref = sorted(ref)
    coef = None
    best = None
    for it in range(iters):
        A = []
        rhs = []
        for k, j in enumerate(ref):
            u = xs[j] - c
            A.append([(u / hs) ** e for e in exps] + [(-1) ** k / wv[j]])
            rhs.append(gv[j])
        sol = _solve(A, rhs)
        coef = [sol[i] / hs ** exps[i] for i in range(n)]
        E = abs(sol[n])
        err = []
        for j in range(grid):
            u = xs[j] - c
            p = sum(coef[i] * u ** exps[i] for i in range(n))
            err.append((p - gv[j]) * wv[j])
        mx = max(abs(e) for e in err)
        if verbose:
            print(it, float(E), float(mx))
        if best is None or mx < best[0]:
            best = (mx, coef)
        if mx <= E * (1 + mpf("1e-3")):
            break
        # New reference: the extremum of each run of one sign, then keep m alternating.
        runs = []
        j = 0
        while j < grid:
            s = err[j] >= 0
            k = j
            bj = j
            while k < grid and (err[k] >= 0) == s:
                if abs(err[k]) > abs(err[bj]):
                    bj = k
                k += 1
            runs.append(bj)
            j = k
        # Reduce to m points: drop the smallest at the ends.
        while len(runs) > m:
            if abs(err[runs[0]]) < abs(err[runs[-1]]):
                runs.pop(0)
            else:
                runs.pop()
        if len(runs) < m:
            break
        ref = runs
    coef = best[1]
    cf = [float(x) for x in coef]
    # Error with f64 coefficients (the first `exact` as double-doubles), exactly evaluated.
    def ddv(v):
        h = float(v)
        return mpf(h) + mpf(float(v - h))
    cr = [ddv(coef[i]) if i < exact else mpf(cf[i]) for i in range(n)]
    mx = mpf(0)
    for j in range(grid):
        u = xs[j] - c
        p = sum(cr[i] * u ** exps[i] for i in range(n))
        mx = max(mx, abs((p - gv[j]) * wv[j]))
    if want_mp:
        return cf, float(mx), float(best[0]), coef
    return cf, float(mx), float(best[0])


def rust_array(name, cf, per_line=3):
    s = f"const {name}: [f64; {len(cf)}] = [\n"
    for i in range(0, len(cf), per_line):
        s += "    " + " ".join(f"{x!r}," for x in cf[i:i + per_line]) + "\n"
    return s + "];\n"
