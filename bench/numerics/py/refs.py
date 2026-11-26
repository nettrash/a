#!/usr/bin/env python3
"""SciPy / NumPy / HiGHS references for the numerics scoreboard.

Usage: refs.py <export dir>   (the directory `numprobe export` wrote)
Prints one line per reference: key, reference, kind, ns, evaluations, note (tab-separated).
kind: "ns"      -- the reference's own time per operation, ns (median of >= 15 runs);
      "solver"  -- the compiled solver's time with the Python callback time taken out, ns; the
                   Rust side adds evaluations x its own cost per evaluation;
      "counts"  -- evaluation counts only (Python overhead dominates the time).
Everything single-threaded (OMP/OPENBLAS/MKL threads 1, HiGHS threads 1)."""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import math
import time

import numpy as np
import scipy
from scipy import integrate, interpolate, optimize, special

D = sys.argv[1]
ONLY = set(sys.argv[2:])
SAMPLES = 1 if os.environ.get("NUMPROBE_QUICK") else 15  # QUICK: a functional check, not a timing
OUT = []


def emit(key, ref, kind, ns, nfev, note):
    note = note.replace("\t", " ")
    line = f"{key}\t{ref}\t{kind}\t{ns:.6g}\t{nfev}\t{note}"
    print(line, flush=True)


def median_time(fn, samples=None, min_ns=1e6):
    """Median ns per call of fn(), each sample batched to >= 1 ms."""
    samples = SAMPLES if samples is None or SAMPLES == 1 else samples
    if SAMPLES == 1:
        min_ns = 1.0
    fn()
    t0 = time.perf_counter_ns()
    fn()
    one = max(time.perf_counter_ns() - t0, 1)
    reps = max(1, int(math.ceil(min_ns / one)))
    ts = []
    for _ in range(samples):
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            fn()
        ts.append((time.perf_counter_ns() - t0) / reps)
    ts.sort()
    return ts[len(ts) // 2]


SCIPY = f"SciPy {scipy.__version__}"
NUMPY = f"NumPy {np.__version__}"


def want(g):
    return not ONLY or g in ONLY


# ------------------------------------------------------------------------------------------------
# Linear programming: HiGHS through SciPy's bindings, timing h.run() only.
# ------------------------------------------------------------------------------------------------
def lp():
    from scipy.optimize._highspy import _core as hc

    names = sorted(f for f in os.listdir(D) if f.startswith("lp_") and f.endswith(".txt"))
    ver = None
    for fname in names:
        with open(os.path.join(D, fname)) as fh:
            tok = fh.read().split()
        it = iter(tok)
        mle, meq, n = int(next(it)), int(next(it)), int(next(it))
        rd = lambda k: np.array([float(next(it)) for _ in range(k)])
        c = rd(n)
        Ale = rd(mle * n).reshape(mle, n)
        ble = rd(mle)
        Aeq = rd(meq * n).reshape(meq, n)
        beq = rd(meq)
        lo = rd(n)
        up = rd(n)
        A = np.vstack([Ale, Aeq]) if meq else Ale
        m = A.shape[0]
        inf = hc.kHighsInf
        rl = np.concatenate([np.full(mle, -inf), beq])
        ru = np.concatenate([ble, beq])
        lp_ = hc.HighsLp()
        lp_.num_col_ = n
        lp_.num_row_ = m
        lp_.col_cost_ = c
        lp_.col_lower_ = np.where(np.isinf(lo), -inf, lo)
        lp_.col_upper_ = np.where(np.isinf(up), inf, up)
        lp_.row_lower_ = rl
        lp_.row_upper_ = ru
        # Column-wise sparse matrix of the nonzeros.
        starts, idx, val = [0], [], []
        for j in range(n):
            nz = np.nonzero(A[:, j])[0]
            idx.extend(nz.tolist())
            val.extend(A[nz, j].tolist())
            starts.append(len(idx))
        lp_.a_matrix_.format_ = hc.MatrixFormat.kColwise
        lp_.a_matrix_.start_ = np.array(starts, dtype=np.int32)
        lp_.a_matrix_.index_ = np.array(idx, dtype=np.int32)
        lp_.a_matrix_.value_ = np.array(val)
        lp_.a_matrix_.num_col_ = n
        lp_.a_matrix_.num_row_ = m
        key = "lp:" + fname[3:-4]
        for solver, label in (("simplex", "dual simplex"), ("ipm", "interior point + crossover")):
            def make():
                h = hc._Highs()
                h.setOptionValue("output_flag", False)
                h.setOptionValue("threads", 1)
                h.setOptionValue("solver", solver)
                h.passModel(lp_)
                return h
            ts = []
            info = None
            for _ in range(SAMPLES + 1):
                h = make()
                t0 = time.perf_counter_ns()
                h.run()
                ts.append(time.perf_counter_ns() - t0)
                info = h.getInfo()
                st = h.modelStatusToString(h.getModelStatus())
            ts = sorted(ts[1:])
            ver = f"HiGHS {h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"
            its = info.simplex_iteration_count if solver == "simplex" else f"{info.ipm_iteration_count} ipm + {info.crossover_iteration_count} crossover"
            emit(key, f"{ver} {label} (via SciPy)", "ns", ts[len(ts) // 2], 0,
                 f"{st}, objective {info.objective_function_value:.12g}, iterations {its}; presolve on, 1 thread; h.run() timed")


# ------------------------------------------------------------------------------------------------
# L-BFGS-B: SciPy's compiled code, the Python callback time taken out.
# ------------------------------------------------------------------------------------------------
def rosen_fg(x):
    """The extended (separable) Rosenbrock function of Moré, Garbow and Hillstrom, n even."""
    xo, xe = x[0::2], x[1::2]
    a = xe - xo**2
    b = 1.0 - xo
    f = float(np.sum(100.0 * a * a + b * b))
    g = np.empty_like(x)
    g[0::2] = -400.0 * xo * a - 2.0 * b
    g[1::2] = 200.0 * a
    return f, g


def lbfgsb():
    for prob, n, bounded in [(0, 1000, False), (0, 10000, False), (0, 100000, False), (0, 1000, True), (0, 10000, True), (5, 1000, True), (5, 100000, True)]:
        if prob == 0:
            x0 = np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
            fg = rosen_fg
            bounds = [(0.0, 0.8)] * n if bounded else None
            if bounded:
                x0 = np.full(n, 0.5)
        else:
            i = np.arange(n)
            lam = 10.0 ** (3.0 * i / (n - 1))
            cc = 2.0 * np.sin(i + 1.0)

            def fg(x, lam=lam, cc=cc):
                d = x - cc
                return float(0.5 * np.sum(lam * d * d)), lam * d

            x0 = np.zeros(n)
            bounds = [(-1.0, 1.0)] * n
        cb = [0, 0]

        def fun(x):
            t0 = time.perf_counter_ns()
            r = fg(x)
            cb[0] += time.perf_counter_ns() - t0
            cb[1] += 1
            return r

        ts, cbs = [], []
        res = None
        for _ in range(SAMPLES + 1):
            cb[0] = cb[1] = 0
            t0 = time.perf_counter_ns()
            res = optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                                    options={"maxcor": 10, "ftol": 1e7 * np.finfo(float).eps, "gtol": 1e-5, "maxiter": 15000, "maxfun": 15000})
            tot = time.perf_counter_ns() - t0
            ts.append(tot - cb[0])
            cbs.append(cb[0])
        ts = sorted(ts[1:])
        solver = ts[len(ts) // 2]
        key = f"lbfgsb:{prob}:{n}:{int(bounded)}"
        emit(key, f"{SCIPY} L-BFGS-B", "solver", solver, res.nfev,
             f"{res.nit} iterations, {res.nfev} f+g, f {res.fun:.10g}, {res.message}; Python callback time ({sorted(cbs)[len(cbs)//2]/1e6:.2f} ms) taken out, the Rust f+g cost per evaluation added")


# ------------------------------------------------------------------------------------------------
# ODEs: solve_ivp counts on the probe's problems (references from the export).
# ------------------------------------------------------------------------------------------------
def ode_problem(prob):
    if prob == 0:
        mu = 0.012277471
        mp = 1 - mu

        def f(t, y):
            a, b = y[0] + mu, y[0] - mp
            d1 = (a * a + y[1] ** 2) ** 1.5
            d2 = (b * b + y[1] ** 2) ** 1.5
            return [y[2], y[3], y[0] + 2 * y[3] - mp * a / d1 - mu * b / d2, y[1] - 2 * y[2] - mp * y[1] / d1 - mu * y[1] / d2]

        return f, None, [0.994, 0.0, 0.0, -2.00158510637908252240537862224], 0.0, 17.0652165601579625588917206249
    if prob == 1:
        e = 0.9

        def f(t, y):
            r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
            return [y[2], y[3], -y[0] / r3, -y[1] / r3]

        return f, None, [1 - e, 0.0, 0.0, math.sqrt((1 + e) / (1 - e))], 0.0, 6 * math.pi
    if prob == 2:
        def f(t, y):
            u = y[0] * y[0] * y[1]
            return [1 + u - 4 * y[0], 3 * y[0] - u]

        return f, None, [1.5, 3.0], 0.0, 20.0
    if prob == 3:
        m = 50

        def f(t, y):
            x = y[:m]
            d = np.empty(2 * m)
            d[:m] = y[m:]
            lap = -2 * x
            lap[1:] += x[:-1]
            lap[:-1] += x[1:]
            d[m:] = lap
            return d

        y0 = np.zeros(100)
        y0[:50] = np.sin(np.pi * np.arange(1, 51) / 51)
        return f, None, y0, 0.0, 10.0
    if prob == 10:
        def f(t, y):
            a = -0.04 * y[0] + 1e4 * y[1] * y[2]
            c = 3e7 * y[1] * y[1]
            return [a, -a - c, c]

        def j(t, y):
            return [[-0.04, 1e4 * y[2], 1e4 * y[1]], [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]], [0, 6e7 * y[1], 0]]

        return f, j, [1.0, 0.0, 0.0], 0.0, 1e11
    if prob == 11:
        def f(t, y):
            return [y[1], 1000.0 * (1 - y[0] ** 2) * y[1] - y[0]]

        def j(t, y):
            return [[0, 1], [-2000.0 * y[0] * y[1] - 1, 1000.0 * (1 - y[0] ** 2)]]

        return f, j, [2.0, 0.0], 0.0, 3000.0
    if prob == 12:
        def f(t, y):
            f6 = 280 * y[5] * y[7] - 1.81 * y[6]
            return [-1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007, 1.71 * y[0] - 8.75 * y[1],
                    -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4], 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
                    -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
                    -280 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6], f6, -f6]

        def j(t, y):
            J = np.zeros((8, 8))
            J[0, :3] = [-1.71, 0.43, 8.32]
            J[1, :2] = [1.71, -8.75]
            J[2, 2:5] = [-10.03, 0.43, 0.035]
            J[3, 1:4] = [8.32, 1.71, -1.12]
            J[4, 4:7] = [-1.745, 0.43, 0.43]
            J[5, 3:8] = [0.69, 1.71, -0.43 - 280 * y[7], 0.69, -280 * y[5]]
            J[6, 5:8] = [280 * y[7], -1.81, 280 * y[5]]
            J[7, 5:8] = [-280 * y[7], 1.81, -280 * y[5]]
            return J

        return f, j, [1, 0, 0, 0, 0, 0, 0, 0.0057], 0.0, 321.8122
    if prob == 13:
        s, w, q = 77.27, 0.161, 8.375e-6

        def f(t, y):
            return [s * (y[1] + y[0] * (1 - q * y[0] - y[1])), (y[2] - (1 + y[0]) * y[1]) / s, w * (y[0] - y[2])]

        def j(t, y):
            return [[s * (1 - 2 * q * y[0] - y[1]), s * (1 - y[0]), 0], [-y[1] / s, -(1 + y[0]) / s, 1 / s], [w, 0, -w]]

        return f, j, [1.0, 2.0, 3.0], 0.0, 360.0
    A = np.loadtxt(os.path.join(D, "ode_lin_A.txt")).reshape(50, 50)
    return (lambda t, y: A @ y), (lambda t, y: A), np.ones(50), 0.0, 10.0


def rs_e(t):
    """A power of ten as Rust's {:e} prints it (1e-6)."""
    return f"1e{int(round(math.log10(t)))}"


def stiff_atol(prob, rtol):
    return 1e-8 * rtol if prob == 10 else 1e-6 * rtol if prob == 14 else 1e-4 * rtol


def ode():
    refs = {}
    with open(os.path.join(D, "ode_ref.txt")) as fh:
        for line in fh:
            p, *v = line.split()
            refs[int(p)] = np.array([float(t) for t in v])
    cases = [(p, tol, m) for p in (0, 1, 2, 3) for tol in (1e-6, 1e-10) for m in ("RK45", "DOP853")]
    cases += [(p, tol, m) for p, tols in ((10, (1e-4, 1e-6, 1e-8)), (11, (1e-4, 1e-6, 1e-8)), (12, (1e-6,)), (13, (1e-6,)), (14, (1e-6,))) for tol in tols for m in ("Radau", "BDF")]
    for prob, tol, meth in cases:
        f, j, y0, t0, t1 = ode_problem(prob)
        stiff = prob >= 10
        atol = stiff_atol(prob, tol) if stiff else tol
        kw = {"jac": j} if stiff else {}
        t_start = time.perf_counter()
        r = integrate.solve_ivp(f, (t0, t1), y0, method=meth, rtol=tol, atol=atol, **kw)
        dt = time.perf_counter() - t_start
        yr = refs[prob]
        y = r.y[:, -1]
        if stiff:
            err = float(np.max(np.abs(y - yr) / (np.abs(yr) + atol)))
        else:
            err = float(np.max(np.abs(y - yr) / (1 + np.abs(yr))))
        steps = len(r.t) - 1
        ours = {"RK45": "DormandPrince5", "DOP853": "DormandPrince853", "Radau": "RadauIia", "BDF": "Bdf"}[meth]
        key = f"ode:{ours}:{prob}:{rs_e(tol)}"
        emit(key, f"{SCIPY} solve_ivp {meth}", "counts", 0, r.nfev,
             f"{steps} steps, {r.nfev} f, {r.njev} J, {r.nlu} LU, err {err:.1e}, status {r.status}; Python per step ({dt*1e3:.0f} ms total): counts only")


# ------------------------------------------------------------------------------------------------
# Least squares: MINPACK's lmder through least_squares(method="lm"), counts.
# ------------------------------------------------------------------------------------------------
def fits():
    with open(os.path.join(D, "fits.json")) as fh:
        data = json.load(fh)
    for fit in data:
        model = fit["model"]
        x = np.array(fit["x"])
        y = np.array(fit["y"])

        def mdl(b):
            if model == 0:
                e = np.exp(-b[1] * x)
                return b[0] * (1 - e), np.stack([1 - e, b[0] * x * e], 1)
            if model == 1:
                num = b[0] + b[1] * x + b[2] * x**2 + b[3] * x**3
                den = 1 + b[4] * x + b[5] * x**2 + b[6] * x**3
                return num / den, np.stack([1 / den, x / den, x**2 / den, x**3 / den, -num * x / den**2, -num * x**2 / den**2, -num * x**3 / den**2], 1)
            if model == 2:
                num = x * x + x * b[1]
                den = x * x + x * b[2] + b[3]
                return b[0] * num / den, np.stack([num / den, b[0] * x / den, -b[0] * num * x / den**2, -b[0] * num / den**2], 1)
            if model == 3:
                e4, e5 = np.exp(-x * b[3]), np.exp(-x * b[4])
                return b[0] + b[1] * e4 + b[2] * e5, np.stack([np.ones_like(x), e4, e5, -b[1] * x * e4, -b[2] * x * e5], 1)
            e = np.exp(-b[1] * x)
            return b[0] * e + b[2], np.stack([e, -b[0] * x * e, np.ones_like(x)], 1)

        r = optimize.least_squares(lambda b: mdl(b)[0] - y, np.array(fit["start"]), jac=lambda b: mdl(b)[1], method="lm")
        cert = fit["certified"]
        if cert:
            dg = min(16.0, *[(-math.log10(abs((bi - ci) / ci)) if bi != ci else 16.0) for bi, ci in zip(r.x, cert)])
            dg = f"{max(dg, 0.0):.1f} digits"
        else:
            dg = "no certified values"
        emit(f"fit:{fit['name']}", f"{SCIPY} least_squares lm (MINPACK lmder)", "counts", 0, r.nfev,
             f"{r.nfev} f, {r.njev} J, success {r.success}, {dg}; defaults xtol = ftol = gtol = 1e-8; counts only")


# ------------------------------------------------------------------------------------------------
# Cubature: scipy.integrate.cubature with the Genz–Malik rule (vectorised Python).
# ------------------------------------------------------------------------------------------------
def genz(fam, d):
    if fam == "oscillatory":
        return lambda X: np.cos(2 * np.pi * 0.3 + 1.5 * X.sum(axis=-1))
    if fam == "product_peak":
        return lambda X: np.prod(1.0 / (5.0 ** -2 + (X - 0.4) ** 2), axis=-1)
    if fam == "gaussian":
        return lambda X: np.exp(-9.0 * ((X - 0.45) ** 2).sum(axis=-1))
    return lambda X: (1.0 + X.sum(axis=-1)) ** (-(d + 1))


def cub():
    for fam, d in [("oscillatory", 3), ("product_peak", 2), ("product_peak", 3), ("gaussian", 4), ("corner_peak", 3), ("gaussian", 5)]:
        g = genz(fam, d)
        cnt = [0]

        def f(X):
            cnt[0] += X.shape[0]
            return g(X)

        a, b = np.zeros(d), np.ones(d)
        run = lambda: integrate.cubature(f, a, b, rule="genz-malik", rtol=1e-6, atol=0, max_subdivisions=100000)
        cnt[0] = 0
        r = run()
        ev = cnt[0]
        ns = median_time(run)
        emit(f"cub:{fam}:{d}", f"{SCIPY} cubature genz-malik", "ns", ns, ev,
             f"{ev} evals, {r.subdivisions} subdivisions, estimate {float(r.estimate):.12g} ± {float(r.error):.1e}, {r.status}; vectorised Python + NumPy")


# ------------------------------------------------------------------------------------------------
# Gaussian rules: scipy.special.roots_* and numpy's *gauss.
# ------------------------------------------------------------------------------------------------
def gauss():
    from numpy.polynomial import hermite, laguerre, legendre

    for n in (5, 20, 100, 101, 1000, 10000):
        for kind, fn, npf in (
            ("legendre", lambda n: special.roots_legendre(n), legendre.leggauss),
            ("hermite", lambda n: special.roots_hermite(n), hermite.hermgauss),
            ("laguerre", lambda n: special.roots_laguerre(n), laguerre.laggauss),
            ("jacobi", lambda n: special.roots_jacobi(n, 0.5, 1.5), None),
        ):
            if kind == "jacobi" and n > 1000:
                continue
            total = {"legendre": 2.0, "hermite": math.sqrt(math.pi), "laguerre": 1.0,
                     "jacobi": 2.0**3 * math.exp(math.lgamma(1.5) + math.lgamma(2.5) - math.lgamma(4.0))}[kind]

            def quality(rule):
                x, w = rule
                bad = int(np.sum(~np.isfinite(x)) + np.sum(~np.isfinite(w)))
                if bad:
                    return f"{bad} non-finite nodes or weights (does less)"
                off = abs(math.fsum(w) - total) / total / np.finfo(float).eps
                return f"weights sum to the total within {off:.0f} ε" + (" (does less)" if off > 1000 else "")

            with np.errstate(all="ignore"):
                q = quality(fn(n))
                ns = median_time(lambda: fn(n), samples=SAMPLES if n < 10000 else 15)
            emit(f"gauss:{kind}:{n}", f"{SCIPY} special.roots_{kind}", "ns", ns, 0,
                 f"{q}; Golub–Welsch (eigvals_banded) + a Newton step; Hermite above 150 points by asymptotic expansions; compiled LAPACK under a Python driver")
            if npf is not None and n <= 1000:
                with np.errstate(all="ignore"):
                    q = quality(npf(n))
                    ns = median_time(lambda: npf(n))
                emit(f"gauss:{kind}:{n}", f"{NUMPY} polynomial.{npf.__name__}", "ns", ns, 0,
                     f"{q}; companion-matrix eigenvalues (O(n³)) + one Newton step; compiled LAPACK under Python")


# ------------------------------------------------------------------------------------------------
# Interpolation: CubicSpline, Akima1DInterpolator, PchipInterpolator; Chebyshev; barycentric; AAA.
# ------------------------------------------------------------------------------------------------
def knots(n):
    i = np.arange(n, dtype=float)
    x = i + 0.3 * np.sin(1.7 * i)
    y = np.sin(0.3 * x) + 0.2 * np.cos(1.1 * x)
    return x, y


def splines():
    rng = np.random.default_rng(1)
    for n in (100, 10000, 1000000):
        x, y = knots(n)
        builds = {
            "cubic": lambda: interpolate.CubicSpline(x, y, bc_type="natural"),
            "not_a_knot": lambda: interpolate.CubicSpline(x, y, bc_type="not-a-knot"),
            "clamped": lambda: interpolate.CubicSpline(x, y, bc_type=((1, 0.3), (1, 0.0))),
            "akima": lambda: interpolate.Akima1DInterpolator(x, y),
            "pchip": lambda: interpolate.PchipInterpolator(x, y),
        }
        cls = {"cubic": "CubicSpline natural", "not_a_knot": "CubicSpline not-a-knot", "clamped": "CubicSpline clamped",
               "akima": "Akima1DInterpolator", "pchip": "PchipInterpolator"}
        for k, fn in builds.items():
            ns = median_time(fn)
            emit(f"spline:{k}:{n}:build", f"{SCIPY} {cls[k]}", "ns", ns, 0, "construction; compiled solver under Python")
        m = 1000000
        t = rng.uniform(x[0], x[-1], m)
        s = builds["cubic"]()
        ns = median_time(lambda: s(t)) / m
        emit(f"spline:cubic:{n}:eval", f"{SCIPY} CubicSpline.__call__ (vectorised)", "ns", ns, 0, "10⁶ random points in one call; per point")
        s = builds["pchip"]()
        ns = median_time(lambda: s(t)) / m
        emit(f"spline:pchip:{n}:eval", f"{SCIPY} PchipInterpolator.__call__ (vectorised)", "ns", ns, 0, "10⁶ random points in one call; per point")


def cheb():
    from numpy.polynomial import chebyshev as C

    f = lambda x: np.exp(x) * np.cos(3 * x)
    a, b = -1.0, 2.0
    for deg in (10, 100, 1000):
        g = lambda t: f(a + (b - a) * (t + 1) / 2)
        ns = median_time(lambda: C.chebinterpolate(g, deg))
        emit(f"cheb:{deg}:build", f"{NUMPY} chebinterpolate", "ns", ns, 0, "first-kind points, vectorised f; Python + NumPy")
        if deg <= 100:
            c = C.chebinterpolate(g, deg)
            t = np.random.default_rng(2).uniform(a, b, 1000000)
            ns = median_time(lambda: C.chebval((2 * t - a - b) / (b - a), c)) / 1e6
            emit(f"cheb:{deg}:eval", f"{NUMPY} chebval (vectorised)", "ns", ns, 0, "10⁶ points in one call; per point")


def bary():
    for n in (100, 1000):
        k = np.arange(n)
        x = 0.5 - 0.5 * np.cos((2 * k + 1) * np.pi / (2 * n))
        y = np.cos(3 * x)
        ns = median_time(lambda: interpolate.BarycentricInterpolator(x, y))
        emit(f"bary:{n}:weights", f"{SCIPY} BarycentricInterpolator (weights)", "ns", ns, 0, "O(n²) weights in NumPy")
        bi = interpolate.BarycentricInterpolator(x, y)
        t = np.random.default_rng(3).uniform(0, 1, 100000)
        ns = median_time(lambda: bi(t)) / t.size
        emit(f"bary:{n}:eval", f"{SCIPY} BarycentricInterpolator.__call__ (vectorised)", "ns", ns, 0, "10⁵ points in one call; per point")


def aaa_cases():
    m = 1000
    lin = np.linspace(-1, 1, m)
    mid = (np.arange(m) + 0.5) / m * 2 - 1
    yield "exp", lin, np.exp(lin)
    yield "tan", mid, np.tan(np.pi * mid / 2)
    yield "abs", lin, np.abs(lin)
    yield "sqrt", np.linspace(0, 1, m), np.sqrt(np.linspace(0, 1, m))
    yield "runge", lin, 1 / (1 + 25 * lin**2)
    g = np.linspace(-1.5, 1.5, 100)
    yield "gamma", g, special.gamma(g)
    z = np.exp(2j * np.pi * np.arange(m) / m)
    yield "circle", z, np.log(2 + z**4) / (1 - 16 * z**4)


def aaa():
    for name, x, y in aaa_cases():
        ns = median_time(lambda: interpolate.AAA(x, y))
        r = interpolate.AAA(x, y)
        emit(f"aaa:{name}:build", f"{SCIPY} interpolate.AAA", "ns", ns, 0,
             f"{len(r.support_points)} support points; clean-up on (default); Python loop + LAPACK SVD")
        t = np.random.default_rng(4).uniform(-0.99, 0.99, 100000) if name != "circle" else np.exp(2j * np.pi * np.random.default_rng(4).uniform(0, 1, 100000))
        if name == "sqrt":
            t = (t + 1) / 2
        ns = median_time(lambda: r(t)) / t.size
        emit(f"aaa:{name}:eval", f"{SCIPY} AAA.__call__ (vectorised)", "ns", ns, 0, "10⁵ points in one call; per point")
        if name in ("abs", "tan", "gamma"):
            # AAA caches its poles, residues and roots: clear the caches before every call so
            # that each call computes them (residues include the poles they need).
            def fresh(fn, attrs):
                def run():
                    for a in attrs:
                        setattr(r, a, None)
                    return fn()
                return run
            for what, fn, attrs in (("poles", r.poles, ("_poles",)), ("residues", r.residues, ("_poles", "_residues")),
                                    ("zeros", r.roots, ("_roots",))):
                ns = median_time(fresh(fn, attrs))
                emit(f"aaa:{name}:{what}", f"{SCIPY} AAA.{fn.__name__}", "ns", ns, 0,
                     "generalised eigenvalues of the arrowhead pencil (LAPACK eigvals), caches cleared per call")


GROUPS = [("lp", lp), ("lbfgsb", lbfgsb), ("ode", ode), ("fits", fits), ("cub", cub), ("gauss", gauss),
          ("splines", splines), ("cheb", cheb), ("bary", bary), ("aaa", aaa)]
for name, fn in GROUPS:
    if want(name):
        try:
            fn()
        except Exception as e:  # a SciPy without the feature (an older distribution's) skips it
            print(f"refs.py: group {name} skipped: {type(e).__name__}: {e}", file=sys.stderr)
