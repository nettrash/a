#!/usr/bin/env python3
"""Reference results for lib/tests/math/numerics/opt.rs.

Prints the Rust tables the test file embeds:

* ONE_DIM: scipy.optimize.minimize_scalar(method="bounded"), which is Brent's fminbound, on
  the one-variable problems of the test, with xatol = 3*sqrt(eps) so that its tolerance,
  sqrt(eps)*|x| + xatol/3, is the tolerance minimize uses near |x| = 1: the minimiser and the
  number of evaluations.
* BATTERY: scipy.optimize.minimize with method BFGS and CG at gtol = sqrt(eps) (the default of
  OptOptions), and Nelder-Mead from the same starting simplex as nelder_mead (x0 and x0 + step
  along each axis) with xatol = fatol = 4*sqrt(eps): whether it succeeded, its iterations, and
  the point.
* NIST: five problems of the NIST StRD nonlinear regression suite (Misra1a, BoxBOD, Thurber,
  MGH09, MGH17), their data, both starts and the certified values, as published by NIST (public
  domain), and what scipy.optimize.least_squares(method="lm"), which is MINPACK's lmder, does
  from each start with its default tolerances (1e-8) and 20,000 evaluations allowed: success,
  evaluations, and the log relative error of its parameters against the certified ones.
* DECAY: the least squares fit of a*exp(-lambda*t) to 2*exp(-1.5t) + 0.05 at t = 0, 0.5, ..., 2,
  by mpmath at 50 digits (Newton on the normal equations from the scipy answer).
* B1: NIST MGH10, and what MINPACK and TRF report from the starts of the first review at which
  the fits' old step test stopped (see b1()).
* POLE_MINIMUM: the local minimum of a*t/(b + t) + c beside the pole of the second review's
  starts, and what MINPACK and TRF report from those starts (see pole()).
* The polynomial fits: how well the rounded samples determine the coefficients, and where
  LAPACK's solvers land (see polynomial()).

    python3 gen_opt.py > out.txt

Written against SciPy 1.18, NumPy 2.3 and mpmath 1.3.
"""
import math
import warnings

import mpmath
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

warnings.filterwarnings("ignore")
EPS = np.finfo(float).eps
SQRT_EPS = math.sqrt(EPS)


def one_dim():
    return [
        ("basin", lambda x: (x - 3.0) ** 2 + 1.0, 0.0, 10.0),
        ("double_well", lambda x: x ** 4 - 2.0 * x * x, 0.0, 2.0),
        ("quartic_well", lambda x: (x * x - 2.0) ** 2, 1.0, 2.0),
        ("exp_line", lambda x: math.exp(x) - 2.0 * x, 0.0, 2.0),
        ("kink", lambda x: abs(x - 2.0) + 1e-3 * (x - 2.0) ** 2, 0.0, 4.0),
        ("x_log_x", lambda x: x * math.log(x), 0.01, 1.0),
        ("sine", math.sin, 2.0, 6.0),
        ("cosh", lambda x: math.cosh(x - 1.0), -10.0, 10.0),
        ("far", lambda x: (x - 1e3) ** 2, 0.0, 1e4),
        ("lorentz", lambda x: -1.0 / (1.0 + (x - 0.3) ** 2), -5.0, 5.0),
        ("sixth", lambda x: x ** 6, -1.0, 2.0),
    ]


def col(v):
    return np.asarray(v, dtype=float)


def rosen(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosen_grad(x):
    g = np.zeros_like(x)
    t = x[1:] - x[:-1] ** 2
    g[:-1] += -400.0 * x[:-1] * t - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * t
    return g


def powell(x):
    a, b, c, d = x
    return (a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4


def powell_grad(x):
    a, b, c, d = x
    return np.array([2 * (a + 10 * b) + 40 * (a - d) ** 3, 20 * (a + 10 * b) + 4 * (b - 2 * c) ** 3,
                     10 * (c - d) - 8 * (b - 2 * c) ** 3, -10 * (c - d) - 40 * (a - d) ** 3])


def beale(x):
    a, b = x
    return (1.5 - a + a * b) ** 2 + (2.25 - a + a * b * b) ** 2 + (2.625 - a + a * b ** 3) ** 2


def beale_grad(x):
    a, b = x
    t1, t2, t3 = 1.5 - a + a * b, 2.25 - a + a * b * b, 2.625 - a + a * b ** 3
    return np.array([2 * t1 * (b - 1) + 2 * t2 * (b * b - 1) + 2 * t3 * (b ** 3 - 1),
                     2 * t1 * a + 4 * t2 * a * b + 6 * t3 * a * b * b])


def wood(x):
    a, b, c, d = x
    return (100 * (a * a - b) ** 2 + (a - 1) ** 2 + (c - 1) ** 2 + 90 * (c * c - d) ** 2
            + 10.1 * ((b - 1) ** 2 + (d - 1) ** 2) + 19.8 * (b - 1) * (d - 1))


def wood_grad(x):
    a, b, c, d = x
    return np.array([400 * a * (a * a - b) + 2 * (a - 1), -200 * (a * a - b) + 20.2 * (b - 1) + 19.8 * (d - 1),
                     2 * (c - 1) + 360 * c * (c * c - d), -180 * (c * c - d) + 20.2 * (d - 1) + 19.8 * (b - 1)])


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_grad(x):
    t1, t2 = x[0] ** 2 + x[1] - 11, x[0] + x[1] ** 2 - 7
    return np.array([4 * x[0] * t1 + 2 * t2, 2 * t1 + 4 * x[1] * t2])


def brown(x):
    return (x[0] - 1e6) ** 2 + (x[1] - 2e-6) ** 2 + (x[0] * x[1] - 2) ** 2


def brown_grad(x):
    t = x[0] * x[1] - 2
    return np.array([2 * (x[0] - 1e6) + 2 * t * x[1], 2 * (x[1] - 2e-6) + 2 * t * x[0]])


def quadratic(n, cond):
    """0.5 (x - c)' Q D Q (x - c), Q = I - (2/n) 1 1' (a reflection), D_i = cond^(i/(n-1))."""
    d = np.array([cond ** (i / (n - 1)) for i in range(n)])
    c = np.array([math.sin(i + 1.0) for i in range(n)])

    def refl(v):
        return v - 2.0 * np.sum(v) / n

    return (lambda x: 0.5 * float(np.sum(d * refl(x - c) ** 2)),
            lambda x: refl(d * refl(x - c)), c)


def battery():
    out = []
    for n in (2, 10, 50):
        out.append((f"rosen{n}", rosen, rosen_grad, [-1.2 if i % 2 == 0 else 1.0 for i in range(n)], 1.0))
    out.append(("powell", powell, powell_grad, [3.0, -1.0, 0.0, 1.0], 1.0))
    out.append(("beale", beale, beale_grad, [1.0, 1.0], 1.0))
    out.append(("wood", wood, wood_grad, [-3.0, -1.0, -3.0, -1.0], 1.0))
    for s in ([0.0, 0.0], [-1.0, -1.0], [-3.0, 3.0], [4.0, -2.0]):
        out.append((f"himmelblau({s[0]:g},{s[1]:g})", himmelblau, himmelblau_grad, s, 1.0))
    out.append(("brown", brown, brown_grad, [1.0, 1.0], 1.0))
    for cond in (1e2, 1e4, 1e6):
        f, g, _ = quadratic(10, cond)
        out.append((f"quad10_{cond:.0e}".replace("+0", ""), f, g, [0.0] * 10, 1.0))
    return out


def rust_list(v):
    return "&[" + ", ".join(repr(float(x)) for x in v) + "]"


def main():
    print("// Made by data/gen_opt.py: minimize_scalar(method=\"bounded\") (Brent's fminbound):")
    print("// name, x, evaluations.")
    print("const ONE_DIM: &[(&str, f64, usize)] = &[")
    for name, f, a, b in one_dim():
        r = minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(xatol=3 * SQRT_EPS, maxiter=5000))
        print(f'    ("{name}", {float(r.x)!r}, {r.nfev}),')
    print("];")

    print("// Made by data/gen_opt.py: scipy.optimize.minimize from the same starts and tolerances:")
    print("// problem, method, success, iterations, x.")
    print("const BATTERY: &[(&str, &str, bool, usize, &[f64])] = &[")
    for name, f, g, x0, step in battery():
        x0 = col(x0)
        for method in ("BFGS", "CG"):
            r = minimize(f, x0, jac=g, method=method, options=dict(gtol=SQRT_EPS, norm=np.inf, maxiter=10000))
            print(f'    ("{name}", "{method}", {str(bool(r.success)).lower()}, {r.nit}, {rust_list(r.x)}),')
        if len(x0) <= 4:
            simplex = np.array([x0] + [x0 + step * np.eye(len(x0))[i] for i in range(len(x0))])
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=simplex, xatol=4 * SQRT_EPS, fatol=4 * SQRT_EPS,
                                      maxiter=100000, maxfev=100000))
            print(f'    ("{name}", "Nelder-Mead", {str(bool(r.success)).lower()}, {r.nit}, {rust_list(r.x)}),')
    print("];")

    models = {
        "Misra1a": (lambda b, x: b[0] * (1 - np.exp(-b[1] * x)),
                    lambda b, x: np.column_stack([1 - np.exp(-b[1] * x), b[0] * x * np.exp(-b[1] * x)])),
        "BoxBOD": (lambda b, x: b[0] * (1 - np.exp(-b[1] * x)),
                   lambda b, x: np.column_stack([1 - np.exp(-b[1] * x), b[0] * x * np.exp(-b[1] * x)])),
        "Thurber": (lambda b, x: (b[0] + b[1] * x + b[2] * x ** 2 + b[3] * x ** 3) / (1 + b[4] * x + b[5] * x ** 2 + b[6] * x ** 3),
                    None),
        "MGH09": (lambda b, x: b[0] * (x ** 2 + x * b[1]) / (x ** 2 + x * b[2] + b[3]), None),
        "MGH17": (lambda b, x: b[0] + b[1] * np.exp(-x * b[3]) + b[2] * np.exp(-x * b[4]), None),
    }

    def jacobian(name, b, x):
        if name in ("Misra1a", "BoxBOD"):
            return models[name][1](b, x)
        if name == "Thurber":
            num = b[0] + b[1] * x + b[2] * x ** 2 + b[3] * x ** 3
            den = 1 + b[4] * x + b[5] * x ** 2 + b[6] * x ** 3
            return np.column_stack([1 / den, x / den, x ** 2 / den, x ** 3 / den,
                                    -num * x / den ** 2, -num * x ** 2 / den ** 2, -num * x ** 3 / den ** 2])
        if name == "MGH09":
            num, den = x ** 2 + x * b[1], x ** 2 + x * b[2] + b[3]
            return np.column_stack([num / den, b[0] * x / den, -b[0] * num * x / den ** 2, -b[0] * num / den ** 2])
        e4, e5 = np.exp(-x * b[3]), np.exp(-x * b[4])
        return np.column_stack([np.ones_like(x), e4, e5, -b[1] * x * e4, -b[2] * x * e5])

    print("// Made by data/gen_opt.py: the NIST StRD problems (data, starts and certified values as")
    print("// NIST publishes them).")
    for name, p in NIST.items():
        up = name.upper()
        print(f"const {up}_X: &[f64] = {rust_list(p['x'])};")
        print(f"const {up}_Y: &[f64] = {rust_list(p['y'])};")
        print(f"const {up}_STARTS: [&[f64]; 2] = [{rust_list(p['start1'])}, {rust_list(p['start2'])}];")
        print(f"const {up}_CERTIFIED: &[f64] = {rust_list(p['certified'])};")
        print(f"const {up}_RSS: f64 = {float(p['rss'])!r};")
    print("// Made by data/gen_opt.py: least_squares(method=\"lm\") (MINPACK) from each start:")
    print("// problem, start, success, evaluations, digits of the parameters that agree with the")
    print("// certified values.")
    print("const NIST_SCIPY: &[(&str, usize, bool, usize, f64)] = &[")
    for name, p in NIST.items():
        x, y = col(p["x"]), col(p["y"])
        f = models[name][0]
        for k, s in enumerate((p["start1"], p["start2"]), 1):
            r = least_squares(lambda b: f(b, x) - y, col(s), jac=lambda b: jacobian(name, b, x), method="lm",
                              max_nfev=20000)
            c = col(p["certified"])
            rel = np.abs((r.x - c) / c)
            lre = float(np.clip(np.min(-np.log10(np.where(rel == 0, 1e-16, rel))), 0, 16))
            print(f'    ("{name}", {k}, {str(bool(r.success)).lower()}, {r.nfev}, {lre:.2f}),')
    print("];")

    # The decay fit, to 50 digits: Newton's method on the gradient of half the sum of squares.
    mpmath.mp.dps = 50
    t = [mpmath.mpf(k) / 2 for k in range(5)]
    yv = [2 * mpmath.exp(-mpmath.mpf(3) / 2 * tk) + mpmath.mpf(1) / 20 for tk in t]

    def grad(a, lam):
        r = [a * mpmath.exp(-lam * tk) - yk for tk, yk in zip(t, yv)]
        ja = [mpmath.exp(-lam * tk) for tk in t]
        jl = [-a * tk * mpmath.exp(-lam * tk) for tk in t]
        return [sum(p * q for p, q in zip(ja, r)), sum(p * q for p, q in zip(jl, r))]

    a, lam = mpmath.findroot(lambda a, lam: grad(a, lam), (mpmath.mpf("2.04"), mpmath.mpf("1.40")))
    print("// Made by data/gen_opt.py: the least squares fit of a·exp(−λt) to 2·exp(−1.5t) + 0.05,")
    print("// by mpmath at 50 digits.")
    print(f"const DECAY: [f64; 2] = [{float(a)!r}, {float(lam)!r}];")
    b1()


def b1():
    """The starts of the first review (B1) from which the step test, measured in one norm over
    all the parameters, took a step that changed one parameter wholly for convergence: NIST MGH10
    (data and certified values as NIST publishes them) from its certified values perturbed by up
    to 90%, a*exp(b*t) through exact points of a = 2, b = 0.5 from (2, 20), and a*exp(-b*t) + c
    through exact points of 2*exp(-0.5t) + 0.3 from two starts with the rate of the wrong sign.
    Printed: what MINPACK (least_squares method="lm") and SciPy's TRF report from them."""
    p = MGH10
    up = "MGH10"
    print("// Made by data/gen_opt.py: NIST StRD MGH10 (data and certified values as NIST publishes")
    print("// them).")
    print(f"const {up}_X: &[f64] = {rust_list(p['x'])};")
    print(f"const {up}_Y: &[f64] = {rust_list(p['y'])};")
    print(f"const {up}_CERTIFIED: &[f64] = {rust_list(p['certified'])};")
    print(f"const {up}_RSS: f64 = {float(p['rss'])!r};")
    x, y = col(p["x"]), col(p["y"])
    f = lambda b: b[0] * np.exp(b[1] / (x + b[2])) - y

    def jac(b):
        d = x + b[2]
        e = np.exp(b[1] / d)
        return np.column_stack([e, b[0] * e / d, -b[0] * b[1] * e / d ** 2])

    t1 = np.arange(1.0, 6.0)
    y1 = 2.0 * np.exp(0.5 * t1)
    t2 = 8.0 * np.arange(25) / 24.0
    y2 = 2.0 * np.exp(-0.5 * t2) + 0.3
    cases = [
        ("MGH10", f, jac, [0.005610260537259397, 10820.234225449876, 36.87926726109717], p["rss"] / 2),
        ("a*exp(b*t)", lambda q: q[0] * np.exp(q[1] * t1) - y1,
         lambda q: np.column_stack([np.exp(q[1] * t1), q[0] * t1 * np.exp(q[1] * t1)]), [2.0, 20.0], 0.0),
    ]
    for s in ([20.0, -17.0, -0.3], [1.0, -10.0, 1.0]):
        cases.append(("a*exp(-b*t) + c", lambda q: q[0] * np.exp(-q[1] * t2) + q[2] - y2,
                      lambda q: np.column_stack([np.exp(-q[1] * t2), -q[0] * t2 * np.exp(-q[1] * t2),
                                                 np.ones_like(t2)]), s, 0.0))
    print("// B1: what MINPACK (lm) and TRF report from the starts of the first review, and the cost")
    print("// they report it at (the least cost is the last number).")
    for name, fun, jacobian, start, least in cases:
        for method in ("lm", "trf"):
            r = least_squares(fun, col(start), jac=jacobian, method=method)
            print(f"//   {name} from {start}, {method}: success {r.success}, status {r.status}, cost {r.cost:.3e}"
                  f" (least {least:.3e})")
    pole()


def pole():
    """The second review's starts beside a pole of the model (P1): y = a*t/(b + t) + c through 30
    exact points of (3, 1.2, 0.4) on [0, 8], from (3, b0, 0.4) with b0 = -t21*(1 - 1e-9) and
    b0 = -t21 + t21*eps, one unit in the last place from the pole at t21 = 8*21/29. Printed: the
    local minimum on that side of the pole (b < -t21), by MINPACK from a point well off it with
    every tolerance at 1e-15, and what MINPACK and TRF report from the two starts."""
    t = 8.0 * np.arange(30) / 29.0
    y = 3.0 * t / (1.2 + t) + 0.4
    fun = lambda p: p[0] * t / (p[1] + t) + p[2] - y
    jac = lambda p: np.column_stack([t / (p[1] + t), -p[0] * t / (p[1] + t) ** 2, np.ones_like(t)])
    r = least_squares(fun, col([0.0105, -5.7, 2.43]), jac=jac, method="lm", xtol=1e-15, ftol=1e-15,
                      gtol=1e-15)
    print("// Made by data/gen_opt.py: the local minimum of a·t/(b + t) + c through 30 exact points of")
    print("// (3, 1.2, 0.4) on [0, 8] on the side b < −t₂₁ of the pole at t₂₁ = 8·21/29 (MINPACK, every")
    print("// tolerance 10⁻¹⁵): the cost and the point.")
    point = ", ".join(repr(float(v)) for v in r.x)
    print(f"const POLE_MINIMUM: (f64, [f64; 3]) = ({float(r.cost)!r}, [{point}]);")
    t21 = t[21]
    print("// What MINPACK (lm) and TRF report from the starts beside the pole:")
    for name, b0 in (("1e-9", -t21 * (1.0 - 1e-9)), ("1 ulp", -t21 + t21 * EPS)):
        for method in ("lm", "trf"):
            r = least_squares(fun, col([3.0, b0, 0.4]), jac=jac, method=method)
            print(f"//   b0 {name} off the pole, {method}: success {r.success}, status {r.status},"
                  f" cost {r.cost:.3e}")
    polynomial()


def powi(x, k):
    """x**k as Rust's f64::powi computes it: compiler-rt's __powidf2, by repeated squaring."""
    result, square = 1.0, x
    while True:
        if k & 1:
            result *= square
        k >>= 1
        if k == 0:
            return result
        square *= square


def left_sum(terms):
    """The sum of the terms from the left, as Rust's Iterator::sum of floats makes it (Python's
    sum() of floats is compensated since 3.12, and rounds differently)."""
    total = -0.0
    for term in terms:
        total += term
    return total


def polynomial():
    """How well the samples of the polynomial fits determine their coefficients: 1, -2, 3, ...
    of degree 8 and 12 at t = i/39, i = 0, ..., 39, built bit for bit as the test file builds them.
    Printed: the condition of the Jacobian and its least singular value, then the largest error,
    relative to the largest coefficient as the test measures it, of the exact least squares
    solution of the samples as they are rounded to floats (mpmath at 60 digits) and of LAPACK's
    solvers (gelsd, gelsy, and dgeqrf by numpy.linalg.qr): what a fit can come to, and how far a
    backward stable solver lands from it by its own rounding."""
    from scipy.linalg import lstsq, solve_triangular
    mpmath.mp.dps = 60
    t = [i / 39.0 for i in range(40)]
    print("// The polynomial fits (polynomial()): the condition of the Jacobian, its least singular")
    print("// value, and the error relative to the largest coefficient of the exact least squares")
    print("// solution of the rounded samples and of LAPACK's solvers.")
    for n in (9, 13):
        truth = [(k + 1) if k % 2 == 0 else -(k + 1) for k in range(n)]
        jacobian = np.array([[powi(ti, k) for k in range(n)] for ti in t])
        y = np.array([left_sum(c * powi(ti, k) for k, c in enumerate(truth)) for ti in t])
        sigma = np.linalg.svd(jacobian, compute_uv=False)

        def error(x):
            return max(abs(float(a) - b) for a, b in zip(x, truth)) / n

        j = mpmath.matrix([[mpmath.mpf(float(v)) for v in row] for row in jacobian])
        yy = mpmath.matrix([mpmath.mpf(float(v)) for v in y])
        exact = mpmath.lu_solve(j.T * j, j.T * yy)
        q, r = np.linalg.qr(jacobian)
        solvers = [("gelsd", lstsq(jacobian, y, lapack_driver="gelsd")[0]),
                   ("gelsy", lstsq(jacobian, y, lapack_driver="gelsy")[0]),
                   ("dgeqrf", solve_triangular(r, q.T @ y))]
        found = ", ".join(f"{name} {error(x):.2e}" for name, x in solvers)
        print(f"//   degree {n - 1}: condition {sigma[0] / sigma[-1]:.2e}, least singular value"
              f" {sigma[-1]:.2e}; exact {error(exact):.2e}; {found}")


MGH10 = dict(
    certified=[5.6096364710E-03, 6.1813463463E+03, 3.4522363462E+02],
    rss=8.7945855171E+01,
    y=[3.478000E+04, 2.861000E+04, 2.365000E+04, 1.963000E+04, 1.637000E+04, 1.372000E+04, 1.154000E+04, 9.744000E+03, 8.261000E+03, 7.030000E+03, 6.005000E+03, 5.147000E+03, 4.427000E+03, 3.820000E+03, 3.307000E+03, 2.872000E+03],
    x=[5.000000E+01, 5.500000E+01, 6.000000E+01, 6.500000E+01, 7.000000E+01, 7.500000E+01, 8.000000E+01, 8.500000E+01, 9.000000E+01, 9.500000E+01, 1.000000E+02, 1.050000E+02, 1.100000E+02, 1.150000E+02, 1.200000E+02, 1.250000E+02],
)

NIST = {
    'Misra1a': dict(
        start1=[500, 0.0001],
        start2=[250, 0.0005],
        certified=[2.3894212918E+02, 5.5015643181E-04],
        rss=1.2455138894E-01,
        y=[10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0],
        x=[77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0],
    ),
    'BoxBOD': dict(
        start1=[1, 1],
        start2=[100, 0.75],
        certified=[2.1380940889E+02, 5.4723748542E-01],
        rss=1.1680088766E+03,
        y=[109, 149, 149, 191, 213, 224],
        x=[1, 2, 3, 5, 7, 10],
    ),
    'Thurber': dict(
        start1=[1000, 1000, 400, 40, 0.7, 0.3, 0.03],
        start2=[1300, 1500, 500, 75, 1, 0.4, 0.05],
        certified=[1.2881396800E+03, 1.4910792535E+03, 5.8323836877E+02, 7.5416644291E+01, 9.6629502864E-01, 3.9797285797E-01, 4.9727297349E-02],
        rss=5.6427082397E+03,
        y=[80.574E0, 84.248E0, 87.264E0, 87.195E0, 89.076E0, 89.608E0, 89.868E0, 90.101E0, 92.405E0, 95.854E0, 100.696E0, 101.060E0, 401.672E0, 390.724E0, 567.534E0, 635.316E0, 733.054E0, 759.087E0, 894.206E0, 990.785E0, 1090.109E0, 1080.914E0, 1122.643E0, 1178.351E0, 1260.531E0, 1273.514E0, 1288.339E0, 1327.543E0, 1353.863E0, 1414.509E0, 1425.208E0, 1421.384E0, 1442.962E0, 1464.350E0, 1468.705E0, 1447.894E0, 1457.628E0],
        x=[-3.067E0, -2.981E0, -2.921E0, -2.912E0, -2.840E0, -2.797E0, -2.702E0, -2.699E0, -2.633E0, -2.481E0, -2.363E0, -2.322E0, -1.501E0, -1.460E0, -1.274E0, -1.212E0, -1.100E0, -1.046E0, -0.915E0, -0.714E0, -0.566E0, -0.545E0, -0.400E0, -0.309E0, -0.109E0, -0.103E0, 0.010E0, 0.119E0, 0.377E0, 0.790E0, 0.963E0, 1.006E0, 1.115E0, 1.572E0, 1.841E0, 2.047E0, 2.200E0],
    ),
    'MGH09': dict(
        start1=[25, 39, 41.5, 39],
        start2=[0.25, 0.39, 0.415, 0.39],
        certified=[1.9280693458E-01, 1.9128232873E-01, 1.2305650693E-01, 1.3606233068E-01],
        rss=3.0750560385E-04,
        y=[1.957000E-01, 1.947000E-01, 1.735000E-01, 1.600000E-01, 8.440000E-02, 6.270000E-02, 4.560000E-02, 3.420000E-02, 3.230000E-02, 2.350000E-02, 2.460000E-02],
        x=[4.000000E+00, 2.000000E+00, 1.000000E+00, 5.000000E-01, 2.500000E-01, 1.670000E-01, 1.250000E-01, 1.000000E-01, 8.330000E-02, 7.140000E-02, 6.250000E-02],
    ),
    'MGH17': dict(
        start1=[50, 150, -100, 1, 2],
        start2=[0.5, 1.5, -1, 0.01, 0.02],
        certified=[3.7541005211E-01, 1.9358469127E+00, -1.4646871366E+00, 1.2867534640E-02, 2.2122699662E-02],
        rss=5.4648946975E-05,
        y=[8.440000E-01, 9.080000E-01, 9.320000E-01, 9.360000E-01, 9.250000E-01, 9.080000E-01, 8.810000E-01, 8.500000E-01, 8.180000E-01, 7.840000E-01, 7.510000E-01, 7.180000E-01, 6.850000E-01, 6.580000E-01, 6.280000E-01, 6.030000E-01, 5.800000E-01, 5.580000E-01, 5.380000E-01, 5.220000E-01, 5.060000E-01, 4.900000E-01, 4.780000E-01, 4.670000E-01, 4.570000E-01, 4.480000E-01, 4.380000E-01, 4.310000E-01, 4.240000E-01, 4.200000E-01, 4.140000E-01, 4.110000E-01, 4.060000E-01],
        x=[0.000000E+00, 1.000000E+01, 2.000000E+01, 3.000000E+01, 4.000000E+01, 5.000000E+01, 6.000000E+01, 7.000000E+01, 8.000000E+01, 9.000000E+01, 1.000000E+02, 1.100000E+02, 1.200000E+02, 1.300000E+02, 1.400000E+02, 1.500000E+02, 1.600000E+02, 1.700000E+02, 1.800000E+02, 1.900000E+02, 2.000000E+02, 2.100000E+02, 2.200000E+02, 2.300000E+02, 2.400000E+02, 2.500000E+02, 2.600000E+02, 2.700000E+02, 2.800000E+02, 2.900000E+02, 3.000000E+02, 3.100000E+02, 3.200000E+02],
    ),
}


if __name__ == "__main__":
    main()
