"""Reference data for lib/tests/math/numerics/system.rs: the More-Garbow-Hillstrom systems of
nonlinear equations (ACM TOMS 7, 1981) solved by SciPy 1.18's MINPACK hybrj (`root(method="hybr")`
with the analytic Jacobian) and by its `broyden1`, from the standard starting points and from ten
times them, and their roots to 40 digits by mpmath's Newton iteration.

Run with python3 (NumPy, SciPy and mpmath installed): python3 gen_systems.py > systems_table.txt
and paste the two tables into system.rs. The functions are written as in system.rs.

ROOTS: for each problem, the root refined in 40-digit arithmetic from the double-precision root of
a damped Newton iteration from the standard start (for the trigonometric system, from the root the
crate's methods reach; Rosenbrock's, Powell's singular and the helical valley's are exact), then
rounded to f64 and printed as bit patterns. SCIPY: for each problem and start, whether hybr and broyden1
reported success, hybr's evaluations, and the largest difference of hybr's answer from ROOTS
relative to the largest entry of the root (NaN where it failed or found another root).
"""
import math
import struct
import warnings

import mpmath as mp
import numpy as np
from scipy.optimize import root

np.seterr(all="ignore")
warnings.simplefilter("ignore")
mp.mp.dps = 40
TPI = 2.0 * math.pi


def theta(x1, x2, m=math):
    tpi = 2 * m.pi
    if x1 > 0:
        return m.atan(x2 / x1) / tpi
    if x1 < 0:
        return m.atan(x2 / x1) / tpi + 0.5
    return 0.25 if x2 >= 0 else -0.25


def problems():
    """(name, n, f(x, m), j(x, m), x0) with m the math module of the arithmetic (math or mpmath)."""
    v = []
    v.append(("rosenbrock", 2,
              lambda x, m: [10 * (x[1] - x[0] ** 2), 1 - x[0]],
              lambda x, m: [[-20 * x[0], 10], [-1, 0]],
              [-1.2, 1.0]))

    def ps_f(x, m):
        s5, s10 = m.sqrt(5), m.sqrt(10)
        return [x[0] + 10 * x[1], s5 * (x[2] - x[3]), (x[1] - 2 * x[2]) ** 2, s10 * (x[0] - x[3]) ** 2]

    def ps_j(x, m):
        s5, s10 = m.sqrt(5), m.sqrt(10)
        a, b = x[1] - 2 * x[2], x[0] - x[3]
        return [[1, 10, 0, 0], [0, 0, s5, -s5], [0, 2 * a, -4 * a, 0], [2 * s10 * b, 0, 0, -2 * s10 * b]]

    v.append(("powell_singular", 4, ps_f, ps_j, [3.0, -1.0, 0.0, 1.0]))
    v.append(("powell_badly_scaled", 2,
              lambda x, m: [10000 * x[0] * x[1] - 1, m.exp(-x[0]) + m.exp(-x[1]) - m.mpf("1.0001") if m is mp else m.exp(-x[0]) + m.exp(-x[1]) - 1.0001],
              lambda x, m: [[10000 * x[1], 10000 * x[0]], [-m.exp(-x[0]), -m.exp(-x[1])]],
              [0.0, 1.0]))

    def hv_f(x, m):
        r = m.sqrt(x[0] ** 2 + x[1] ** 2)
        return [10 * (x[2] - 10 * theta(x[0], x[1], m)), 10 * (r - 1), x[2]]

    def hv_j(x, m):
        r2 = x[0] ** 2 + x[1] ** 2
        r = m.sqrt(r2)
        tpi = 2 * m.pi
        return [[100 * x[1] / (tpi * r2), -100 * x[0] / (tpi * r2), 10], [10 * x[0] / r, 10 * x[1] / r, 0], [0, 0, 1]]

    v.append(("helical_valley", 3, hv_f, hv_j, [-1.0, 0.0, 0.0]))

    def trig(n):
        def f(x, m):
            sc = sum(m.cos(t) for t in x)
            return [n - sc + (i + 1) * (1 - m.cos(x[i])) - m.sin(x[i]) for i in range(n)]

        def j(x, m):
            rows = []
            for i in range(n):
                row = [m.sin(x[c]) for c in range(n)]
                row[i] += (i + 1) * m.sin(x[i]) - m.cos(x[i])
                rows.append(row)
            return rows
        return (f"trigonometric_{n}", n, f, j, [1.0 / n] * n)

    v.append(trig(10))

    def dbv(n):
        def f(x, m):
            h = m.mpf(1) / (n + 1) if m is mp else 1.0 / (n + 1)
            out = []
            for k in range(n):
                t = (k + 1) * h
                xm = x[k - 1] if k > 0 else 0
                xp = x[k + 1] if k + 1 < n else 0
                out.append(2 * x[k] - xm - xp + h * h * (x[k] + t + 1) ** 3 / 2)
            return out

        def j(x, m):
            h = m.mpf(1) / (n + 1) if m is mp else 1.0 / (n + 1)
            rows = [[0] * n for _ in range(n)]
            for k in range(n):
                t = (k + 1) * h
                rows[k][k] = 2 + 3 * h * h * (x[k] + t + 1) ** 2 / 2
                if k > 0:
                    rows[k][k - 1] = -1
                if k + 1 < n:
                    rows[k][k + 1] = -1
            return rows
        h = 1.0 / (n + 1)
        return (f"discrete_boundary_{n}", n, f, j, [(k + 1) * h * ((k + 1) * h - 1) for k in range(n)])

    for n in [10, 50, 100]:
        v.append(dbv(n))

    def tri(n):
        def f(x, m):
            return [(3 - 2 * x[k]) * x[k] - (x[k - 1] if k > 0 else 0) - 2 * (x[k + 1] if k + 1 < n else 0) + 1 for k in range(n)]

        def j(x, m):
            rows = [[0] * n for _ in range(n)]
            for k in range(n):
                rows[k][k] = 3 - 4 * x[k]
                if k > 0:
                    rows[k][k - 1] = -1
                if k + 1 < n:
                    rows[k][k + 1] = -2
            return rows
        return (f"broyden_tridiagonal_{n}", n, f, j, [-1.0] * n)

    def band(n):
        def f(x, m):
            out = []
            for k in range(n):
                s = 0
                for i in range(max(0, k - 5), min(n - 1, k + 1) + 1):
                    if i != k:
                        s += x[i] * (1 + x[i])
                out.append(x[k] * (2 + 5 * x[k] ** 2) + 1 - s)
            return out

        def j(x, m):
            rows = [[0] * n for _ in range(n)]
            for k in range(n):
                for i in range(max(0, k - 5), min(n - 1, k + 1) + 1):
                    if i != k:
                        rows[k][i] = -(1 + 2 * x[i])
                rows[k][k] = 2 + 15 * x[k] ** 2
            return rows
        return (f"broyden_banded_{n}", n, f, j, [-1.0] * n)

    for n in [10, 50, 100]:
        v.append(tri(n))
    for n in [10, 50, 100]:
        v.append(band(n))
    return v


def damped_newton(f, j, x, iters=200):
    """Newton with backtracking on 1/2|F|^2, in double precision: the root a line-search Newton
    reaches from x (the trigonometric system has several)."""
    x = np.array(x, dtype=float)
    for _ in range(iters):
        F = np.array(f(x, math), dtype=float)
        if np.max(np.abs(F)) < 1e-15:
            break
        p = np.linalg.solve(np.array(j(x, math), dtype=float), -F)
        lam, phi = 1.0, 0.5 * F @ F
        while lam > 1e-12:
            Ft = np.array(f(x + lam * p, math), dtype=float)
            if 0.5 * Ft @ Ft <= phi * (1 - 2e-4 * lam):
                break
            lam *= 0.5
        x = x + lam * p
    return x


def refine(f, j, x):
    """Newton's iteration in 40 digits from x."""
    xm = mp.matrix([mp.mpf(float(t)) for t in x])
    for _ in range(8):
        F = mp.matrix(f(list(xm), mp))
        J = mp.matrix(j(list(xm), mp))
        d = mp.lu_solve(J, -F)
        xm = xm + d
        if mp.norm(d, mp.inf) < mp.mpf(10) ** -35 * (1 + mp.norm(xm, mp.inf)):
            break
    return [float(t) for t in xm]


# The roots known exactly.
EXACT = {"rosenbrock": [1.0, 1.0], "powell_singular": [0.0] * 4, "helical_valley": [1.0, 0.0, 0.0]}

# The trigonometric system has several roots, and which one an iteration reaches depends on the
# details of its line search: this is the one newton_system and broyden of the crate reach from the
# standard start (both of them), to the digits they print, refined here to 40 digits. The test
# checks the accuracy of that root, and checks itself that both methods reach it.
START = {"trigonometric_10": [0.03439628892573701, 0.03503231574134915, 0.03571919582595786,
                              0.03646522421825059, 0.03728091173885551, 0.038179862547483484,
                              0.039180141082452706, 0.040306502644076266, 0.1797201916971712,
                              0.1562408814270026]}


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


if __name__ == "__main__":
    rows = []
    print("// ROOTS")
    for name, n, f, j, x0 in problems():
        fd = lambda x: np.array(f(list(x), math), dtype=float)
        jd = lambda x: np.array(j(list(x), math), dtype=float)
        if name in EXACT:
            ref = EXACT[name]
        elif name in START:
            ref = refine(f, j, START[name])
        else:
            start = damped_newton(f, j, x0)
            ref = refine(f, j, start)
        hexes = ", ".join(f"0x{bits(t):016x}" for t in ref)
        print(f'    ("{name}", &[{hexes}]),')
        for scale in [1.0, 10.0]:
            xs = np.array(x0) * scale
            h = root(fd, xs, jac=jd, method="hybr")
            b = root(fd, xs, method="broyden1", options={"maxiter": 1000})
            if h.success:
                dist = np.max(np.abs(h.x - np.array(ref))) / max(np.max(np.abs(ref)), 1e-300)
            else:
                dist = math.nan
            rows.append((name, scale, h.success, h.nfev, b.success, dist))
    print("// SCIPY")
    for name, scale, hs, hn, bs, dist in rows:
        d = "f64::NAN" if math.isnan(dist) else f"{dist:.1e}"
        print(f'    ("{name}", {scale:.1f}, {str(hs).lower()}, {hn}, {str(bs).lower()}, {d}),')
