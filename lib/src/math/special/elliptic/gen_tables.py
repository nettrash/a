"""Writes tables.rs next to this script: the fitted tables of K(m) and E(m).

Chebyshev fits (mpmath.chebyfit at 200 bits) of
  * K(m) and E(m) in t = m - c on intervals of width 0.05: degree 9 on [-1, 0.5) (30 rows) and
    degree 12 on [0.5, 0.8) (6 rows);
  * for m1 = 1 - m in [0, 0.2] (DLMF 19.12.1-2): K(m) = -ln(m1) K(m1)/pi + B(m1) and
    E(m) = 1 + m1 (-ln(m1) S1(m1) + S2(m1)), with K(m1)/pi (degree 12), B (11), S1 (12), S2 (11).
Each row is [centre, c0 hi, c0 lo, c1, ...]: the coefficients rounded to f64, the constant term a
double-double. The script checks every fit on a grid of 401 points (relative error below 2^-56.9;
2^-57 but for S1, whose term is below 0.32 of E) and writes the file exactly as rustfmt leaves it
(the tables are #[rustfmt::skip]). It needs no cache.

Run: python3 gen_tables.py (mpmath 1.3; about a minute).
"""

import os
import sys

import mpmath as mp

mp.mp.prec = 200


def dd(x):
    hi = float(x)
    return hi, float(x - mp.mpf(hi))


def lit(x):
    r = repr(float(x))
    if "e" not in r and "." not in r and "inf" not in r and "nan" not in r:
        r += ".0"
    return r


def fit(f, a, b, deg):
    c = mp.mpf(float((a + b) / 2))
    p = mp.chebyfit(lambda t: f(c + t), [a - c, b - c], deg + 1)[::-1]
    hi, lo = dd(p[0])
    coefs = [float(q) for q in p[1:]]
    worst = mp.mpf(0)
    for i in range(401):
        t = (a - c) + (b - a) * mp.mpf(i) / 400
        acc = mp.mpf(0)
        for q in reversed(coefs):
            acc = acc * t + mp.mpf(q)
        v = mp.mpf(hi) + mp.mpf(lo) + acc * t
        worst = max(worst, abs(v / f(c + t) - 1))
    assert worst < mp.mpf(2) ** -56.9, (a, b, deg, worst)
    return [float(c), hi, lo] + coefs


out = [
    "//! The fitted tables of `K(m)` and `E(m)`, written by `gen_tables.py` (next to this file):",
    "//! Chebyshev fits made with mpmath at 200 bits, coefficients rounded to `f64` and the",
    "//! constant terms kept as double-doubles; the relative error of each fit is below 2^-57 on",
    "//! its interval.",
    "",
]


def table(name, f, a0, n, deg):
    w = mp.mpf(1) / 20
    rows = [fit(f, mp.mpf(a0) + w * i, mp.mpf(a0) + w * (i + 1), deg) for i in range(n)]
    a0s = lit(a0).rstrip("0").rstrip(".") if a0 != 0.5 else "0.5"
    out.append(f"/// {n} intervals of width 0.05 from {a0s}, degree {deg}: centre, c0 (hi, lo), c1..c{deg}.")
    out.append("#[rustfmt::skip]")
    out.append(f"pub(super) const {name}: [[f64; {deg + 3}]; {n}] = [")
    for r in rows:
        out.append("    [" + ", ".join(lit(x) for x in r) + "],")
    out.append("];")


def one(name, doc, f, deg):
    r = fit(f, mp.mpf(0), mp.mpf("0.2"), deg)
    out.append(f"/// {doc}degree {deg}: centre, c0 (hi, lo), c1..c{deg}.")
    out.append("#[rustfmt::skip]")
    out.append(f"pub(super) const {name}: [f64; {deg + 3}] = [" + ", ".join(lit(x) for x in r) + "];")


def B(m1):
    if m1 == 0:
        return mp.log(4)
    return mp.ellipk(1 - m1) + mp.log(m1) / mp.pi * mp.ellipk(m1)


J = 120
cj = [mp.rf(mp.mpf(1) / 2, j) * mp.rf(mp.mpf(3) / 2, j) / (mp.factorial(j) * mp.factorial(j + 1)) for j in range(J)]
dj = [mp.digamma(1 + j) - mp.digamma(mp.mpf(1) / 2 + j) - mp.mpf(1) / ((2 * j + 1) * (2 * j + 2)) for j in range(J)]


def S1(m1):
    return sum(cj[j] * m1**j for j in range(J)) / 4


def S2(m1):
    return sum(cj[j] * dj[j] * m1**j for j in range(J)) / 2


table("K_LO", mp.ellipk, -1, 30, 9)
table("K_HI", mp.ellipk, 0.5, 6, 12)
table("E_LO", mp.ellipe, -1, 30, 9)
table("E_HI", mp.ellipe, 0.5, 6, 12)
one("K_B", "On m1 in [0, 0.2], ", B, 11)
one("E_S1", "On m1 in [0, 0.2], ", S1, 12)
one("E_S2", "On m1 in [0, 0.2], ", S2, 11)
one("K_P", "K(m1)/π on m1 in [0, 0.2], ", lambda m1: mp.ellipk(m1) / mp.pi, 12)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.rs")
with open(path, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(path, file=sys.stderr)
