"""Prints the tables of lambert.rs: the branch pieces in p = √(2(1 + e x)), the piece of W₋₁ in
s = √(2(−ln(−x) − 1)), the pieces of e^(−W₀) in p, the starting values (in s, and of W₀ for
3 < x ≤ e⁴⁰ in ln x) and the pieces of the f32 form of W₀ in p.
A few minutes."""

import os
import sys
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "a-special-zeta-pycache")
sys.dont_write_bytecode = True

import mpmath
from gen_fit import check_f64, lawson_rat, mpf, remez, rs

mpmath.mp.dps = 40


def poly(name, f, a, b, d, x0, full=True, rel=True, w=None, doc=""):
    """Prints the Rust table of a minimax polynomial in t − x0 (with the low part of its head)."""
    c, e = remez(f, a, b, d, rel=rel, x0=x0, w=w)
    print(f"// {name}: {doc} [{a}, {b}] in t − {x0!r}, degree {d}, error {mpmath.nstr(e, 3)}")
    print(f"const {name}: [f64; {d + 1}] = [{rs(c)}];")
    if full:
        print(f"const {name}_LO: f64 = {float(c[0] - mpf(float(c[0])))!r};")
    return c


def rat(name, f, a, b, m, n, x0=0, w=None, doc="", iters=200, npts=300):
    """Prints the Rust tables of a rational near-minimax fit p/q in t − x0."""
    p, q, e = lawson_rat(f, a, b, m, n, x0=x0, w=w, iters=iters, npts=npts)
    print(f"// {name}: {doc} [{a}, {b}] in t − {x0!r}, degree {m}/{n}, error {mpmath.nstr(e, 3)}")
    print(f"const {name}_P: [f64; {m + 1}] = [{rs(p)}];")
    print(f"const {name}_Q: [f64; {n + 1}] = [{rs(q)}];")

E = mpmath.e


def w_of_p(p):
    x = (p * p / 2 - 1) / E
    return mpf(-1) if p == 0 else mpmath.lambertw(x, 0 if p > 0 else -1).real


def h_of_p(p):
    return mpmath.exp(-mpmath.lambertw((p * p / 2 - 1) / E).real)


def w0_of_l(lv):
    return mpmath.lambertw(mpmath.exp(lv)).real


def wm1_of_s(s):
    return mpmath.lambertw(-mpmath.exp(-(s * s / 2 + 1)), -1).real


poly("BRANCH_MID", w_of_p, -0.3, 0.3, 16, 0)
poly("BRANCH_W0", w_of_p, 0.3, 0.66, 12, 0.48)
poly("BRANCH_WM1", w_of_p, -0.66, -0.3, 15, -0.48)
poly("FULL_S1", wm1_of_s, 0.6, 1.12, 10, 0.86)
for i, (a, b) in enumerate([(0.66, 1.2), (1.2, 1.8), (1.8, 2.6), (2.6, 3.5), (3.5, 4.31)]):
    poly(f"W0_H{i}", h_of_p, a, b, 14 if i < 4 else 12, (a + b) / 2)
poly("INIT_S2", wm1_of_s, 1.1, 2.0, 5, 1.55, full=False)
poly("INIT_S3", wm1_of_s, 2.0, 6.0, 6, 4.0, full=False)
poly("INIT_S4", wm1_of_s, 6.0, 12.0, 6, 9.0, full=False)


def cheb_mono(f, a, b, d):
    """The Chebyshev interpolant of degree `d` of `f` on `[a, b]` in powers of `x − (a + b)/2`
    (`mpf` coefficients)."""
    c, h = (a + b) / 2, (b - a) / 2
    n = d + 1
    xs = [mpmath.cos(mpmath.pi * (j + mpmath.mpf(1) / 2) / n) for j in range(n)]
    fs = [f(c + h * t) for t in xs]
    cheb = [2 * mpmath.fsum(fs[j] * mpmath.cos(mpmath.pi * i * (j + mpmath.mpf(1) / 2) / n) for j in range(n)) / n
            for i in range(n)]
    cheb[0] /= 2
    T = [[mpmath.mpf(1)], [mpmath.mpf(0), mpmath.mpf(1)]]
    for i in range(2, n):
        nx = [mpmath.mpf(0)] * (i + 1)
        for m_, v in enumerate(T[-1]): nx[m_ + 1] += 2 * v
        for m_, v in enumerate(T[-2]): nx[m_] -= v
        T.append(nx)
    mono = [mpmath.mpf(0)] * n
    for i in range(n):
        for m_, v in enumerate(T[i]): mono[m_] += cheb[i] * v
    return [mono[i] / h ** i for i in range(n)]


def horner_f(cf, u):
    v = 0.0
    for k in reversed(cf): v = v * u + k
    return v


# The starting values of W₀ for 3 < x ≤ e⁴⁰: W₀(eᴸ) on [1 + 2k, 3 + 2k], degree 7 in L − (2 + 2k).
rows, worst = [], 0
for k in range(20):
    a, b = mpmath.mpf(1 + 2 * k), mpmath.mpf(3 + 2 * k)
    f = lambda L: mpmath.lambertw(mpmath.exp(L)).real
    cf = [float(v) for v in cheb_mono(f, a, b, 7)]
    c = (a + b) / 2
    err = max(abs((horner_f(cf, float(L - c)) - f(L)) / f(L)) for L in (a + (b - a) * j / 200 for j in range(201)))
    worst = max(worst, err)
    rows.append(cf)
print("/// `W₀(eᴸ)` on the pieces `[1 + 2k, 3 + 2k]` of `L`, `k = 0, …, 19`: Chebyshev interpolants of")
print(f"/// degree 7 in `L − (2 + 2k)` (mpmath at 40 digits), relative error at most {mpmath.nstr(worst, 3)}.")
print("const INIT_W0: [[f64; 8]; 20] = [")
for r in rows:
    print("    [" + ", ".join(repr(v) for v in r) + "],")
print("];")


# The f32 form of W₀ on −1/e ≤ x ≤ 3: e^(−W₀) on the halves [i/2, (i + 1)/2] of p = √(2(1 + e·x)),
# degree 13 in p − (i/2 + 1/4).
def ep(p):
    if p == 0:
        return mpmath.e
    return mpmath.exp(-mpmath.lambertw((p * p / 2 - 1) / mpmath.e).real)


rows, worst = [], 0
for i in range(9):
    a, b = mpmath.mpf(i) / 2, mpmath.mpf(i + 1) / 2
    c = (a + b) / 2
    cf = [float(v) for v in cheb_mono(ep, a, b, 13)]
    err = max(abs(horner_f(cf, float(p - c)) / ep(p) - 1) for p in (a + (b - a) * j / 300 for j in range(301)))
    worst = max(worst, err)
    rows.append(cf)
print("/// `e^(−W₀(x))` for the `f32` form on `−1/e ≤ x ≤ 3`: Chebyshev interpolants of degree 13 in")
print("/// `p − (i/2 + 1/4)` on the pieces `[i/2, (i + 1)/2]` of `p = √(2(1 + e·x))`, `i = 0, …, 8`")
print(f"/// (mpmath at 40 digits), relative error at most {mpmath.nstr(worst, 2)} (an `f32` needs about `2⁻³⁴`).")
print("const W0_F32: [[f64; 14]; 9] = [")
for r in rows:
    print("    [\n" + "".join(f"        {repr(v)},\n" for v in r) + "    ],")
print("];")
