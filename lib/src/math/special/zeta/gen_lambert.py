"""Prints the tables of lambert.rs: the branch pieces in p = √(2(1 + e x)), the piece of W₋₁ in
s = √(2(−ln(−x) − 1)), the pieces of e^(−W₀) in p, and the starting values (in s and ln x).
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
poly("INIT_L1", w0_of_l, 1.09, 4.0, 6, 2.55, full=False)
poly("INIT_L2", w0_of_l, 4.0, 8.0, 6, 6.0, full=False)
poly("INIT_L3", w0_of_l, 8.0, 16.0, 7, 12.0, full=False)
