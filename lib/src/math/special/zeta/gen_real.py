"""Prints the tables of real.rs: R(s) = ζ(s) − 1/(s − 1) on [−1/2, 3] and the rationals of
(ζ(s) − 1)·2ˢ on [3, 8], [8, 20] and [20, 56] weighted relative to ζ. Tens of minutes."""

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



def r(s):
    return +mpmath.euler if s == 1 else mpmath.zeta(s) - 1 / (s - 1)


def v(s):
    return (mpmath.zeta(s) - 1) * mpf(2) ** s


poly("ZETA_R", r, -0.5, 3, 16, 1, rel=False)
for nm, a, b, m in [("ZETA_V1", 3, 8, 6), ("ZETA_V2", 8, 20, 6), ("ZETA_V3", 20, 56, 4)]:
    rat(nm, v, a, b, m, m, x0=(a + b) / 2, w=lambda s: mpf(2) ** s * mpmath.zeta(s), iters=100)
