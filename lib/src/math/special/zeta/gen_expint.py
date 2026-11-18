"""Prints the tables of expint.rs: E₁ + ln x on (0, 1]; x eˣ E₁(x) − H₀ on [1, 2.5] (rational);
(Ei(x) − ln(x/x₀))/(x − x₀) on four pieces of (0, 6]; x e⁻ˣ Ei(x) − 1 on four pieces of [6, 48]
(rationals in scaled variables) and (x e⁻ˣ Ei(x) − 1)·x in 1/x past 48. E₁ past 2.5 is Boost.Math's
rational (expint.hpp), not fitted here. The rationals take tens of minutes."""

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

X0 = mpmath.findroot(mpmath.ei, 0.37)


def e1_small(x):
    return mpmath.e1(x) + mpmath.log(x) if x > 0 else -mpmath.euler


def ei_r(x):
    if x == 0:
        return (mpmath.euler + mpmath.log(X0)) / (-X0)
    if abs(x - X0) < mpf(10) ** -30:
        return mpmath.exp(X0) / X0 - 1 / X0
    return (mpmath.ei(x) - mpmath.log(x / X0)) / (x - X0)


def h_ei(x):
    return x * mpmath.exp(-x) * mpmath.ei(x)


def h_e1(x):
    return x * mpmath.exp(x) * mpmath.e1(x)


poly("E1_SMALL", e1_small, 0, 1, 14, 0, rel=False)
for nm, a, b, d in [("EI_R1", 0, 1.5, 12), ("EI_R2", 1.5, 3, 13), ("EI_R3", 3, 4.5, 13), ("EI_R4", 4.5, 6, 13)]:
    poly(nm, ei_r, a, b, d, (a + b) / 2)
c, h = mpf(1.75), mpf(0.75)
h0 = h_e1(c)
print(f"// E1_MID_H0 = {float(h0)!r}, low part {float(h0 - float(h0))!r}")
rat("E1_MID", lambda u: h_e1(c + h * u) - h0, -1, 1, 6, 6, w=lambda u: abs(h_e1(c + h * u)), doc="in u = (x − 1.75)/0.75,")
for nm, a, b, m in [("EI_K1", 6, 10, 7), ("EI_K2", 10, 20, 8), ("EI_K3", 20, 30, 7), ("EI_K4", 30, 48, 7)]:
    cc, hh = mpf(a + b) / 2, mpf(b - a) / 2
    rat(nm, lambda u: h_ei(cc + hh * u) - 1, -1, 1, m, m, w=lambda u: abs(h_ei(cc + hh * u)), doc=f"in u = (x − {cc})/{hh},")
g = lambda t: (h_ei(1 / t) - 1) / t
poly("EI_T1", g, 1 / 128.0, 1 / 48.0, 10, 0, full=False)
poly("EI_T2", g, 1 / 717.0, 1 / 128.0, 7, 0, full=False)
