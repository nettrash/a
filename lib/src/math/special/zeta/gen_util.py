"""Prints the tables of util.rs, complex.rs and hurwitz.rs: sin(πr)/r and (cos(πr) − 1)/r² on
|r| ≤ 1/4, 2^(j/32) as double-doubles, the split of ln 2/32, and B₂ₖ/(2k)!. Seconds."""

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


sp = lambda z: mpmath.sin(mpmath.pi * mpmath.sqrt(z)) / mpmath.sqrt(z) if z > 0 else mpmath.pi
cp = lambda z: (mpmath.cos(mpmath.pi * mpmath.sqrt(z)) - 1) / z if z > 0 else -mpmath.pi**2 / 2
poly("SIN_PI", sp, 0, 1 / 16.0, 7, 0, full=False)
poly("COS_PI", cp, 0, 1 / 16.0, 7, 0, full=False)
print("const EXP2_32: [(f64, f64); 32] = [" + ", ".join(
    f"({float(mpmath.mpf(2) ** (mpf(j) / 32))!r}, {float(mpmath.mpf(2) ** (mpf(j) / 32) - float(mpmath.mpf(2) ** (mpf(j) / 32)))!r})" for j in range(32)) + "];")
l = mpmath.log(2) / 32
m, e = mpmath.frexp(l)
hi = float(mpmath.ldexp(mpmath.floor(m * 2**36), e - 36))
print(f"const LN2_32_HI: f64 = {hi!r};\nconst LN2_32_LO: f64 = {float(l - hi)!r};")
print("const BERNOULLI_2K: [f64; 24] = [" + ", ".join(repr(float(mpmath.bernoulli(2 * k) / mpmath.factorial(2 * k))) for k in range(1, 25)) + "];")
