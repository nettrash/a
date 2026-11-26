"""Prints the tables of polylog.rs: the odd part of the dilogarithm's series in y², the piece
about the zero of Re Li₂ past 1, ln p of the primes below 256, (ζ(k) − 1)/k, and the 16-point
Gauss–Legendre rule. A minute."""

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



def dilog_p(z):
    if z == 0:
        return mpmath.bernoulli(2) / 6
    y = mpmath.sqrt(z)
    return (mpmath.polylog(2, 1 - mpmath.exp(-y)) - y + y * y / 4) / y**3


poly("DILOG_P", dilog_p, 0, mpmath.log(4) ** 2, 8, 0, full=False)
x0 = mpmath.findroot(lambda t: mpmath.polylog(2, t).real, 12.6)
g = lambda x: mpmath.polylog(2, x).real / (x - x0) if abs(x - x0) > mpf(10) ** -25 else mpmath.diff(lambda t: mpmath.polylog(2, t).real, x0)
poly("DILOG_ZERO", g, 10.5, 15, 14, x0)
print(f"const DILOG_X0_HI: f64 = {float(x0)!r};\nconst DILOG_X0_LO: f64 = {float(x0 - float(x0))!r};")
ps = [p for p in range(2, 257) if all(p % q for q in range(2, int(p**0.5) + 1))]
print("const LN_PRIMES: [(f64, f64); 54] = [" + ", ".join(f"({float(mpmath.log(p))!r}, {float(mpmath.log(p) - float(mpmath.log(p)))!r})" for p in ps) + "];")
print("const ZETA_M1_K: [f64; 28] = [" + ", ".join(repr(float((mpmath.zeta(k) - 1) / k)) for k in range(2, 30)) + "];")
n = 16
nodes = []
for i in range(1, n + 1):
    x = mpmath.cos(mpmath.pi * (i - mpf(1) / 4) / (n + mpf(1) / 2))
    for _ in range(100):
        p0, p1 = mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        x = x - p1 / dp
    nodes.append((x, 2 / ((1 - x * x) * dp * dp)))
pos = sorted(t for t in nodes if t[0] > 0)
print("const GL16_X: [f64; 8] = [" + ", ".join(repr(float(x)) for x, _ in pos) + "];")
print("const GL16_W: [f64; 8] = [" + ", ".join(repr(float(w)) for _, w in pos) + "];")
