"""Prints the tables of complex.rs and util.rs that gen_util.py does not: the Taylor coefficients
of the Riemann–Siegel function F(z) = (e^(πi(z²/2 + 3/8)) − i√2 cos(πz/2))/(2 cos πz) in z² (F is
even and entire), those of the entire ζ(s) − 1/(s − 1) about s = 1, (−1)ⁿγₙ/n! with γₙ the
Stieltjes constants, sin(πj/32) and cos(πj/32) as double-doubles, atan(k/8) as double-doubles
(util.rs), and the coefficients B₂ₖ/(2k(2k − 1))
of Stirling's series (util.rs). Seconds."""

import os
import sys
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "a-special-zeta-pycache")
sys.dont_write_bytecode = True

import mpmath

mpmath.mp.dps = 200


def dd(v):
    hi = float(v)
    return hi, float(v - hi)


def rs_f_coefficients(n):
    """The first n Taylor coefficients of F in z², by the division of the series of its numerator
    and denominator (at 200 digits: the series of 1/cos πz grows as 2²ⁿ, and the coefficients of F come from its cancellation)."""
    e38 = mpmath.expjpi(mpmath.mpf(3) / 8)
    half_pi = mpmath.pi / 2
    num = [e38 * (1j * half_pi) ** k / mpmath.factorial(k) - 1j * mpmath.sqrt(2) * (-1) ** k * half_pi ** (2 * k) / mpmath.factorial(2 * k) for k in range(n)]
    den = [2 * (-1) ** k * mpmath.pi ** (2 * k) / mpmath.factorial(2 * k) for k in range(n)]
    out = []
    for k in range(n):
        v = num[k] - sum(out[j] * den[k - j] for j in range(k))
        out.append(v / den[0])
    return out


c = rs_f_coefficients(51)
# a check against F itself at points away from its removable singularities
for z in (mpmath.mpf("0.3"), mpmath.mpf("0.5"), mpmath.mpf("-0.97"), mpmath.mpf(1)):
    f = (mpmath.expjpi(z * z / 2 + mpmath.mpf(3) / 8) - 1j * mpmath.sqrt(2) * mpmath.cos(mpmath.pi * z / 2)) / (2 * mpmath.cos(mpmath.pi * z)) if z != mpmath.mpf("0.5") else None
    s = mpmath.fsum(c[k] * z ** (2 * k) for k in range(51))
    if f is not None:
        assert abs(s - f) < mpmath.mpf(10) ** -50, (z, s, f)
# and the decay the tables rely on
assert abs(c[50]) < mpmath.mpf(10) ** -60 and all(abs(c[n + 1]) < abs(c[n]) for n in range(20, 50))
print("const RS_F: [(f64, f64); 51] = [" + ", ".join(f"({float(c[n].real)!r}, {float(c[n].imag)!r})" for n in range(51)) + "];")
st = [(-1) ** n * mpmath.stieltjes(n) / mpmath.factorial(n) for n in range(41)]
# 30 terms leave below 2⁻⁶² on |s − 1| ≤ 3 (each term past them, and their sum)
assert all(abs(st[n]) * 3 ** n < mpmath.mpf(2) ** -62 for n in range(30, 41))
assert sum(abs(st[n]) * 3 ** n for n in range(30, 41)) < mpmath.mpf(2) ** -62
st = st[:30]
print("const STIELTJES: [f64; 30] = [" + ", ".join(repr(float(c)) for c in st) + "];")
print(f"const STIELTJES_LO: f64 = {float(st[0] - float(st[0]))!r};")
print("const ATAN_EIGHTHS: [(f64, f64); 9] = [" + ", ".join("(%r, %r)" % dd(mpmath.atan(mpmath.mpf(k) / 8)) for k in range(9)) + "];")
print("const SINCOS_PI_32: [(f64, f64, f64, f64); 9] = [" + ", ".join("(%r, %r, %r, %r)" % (dd(mpmath.sinpi(mpmath.mpf(j) / 32)) + dd(mpmath.cospi(mpmath.mpf(j) / 32))) for j in range(9)) + "];")
print("const STIRLING: [f64; 10] = [" + ", ".join(repr(float(mpmath.bernoulli(2 * k) / (2 * k * (2 * k - 1)))) for k in range(1, 11)) + "];")
for name, v in [
    ("PI_2", mpmath.pi / 2),
    ("LN_2PI", mpmath.log(2 * mpmath.pi)),
    ("INV_2PI", 1 / (2 * mpmath.pi)),
    ("ONE_LN_2PI", 1 + mpmath.log(2 * mpmath.pi)),
]:
    h, l = dd(v)
    print(f"const {name}_HI: f64 = {h!r};\nconst {name}_LO: f64 = {l!r};")
