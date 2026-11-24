"""Writes mpmath_functions.txt: the exponential, logarithm, square root and powers of quaternions
at 60 significant digits, by mpmath, rounded to the nearest double.

Each line is `f w x y z` then, for that input, exp, ln, sqrt, q^(1/3), q^(-2.5) and q^p for
p = 0.5 + 0.25i - 0.125j + k (the logarithm multiplied by p on the right), four parts each, the
scalar first; `inf` where a part overflows. The functions are the polar forms: for q = w + v,
r = |v|, u = v/r, exp(q) = e^w (cos r + u sin r) and ln(q) = ln|q| + u atan2(r, w); where v = 0,
those of the complex number w + x i. Run with mpmath installed: `python3 -B gen_mpmath.py`.
"""

import mpmath
from mpmath import mp, mpf

from gen_common import line

mp.dps = 60


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def qexp(q):
    w, x, y, z = q
    r = mpmath.sqrt(x * x + y * y + z * z)
    e = mpmath.exp(w)
    if r == 0:
        return [e, x, y, z]
    f = e * mpmath.sin(r) / r
    return [e * mpmath.cos(r), x * f, y * f, z * f]


def qln(q):
    w, x, y, z = q
    r = mpmath.sqrt(x * x + y * y + z * z)
    re = mpmath.log(mpmath.sqrt(w * w + r * r))
    if r == 0:
        # The complex logarithm's argument, with the sign of the zero x.
        return [re, mpmath.atan2(x, w) if w < 0 else mpf(0), y, z]
    f = mpmath.atan2(r, w) / r
    return [re, x * f, y * f, z * f]


def qpow(q, p):
    return qexp(qmul(qln(q), p))


def scal(q, t):
    return [t * c for c in q]


INPUTS = [
    (1.0, 2.0, 3.0, 4.0),
    (0.5, -0.25, 0.125, 2.0),
    (1.0, 1e-10, 0.0, 0.0),
    (1.0, 1e-200, -1e-210, 3e-205),
    (1.0 + 2.0**-40, 1e-8, 2e-8, -3e-8),
    (-1.0, 1e-12, 0.0, 0.0),
    (-2.0, 0.0, 0.0, 0.0),
    (-3.0, 1e-300, 0.0, 0.0),
    (0.0, 0.0, 0.0, 5.0),
    (0.0, 1.0, 1.0, 1.0),
    (1e300, 1e300, -1e300, 1e300),
    (1e-300, 2e-300, 0.0, -3e-300),
    (3e-320, 0.0, 4e-320, 0.0),
    (1.7e308, 1e308, 0.0, 0.0),
    (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    (0.6, 0.48, 0.64, 0.0),
    (2.0, 700.0, 0.0, 0.0),
    (709.5, 0.1, 0.2, 0.3),
    (710.0, 1.0, 0.0, 0.0),
    (-700.0, 1.0, 2.0, 3.0),
    (1e-5, 3.0, 4.0, 0.0),
    (-1e-5, 3.0, -4.0, 12.0),
    (5.0, 1e-17, 0.0, 0.0),
    (0.1, 20.0, -30.0, 40.0),
]

P = [mpf(0.5), mpf(0.25), mpf(-0.125), mpf(1)]

out = ["# mpmath " + mpmath.__version__ + " at 60 digits; written by gen_mpmath.py"]
for q in INPUTS:
    m = [mpf(c) for c in q]
    vals = []
    for f in [qexp, qln, lambda a: qpow(a, [mpf(0.5), 0, 0, 0]),
              lambda a: qpow(a, [mpf(1) / 3, 0, 0, 0]), lambda a: qpow(a, [mpf(-2.5), 0, 0, 0]),
              lambda a: qpow(a, P)]:
        r = f(m)
        vals += [float(c) for c in r]
    out.append(line("f", list(q) + vals))

with open("mpmath_functions.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
