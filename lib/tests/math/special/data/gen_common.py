"""Shared helpers of the generators of the special-function reference data (mpmath, 256 bits).

A data file has one case a line, `a1 a2 ... : hi1 lo1 hi2 lo2 ...`, each value the 16
hexadecimal digits of the bits of an f64: the exact arguments (f32 values in an `_f32` file),
then every reference value as the double-double hi + lo (a complex one as its real part, then
its imaginary part). A NaN reference means the function must give NaN, an infinite one that
infinity. See `../common.rs` for the reader and the error measures.

Each area has its own generator, `gen_<area>.py`, which imports this module; run it from this
directory with `python3 gen_<area>.py`. The data are deterministic: every generator seeds its
own `random.Random`.
"""

import math
import random
import struct

import mpmath

mp = mpmath.mp
mp.prec = 256

F64_MAX = mpmath.mpf(1.7976931348623157e308)


def bits(x):
    """The 16 hexadecimal digits of the bits of the f64 `x`."""
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def to_f32(x):
    """The f32 nearest to `x`, as a float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def dd(v):
    """The double-double `(hi, lo)` of a real mpmath value (or a float, NaN or infinity)."""
    if isinstance(v, float):
        return (v, 0.0)
    v = mpmath.mpf(v)
    if mpmath.isnan(v):
        return (math.nan, 0.0)
    if mpmath.isinf(v) or abs(v) > F64_MAX * (1 + mpmath.mpf(2) ** -54):
        return (math.copysign(math.inf, float(mpmath.sign(v))), 0.0)
    hi = float(v)
    lo = float(v - hi) if math.isfinite(hi) else 0.0
    return (hi, lo)


def refs(values):
    """The reference words of a list of values, real or complex (`mpc`)."""
    out = []
    for v in values:
        if isinstance(v, mpmath.mpc):
            out += [dd(v.real), dd(v.imag)]
        else:
            out.append(dd(v))
    return out


def line(args, values):
    """One data line of the float arguments `args` and the reference values `values`."""
    r = refs(values)
    return " ".join(bits(a) for a in args) + " : " + " ".join(
        bits(h) + " " + bits(l) for h, l in r
    )


def write(path, header, rows):
    """Writes the data file `path`: the comment `header`, then a line of each `(args, values)`."""
    with open(path, "w") as f:
        for h in header.strip().splitlines():
            f.write("# " + h + "\n")
        for args, values in rows:
            f.write(line(args, values) + "\n")
    print(f"{path}: {len(rows)} cases")


def uniform(rng, a, b, n, single=False):
    """`n` floats uniform in `[a, b]` (f32 values if `single`)."""
    out = [rng.uniform(a, b) for _ in range(n)]
    return [to_f32(x) for x in out] if single else out


def log_uniform(rng, a, b, n, single=False):
    """`n` floats with the logarithm uniform in `[ln a, ln b]`, `0 < a < b`."""
    out = [math.exp(rng.uniform(math.log(a), math.log(b))) for _ in range(n)]
    return [to_f32(x) for x in out] if single else out


def mp_arg(x):
    """The exact mpmath value of a float argument."""
    return mpmath.mpf(x)
