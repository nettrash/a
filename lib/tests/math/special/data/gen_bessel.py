"""The reference data of the bessel area of issue #20: J, Y, I, K (plain and scaled) of real
orders, the spherical functions and the Airy functions, f64 and f32, with mpmath at 256 bits.

Files `bessel_<func>_<type>.txt`. For `j`, `y`, `sj`, `sy` each case carries two references:
the value, then the modulus `M = √(J² + Y²)` (of the order, at the argument; for the spherical
ones `√(π/(2x))·M`), the scale of the absolute error near the zeros. Run from this directory:
`python3 gen_bessel.py`.
"""

import math
import random

import mpmath

from gen_common import log_uniform, mp_arg, to_f32, uniform, write

mp = mpmath.mp
rng = random.Random(2020)
HALF = mpmath.mpf(1) / 2


def J(n, x):
    return mpmath.besselj(n, x)


def Y(n, x):
    return mpmath.bessely(n, x)


def jy_mod(nu, x):
    if x == 0:
        return mpmath.inf
    return mpmath.sqrt(J(nu, x) ** 2 + Y(nu, x) ** 2)


def ref_j(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    if x < 0:
        v = J(nu, -x) * (-1) ** int(nu)
        return [v, jy_mod(nu, -x)]
    if x == 0:
        return [J(nu, x), mpmath.mpf(1)]
    return [J(nu, x), jy_mod(nu, x)]


def ref_y(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    return [Y(nu, x), jy_mod(nu, x)]


def ref_i(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    if x < 0:
        return [mpmath.besseli(nu, -x) * (-1) ** int(nu)]
    return [mpmath.besseli(nu, x)]


def ref_k(nu, x):
    return [mpmath.besselk(mp_arg(nu), mp_arg(x))]


def ref_is(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    s = (-1) ** int(nu) if x < 0 else 1
    return [s * mpmath.besseli(nu, abs(x)) * mpmath.exp(-abs(x))]


def ref_ks(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    return [mpmath.besselk(nu, x) * mpmath.exp(x)]


def sph(fn, n, x):
    x = mp_arg(x)
    f = mpmath.sqrt(mpmath.pi / (2 * x))
    return f * fn(n + HALF, x)


def ref_sj(n, x):
    n = int(n)
    xa = abs(mp_arg(x))
    v = sph(J, n, xa) * (-1) ** n if x < 0 else sph(J, n, xa)
    m = mpmath.sqrt(mpmath.pi / (2 * xa)) * jy_mod(n + HALF, xa)
    return [v, m]


def ref_sy(n, x):
    n = int(n)
    xa = abs(mp_arg(x))
    v = sph(Y, n, xa) * (-1) ** (n + 1) if x < 0 else sph(Y, n, xa)
    m = mpmath.sqrt(mpmath.pi / (2 * xa)) * jy_mod(n + HALF, xa)
    return [v, m]


def ref_si(n, x):
    n = int(n)
    xa = abs(mp_arg(x))
    v = sph(mpmath.besseli, n, xa)
    return [v * (-1) ** n if x < 0 else v]


def ref_sk(n, x):
    return [sph(mpmath.besselk, int(n), x)]


def airy_ref(x, d, b):
    """The value of Ai (or Bi, `b`; their derivatives, `d`) at `x`, then the modulus
    √(Ai² + Bi²) (of the derivatives for `d`), the scale of the error near the zeros."""
    x = mp_arg(x)
    a, bb = mpmath.airyai(x, d), mpmath.airybi(x, d)
    return [bb if b else a, mpmath.sqrt(a * a + bb * bb)]


AIRY = {
    'ai': lambda x: airy_ref(x, 0, False),
    'aip': lambda x: airy_ref(x, 1, False),
    'bi': lambda x: airy_ref(x, 0, True),
    'bip': lambda x: airy_ref(x, 1, True),
}


def near(v, k, single):
    """The `k` floats on each side of the value `v` (f32 ones if `single`)."""
    out = []
    if single:
        import struct
        b = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        for d in range(-k, k + 1):
            out.append(struct.unpack("<f", struct.pack("<I", b + d))[0])
    else:
        x = float(v)
        lo = hi = x
        out.append(x)
        for _ in range(k):
            lo = math.nextafter(lo, -math.inf)
            hi = math.nextafter(hi, math.inf)
            out += [lo, hi]
    return out


def zeros_j(nu, ks):
    return [mpmath.besseljzero(nu, k) for k in ks]


def zeros_y(nu, ks):
    return [mpmath.besselyzero(nu, k) for k in ks]


def f32(xs):
    return [to_f32(x) for x in xs]


def orders_args(rng, single):
    """(ν, x) pairs over the regions of the real orders."""
    s = single
    out = []
    r = lambda a, b: to_f32(rng.uniform(a, b)) if s else rng.uniform(a, b)
    lu = lambda a, b: log_uniform(rng, a, b, 1, s)[0]
    n = 26 if s else 40
    for _ in range(n):
        out.append((r(0, 1), lu(1e-3, 2)))
        out.append((r(0, 1), r(2, 30)))
        out.append((r(1, 10), lu(1e-2, 30)))
        out.append((r(0, 10), lu(30, 1e4)))
    for _ in range(n // 2):
        out.append((r(10, 100), lu(0.5, 300)))
        out.append((float(rng.randint(2, 60)), lu(0.1, 200)))
        out.append((r(-12, 0), lu(0.05, 50)))
    for _ in range(n // 6):
        out.append((float(-rng.randint(1, 30)), lu(0.1, 100)))
        out.append((r(100, 400), lu(10, 2000)))
    return out


def gen_jy():
    for single in (False, True):
        t = "f32" if single else "f64"
        cv = (lambda xs: f32(xs)) if single else (lambda xs: xs)
        for name, ref, zf in (("j", ref_j, zeros_j), ("y", ref_y, zeros_y)):
            rows = []
            for nu in (0.0, 1.0):
                m = 60 if single else 100
                xs = uniform(rng, 0, 30, m, single) + log_uniform(rng, 30, 1e15, m // 3, single)
                xs += log_uniform(rng, 1e-300 if not single else 1e-30, 1e-2, 12, single)
                ks = list(range(1, 21)) if not single else list(range(1, 11))
                for z in zf(int(nu), ks + [100, 1000, 10 ** 4]):
                    xs += near(z, 1 if not single else 0, single)
                rows += [((nu, x), ref(nu, x)) for x in xs if x > 0]
            rows += [((nu, x), ref(nu, x)) for nu, x in orders_args(rng, single)]
            if name == "j":
                for _ in range(12 if single else 20):
                    nu = float(rng.randint(0, 20))
                    x = -(to_f32(rng.uniform(0.1, 40)) if single else rng.uniform(0.1, 40))
                    rows.append(((nu, x), ref(nu, x)))
            header = f"bessel_{name}(nu, x), {t}: references the value, then the modulus M = sqrt(J^2 + Y^2)"
            write(f"bessel_{name}_{t}.txt", header, rows)


def gen_ik():
    for single in (False, True):
        t = "f32" if single else "f64"
        for name, ref in (("i", ref_i), ("k", ref_k), ("is", ref_is), ("ks", ref_ks)):
            rows = []
            m = 40 if single else (80 if name in ("i", "k") else 30)
            for nu in (0.0, 1.0):
                xs = uniform(rng, 0, 1, m // 2, single) + uniform(rng, 1, 7.75, m // 2, single)
                xs += log_uniform(rng, 7.75, 700, m // 2, single)
                xs += log_uniform(rng, 1e-30, 1e-2, 6, single)
                if name in ("is", "ks"):
                    xs += log_uniform(rng, 700, 1e10, 8, single)
                rows += [((nu, x), ref(nu, x)) for x in xs if x > 0]
            args = orders_args(rng, single)
            if name in ("is", "ks"):
                args = args[:len(args) // 3]
            rows += [((nu, x), ref(nu, x)) for nu, x in args]
            if name in ("i", "is"):
                for _ in range(8 if single else 16):
                    nu = float(rng.randint(0, 20))
                    x = -(to_f32(rng.uniform(0.1, 40)) if single else rng.uniform(0.1, 40))
                    rows.append(((nu, x), ref(nu, x)))
            if name == "k":
                # near the underflow
                for x in uniform(rng, 650, 740, 8, single):
                    nu = rng.choice([0.0, 0.5, 2.0, 7.25])
                    rows.append(((nu, x), ref(nu, x)))
            write(f"bessel_{name}_{t}.txt", f"bessel_{name}(nu, x), {t}", rows)


def gen_sph():
    for single in (False, True):
        t = "f32" if single else "f64"
        for name, ref in (("sj", ref_sj), ("sy", ref_sy), ("si", ref_si), ("sk", ref_sk)):
            rows = []
            m = 25 if single else 50
            lu = lambda a, b: log_uniform(rng, a, b, 1, single)[0]
            for _ in range(m):
                rows.append((float(rng.randint(0, 2)), lu(1e-3, 100)))
                rows.append((float(rng.randint(3, 12)), lu(1e-2, 100)))
                rows.append((float(rng.randint(13, 80)), lu(0.1, 300)))
                rows.append((float(rng.randint(0, 20)), lu(100, 1e8 if not single else 1e6)))
            if name in ("sj", "sy", "si"):
                for _ in range(m // 5):
                    rows.append((float(rng.randint(0, 10)), -lu(0.1, 50)))
            if name == "sk":
                rows = [(n, abs(x)) for n, x in rows]
            rows = [((n, x), ref(n, x)) for n, x in rows]
            extra = ", then the modulus" if name in ("sj", "sy") else ""
            write(f"bessel_{name}_{t}.txt", f"sph_bessel_{name[1]}(n, x), {t}: the value{extra}", rows)


def gen_airy():
    for single in (False, True):
        t = "f32" if single else "f64"
        for name, ref in AIRY.items():
            m = 40 if single else 70
            xs = uniform(rng, -8, 4, m, single) + uniform(rng, 4, 100, m // 3, single)
            xs += [-v for v in log_uniform(rng, 8, 1e6 if not single else 1e5, m // 2, single)]
            xs += log_uniform(rng, 1e-20, 1e-1, 6, single)
            if not single:
                xs += [-v for v in log_uniform(rng, 1e6, 1e9, 10, single)]
            zf = {"ai": lambda k: mpmath.airyaizero(k), "aip": lambda k: mpmath.airyaizero(k, 1),
                  "bi": lambda k: mpmath.airybizero(k), "bip": lambda k: mpmath.airybizero(k, 1)}[name]
            for k in list(range(1, 13)) + [50, 1000]:
                xs += near(zf(k), 1 if not single else 0, single)
            if name in ("ai", "aip"):
                xs += uniform(rng, 100, 104, 4, single)  # near the underflow of Ai at 104.2... (f64)
            rows = [((x,), ref(x)) for x in xs]
            write(f"bessel_{name}_{t}.txt", f"airy_{name}(x), {t}: the value, then the modulus", rows)


if __name__ == "__main__":
    gen_jy()
    gen_ik()
    gen_sph()
    gen_airy()
