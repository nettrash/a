"""Reference data of the erf area (issue #20): erf, erfc, erfcx, their inverses, Dawson's
integral, the Fresnel integrals, and the complex erf, erfc, erfcx and Faddeeva function, with
mpmath at 256 bits, and more where an argument needs it. Run from this directory:
`python3 gen_erf.py`. Deterministic.

A complex case has three references: the real part, the imaginary part, and the scale of its
error bound: the larger of the two terms that the library's formula subtracts there (2e^(-z^2)
and w(-z) below the real axis, 1 and erfc(+-z) for erf, 2 and erfc(-z) for erfc left of the
imaginary axis, 2e^(z^2) and erfcx(-z) for erfcx there), or |v| where nothing cancels. A scale
beyond the largest double (|v| is then beyond it too, or close) is written negated and divided by
2^1024, so that the error of a finite part next to an infinite one can be measured against it.
Every complex value is computed twice, 128 bits apart, and both parts must agree to 110 bits.
"""

import math
import random
import struct

import mpmath

from gen_common import F64_MAX, log_uniform, mp, to_f32, uniform, write

M = mpmath.mpf


def erfcx(x):
    x = M(x)
    if x > 1e100:
        # 1/(x√π)·(1 − 1/(2x²) + …): the next term is below 2⁻⁶⁰⁰ there.
        return 1 / (x * mpmath.sqrt(mpmath.pi)) * (1 - 1 / (2 * x * x))
    return mpmath.exp(x * x) * mpmath.erfc(x)


def dawson(x):
    x = M(x)
    if abs(x) > 1e100:
        return 1 / (2 * x) * (1 + 1 / (2 * x * x))
    return mpmath.sqrt(mpmath.pi) / 2 * mpmath.exp(-x * x) * mpmath.erfi(x)


def fresnel(x):
    x = M(x)
    if abs(x) > 1e30:
        # 1/2 ± O(1/x): the oscillating part is below 2⁻¹⁰⁰.
        return [M(0.5) * mpmath.sign(x)] * 2
    return [mpmath.fresnels(x), mpmath.fresnelc(x)]


def erfc_inv(c):
    c = M(c)
    if c == 0:
        return mpmath.inf
    if c == 2:
        return -mpmath.inf
    if c > 1:
        return -erfc_inv(2 - c)
    if c > 0.5:
        return mpmath.erfinv(1 - c)
    # Newton on ln erfc(y) = ln c.
    y = mpmath.sqrt(-mpmath.log(c))
    for _ in range(200):
        e = mpmath.erfc(y)
        y2 = y + (mpmath.log(e) - mpmath.log(c)) * e * mpmath.sqrt(mpmath.pi) / 2 * mpmath.exp(y * y)
        if abs(y2 - y) < M(2) ** -250 * y:
            return y2
        y = y2
    raise ValueError(c)


def erf_inv(x):
    x = M(x)
    if abs(x) == 1:
        return mpmath.inf * mpmath.sign(x)
    if abs(x) > 1:
        return mpmath.nan
    if abs(x) <= 0.5:
        return mpmath.erfinv(x)
    return mpmath.sign(x) * erfc_inv(1 - abs(x))


INF, NAN = math.inf, math.nan
REAL = {
    "erf": lambda x: [M(1) * (1 if x > 0 else -1) if math.isinf(x) else mpmath.erf(M(x))],
    "erfc": lambda x: [M(0) if x == INF else M(2) if x == -INF else mpmath.erfc(M(x))],
    "erfcx": lambda x: [M(0) if x == INF else mpmath.inf if x == -INF else erfcx(x)],
    "erf_inv": lambda x: [erf_inv(x)],
    "erfc_inv": lambda x: [mpmath.nan if not 0 <= x <= 2 else erfc_inv(x)],
    "dawson": lambda x: [M(0) if math.isinf(x) else dawson(x)],
    "fresnel": lambda x: [M(0.5) * (1 if x > 0 else -1)] * 2 if math.isinf(x) else fresnel(x),
}


def real_rows(name, xs):
    rows = []
    for x in xs:
        if math.isnan(x):
            rows.append(([x], [NAN] * (2 if name == "fresnel" else 1)))
        else:
            rows.append(([x], REAL[name](x)))
    return rows


def neg(xs):
    return [-x for x in xs]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def around(bits, k=2):
    """The double of the bits `bits` and its `k` neighbours on either side."""
    return [from_bits(bits + d) for d in range(-k, k + 1)]


# The arguments where the audit of 2026-11-20 found the largest errors (s_erf-5, s_erf-7) and
# where a doc figure or a test bound was exceeded, with their neighbours.
WORST = {
    "erf": [0xBFDE58E15B63ACE0],
    "erfc": [0x3FDF96DF86FD888A, 0x4011FFB6849BB2C1],
    "erfcx": [0xBFD7BFBADD228C4C, 0x4011DBA8EAFD79F5, 0x7FB3F617951893A7, 0x7FB7E4D32188B3C1, 0x7FA8E40C126C6DC3],
    "erf_inv": [0xBFE09370E8F9F4B5],
    "erfc_inv": [0x3FF85167D19E6523],
    "dawson": [0xBFDFD6EEBCCC5912, 0x4011F7339F9A252D],
    "fresnel": [0xBFEFC6CDCA356E4C, 0x3FFD56D9EED957A7, 0x4022CA115C9F9430, 0xBFEF0F676040E97C,
                0x3FF8E07BA209E902, 0x40273C4566F0F90A],
}


# The worst arguments of the review of 2026-11-21 and of a dense search after it (2 to 6 million
# arguments a window and a local search; the bounds of docs/math/special.md are above them).
WORST2 = {
    "erf": [0xBFDDEFB74952CD3C, 0x3FDE62F2D3333316, 0x3FE006F22BA04667],
    "erfc": [0x3FDFBDFA2A0B81CE, 0x3FDFDC2AC406C1D1, 0x4011FF32AD3ACCE8, 0xBFE026828737E1A2],
    "erfcx": [0xBFD7EBC784C6FEFE, 0xBFD7D4F09AF423C0, 0x4011D7471BA4166B],
    "erf_inv": [0x3FDFF09219EAB9AA, 0x3FE0A398F436A424, 0x3FE0A0A2C8600854],
    "erfc_inv": [0x3FDEB428F6DAD674, 0x3FF83738647BFDE2, 0x3FDECC81B9427584],
    "dawson": [0x3FDFE4A872533F6B, 0xBFDF517AA8F3FD29, 0xBFDFD873263AFE74, 0x3FDFEB0161652B97, 0x4011F70535BC3AC0],
    "fresnel": [0x3FEFC48C9BE06E48, 0x3FEFE8ADC6EEBE5E, 0x3FEFFF74EC9098FA],
}


# The worst arguments of the review of 2026-11-22 (dense searches: 4 million arguments a region and
# a local search; Fresnel from 1: 40 million and windows), above the largest errors found before.
WORST3 = {
    "erf": [0x3FE00A0D3069EE0A],
    "erfc": [0x3FDFBD9F64599938, 0x4011FFB5C5BFD1B0],
    "erfcx": [0x4011FFF8C49CF4BD, 0x40221EE3D875F203],
    "erf_inv": [0x3FE0A245D4F5F459],
    "erfc_inv": [0x3FDEF5CD96A55571],
    "dawson": [0x3FDFFC4F400076FB, 0x4011F6C5D74A7D2F],
    "fresnel": [0x3FF93207B33B633A, 0x3FF8F3251DC24AE7, 0x3FFDE3BCB3F55900, 0x3FFDDA4F374B2720, 0x3FEFE360561B1FEB,
                0x3FEFD2D8771DA0E7],
}


def gen_real(rng, name, single):
    """The arguments of a real function, by region (f32 values if `single`)."""
    s = single
    n = (lambda k: max(k // 5, 8)) if s else (lambda k: k)
    top = 3e38 if s else 1.7e308
    tiny = 1e-44 if s else 1e-310
    sp = [0.0, -0.0, INF, -INF, NAN]
    if name == "erf":
        xs = (log_uniform(rng, tiny, 1e-5, n(40), s) + uniform(rng, -0.5, 0.5, n(250), s)
              + uniform(rng, 0.5, 2, n(200), s) + uniform(rng, 2, 6, n(150), s)
              + neg(uniform(rng, 0.5, 6, n(100), s)) + log_uniform(rng, 6, 30, n(20), s))
    elif name == "erfc":
        xs = (uniform(rng, -0.5, 0.5, n(200), s) + neg(uniform(rng, 0.5, 6.5, n(150), s))
              + uniform(rng, 0.5, 2, n(200), s) + uniform(rng, 2, 8, n(200), s)
              + uniform(rng, 8, 26.5, n(200), s) + uniform(rng, 26.5, 27.3, n(60), s)
              + log_uniform(rng, tiny, 1e-5, n(20), s) + [27.5, 1e10, -1e10, 7.0 if s else 26.55])
        if s:
            xs = [x for x in xs if x < 10.1 or x > 11]  # f32 erfc underflows past 10.05
    elif name == "erfcx":
        xs = (neg(uniform(rng, 0.5, 26.6, n(150), s)) + uniform(rng, -0.5, 0.5, n(200), s)
              + uniform(rng, 0.5, 8, n(300), s) + log_uniform(rng, 8, 1e4, n(150), s)
              + log_uniform(rng, 1e4, top, n(60), s) + [-27.0])
        if s:
            xs = [x for x in xs if x > -9.4]  # 2e^(x²) is beyond f32 there
    elif name == "erf_inv":
        xs = (uniform(rng, -0.5, 0.5, n(200), s) + uniform(rng, 0.5, 1, n(100), s)
              + neg(uniform(rng, 0.5, 1, n(50), s))
              + [1 - v for v in log_uniform(rng, 1e-16 if not s else 6e-8, 0.5, n(150), s)]
              + log_uniform(rng, tiny, 1e-3, n(40), s) + [1.0, -1.0, 1.5, -1.0000001])
        xs = [to_f32(x) for x in xs] if s else xs
    elif name == "erfc_inv":
        xs = (uniform(rng, 0, 2, n(200), s) + log_uniform(rng, 1e-300 if not s else 1e-37, 0.5, n(200), s)
              + ([] if s else log_uniform(rng, 5e-324, 1e-300, 40))
              + [2 - v for v in log_uniform(rng, 1e-16 if not s else 6e-8, 0.5, n(100), s)]
              + uniform(rng, 0.5, 1.5, n(100), s) + [0.0, 2.0, -0.5, 2.5])
        xs = [to_f32(x) for x in xs] if s else xs
    elif name == "dawson":
        xs = (log_uniform(rng, tiny, 1e-4, n(30), s) + uniform(rng, -0.5, 0.5, n(200), s)
              + uniform(rng, 0.5, 8, n(300), s) + log_uniform(rng, 8, 1e6, n(100), s)
              + log_uniform(rng, 1e6, top, n(30), s) + neg(uniform(rng, 0.5, 20, n(50), s)))
    elif name == "fresnel":
        xs = (log_uniform(rng, tiny, 1e-4, n(30), s) + uniform(rng, 0, 1, n(200), s)
              + uniform(rng, 1, 8, n(300), s) + log_uniform(rng, 8, 1e5, n(100), s)
              + log_uniform(rng, 1e5, top, n(50), s) + neg(uniform(rng, 0, 10, n(50), s)))
    xs = xs + sp
    if not s:
        # Added with the fixes of 2026-11-21, from their own generator so that the cases above
        # stay as they were: the audit's worst arguments, and the top binades of erfcx and
        # Dawson's integral (from 2¹⁰¹⁵ the low words of 1/x were subnormal; subnormal results
        # from about 2.5·10³⁰⁷).
        r2 = random.Random(f"erf {name} fixes")
        for b in WORST[name]:
            xs += around(b)
        if name in ("erfcx", "dawson"):
            big = [2.0 ** r2.uniform(1015, 1024) for _ in range(80)]
            big = [x for x in big if x < F64_MAX]
            xs += big if name == "erfcx" else [x * r2.choice([-1, 1]) for x in big]
            xs += [2.0 ** 960, from_bits(0x7FEFFFFFFFFFFFFF), 2.5e307]
        # Added after the review of 2026-11-21, from a third generator: the worst arguments of a
        # dense search (hunt_r.py), and more of the range from 2^960 where erfcx and Dawson's
        # integral are correctly rounded (the tests check their bits there).
        r3 = random.Random(f"erf {name} review")
        for b in WORST2[name]:
            xs += around(b)
        if name in ("erfcx", "dawson"):
            mid = [2.0 ** r3.uniform(960, 1015) for _ in range(30)]
            xs += mid if name == "erfcx" else [x * r3.choice([-1, 1]) for x in mid]
        # Added after the review of 2026-11-22: its worst arguments.
        for b in WORST3[name]:
            xs += around(b)
    return xs


# ---- complex functions: a reference for any finite argument ----


def work_bits(x, y):
    """Bits for z = x + iy: 2·log₂|z| for the phase of e^(∓z²); next to an axis, so that a part
    far below |v| is resolved too, 1.5|z|² below |z| = 40 (e^(−x²) in Re w(x + iy)) and the bits
    of the small coordinate over the large one (a part of the size of the small coordinate)."""
    ax, ay = abs(x), abs(y)
    a = max(ax, ay, 1.0)
    m = min(ax, ay)
    p = 256 + 64 + 2 * math.log2(a)
    if m < 1e-3 * a:
        if a < 40:
            p += 1.5 * a * a
        if m > 0:
            p += math.log2(a) - math.log2(m)
    return int(p)


def w_series(z):
    """w(z) for |z| > 10⁴ in the closed upper half-plane: i/(√π z)·Σ (2n − 1)!!/(2z²)ⁿ while the
    terms fall (what it leaves out is below e^(−|z|²)·|w|, but e^(−x²) on the real axis)."""
    s = t = mpmath.mpc(1)
    z2 = 2 * z * z
    n = 1
    while True:
        t = t * (2 * n - 1) / z2
        if abs(t) < M(2) ** (-mp.prec - 10):
            break
        s += t
        n += 1
    return 1j / (mpmath.sqrt(mpmath.pi) * z) * s


def w_at(z):
    """w(z) at the current precision."""
    if max(abs(z.real), abs(z.imag)) > 1e4:
        if z.imag >= 0:
            return w_series(z)
        return 2 * mpmath.exp(-z * z) - w_series(-z)
    return mpmath.exp(-z * z) * mpmath.erfc(-1j * z)


def value_scale(fn, z):
    """The value of `fn` at z and the scale of its error bound, at the current precision."""
    x, y = z.real, z.imag
    huge = max(abs(x), abs(y)) > 1e4
    # (The larger of the two terms: up to 2026-11-22 their sum, up to twice as large at a zero.)
    if fn == "faddeeva":
        v = w_at(z)
        s = max(2 * abs(mpmath.exp(-z * z)), abs(w_at(-z))) if y < 0 and x != 0 else abs(v)
    elif fn == "erfcx":
        v = w_at(1j * z)
        s = max(2 * abs(mpmath.exp(z * z)), abs(w_at(-1j * z))) if x < 0 and y != 0 else abs(v)
    elif fn == "erfc":
        if x == 0:
            v = 1 - 1j * mpmath.erfi(y)
        elif not huge:
            v = mpmath.erfc(z)
        elif x > 0:
            v = mpmath.exp(-z * z) * w_at(1j * z)
        else:
            v = 2 - mpmath.exp(-z * z) * w_at(-1j * z)
        s = max(M(2), abs(2 - v)) if x < 0 and y != 0 else abs(v)
    elif fn == "erf":
        if x == 0:
            v = 1j * mpmath.erfi(y)
        elif not huge:
            v = mpmath.erf(z)
        elif x > 0:
            v = 1 - mpmath.exp(-z * z) * w_at(1j * z)
        else:
            v = mpmath.exp(-z * z) * w_at(-1j * z) - 1
        if y == 0 or x == 0 or x * x + y * y < 1:
            s = abs(v)
        else:
            s = max(M(1), abs(1 - v) if x > 0 else abs(1 + v))
    if huge and fn in ("erf", "erfc") and 0 < abs(x) < 30 and abs(v) < M(2) ** 1000:
        # Next to the imaginary axis e^(−z²)·w(iz) needs the part of w(iz) beyond its series.
        raise ValueError(f"no reference for {fn} at {z}")
    return mpmath.mpc(v), M(s)


def complex_ref(fn, x, y, parts=False):
    """`value_scale` at `work_bits` and then at 128, 256, … more until two in a row agree: each
    part to 110 bits (or below 2⁻¹²⁰⁰, far under the least subnormal), the scale to 60. Without
    `parts`, a part whose uncertainty is below 2⁻²⁰⁰ of |v| is taken as it is, or as 0 if not 20
    of its bits are settled (Im erf(z) = −Im erfc(−z), of size e^(y² − x²) next to the real
    axis, say): only an error relative to |v| or to the scale is measured on such a case. With
    `parts` (the cases whose parts are checked on their own) every part must be settled."""
    z = mpmath.mpc(x, y)
    p = work_bits(x, y)
    with mpmath.workprec(p):
        prev, s_prev = value_scale(fn, z)
    for extra in (128, 256, 512, 1024, 2048):
        with mpmath.workprec(p + extra):
            cur, s_cur = value_scale(fn, z)
            out = []
            for a, b in ((prev.real, cur.real), (prev.imag, cur.imag)):
                d = abs(a - b)
                if mpmath.isinf(b) or mpmath.isnan(b) or d <= M(2) ** -110 * max(abs(b), M(2) ** -1200):
                    out.append(b)
                elif not parts and d <= M(2) ** -200 * abs(cur):
                    out.append(b if abs(b) > M(2) ** 20 * d else M(0))
            if len(out) == 2 and (mpmath.isinf(s_cur) or abs(s_prev - s_cur) <= M(2) ** -60 * s_cur):
                return mpmath.mpc(out[0], out[1]), s_cur
        prev, s_prev = cur, s_cur
    raise ValueError(f"{fn} at ({x!r}, {y!r}): no settled reference")


def scale_word(s):
    """The scale as it is written: itself, or beyond the largest double negated and divided by
    2^1024 (exact)."""
    return s if s <= F64_MAX else -s * M(2) ** -1024


def complex_rows(name, pts, keep=lambda v, s: 1e-290 < abs(v) < F64_MAX, parts=False):
    """The rows of `pts` whose value passes `keep` (the first sets: |v| in the normal range)."""
    rows = []
    for x, y in pts:
        v, s = complex_ref(name, x, y, parts)
        if keep(v, s):
            rows.append(([x, y], [v, scale_word(s)]))
    return rows


def box(rng, a, b, c, d, n):
    return [(rng.uniform(a, b), rng.uniform(c, d)) for _ in range(n)]


def polar(rng, lo, hi, n, upper=False):
    out = []
    for _ in range(n):
        r = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        t = rng.uniform(0 if upper else -math.pi, math.pi)
        out.append((r * math.cos(t), r * math.sin(t)))
    return out


def axes(rng, n):
    out = []
    for _ in range(n):
        t = rng.uniform(-12, 12)
        e = 10 ** rng.uniform(-12, 0) * rng.choice([-1, 1])
        out.append((t, e) if rng.random() < 0.5 else (e, t))
    return out


def diagonal(rng, n, lo=1e2, hi=1.7e308):
    """Next to the diagonal Re z = Im z (first quadrant), |Re z| log-uniform: y = √(x² + u),
    |u| < 300, where e^(−z²) has a representable size, up to 10⁹; on it beyond (y² − x² is 0 or
    beyond 2⁵¹² there): the phase −2xy of e^(−z²) is all that matters (s_erf-1, 2, 3)."""
    out = []
    for _ in range(n):
        x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        y = math.sqrt(max(x * x + rng.uniform(-300, 300), 0.0)) if x < 1e9 else x
        out.append((x, y))
    return out


# Zeros of erf and erfc in the first and second quadrants (refined by findroot below); w(z) is
# zero at i times a zero of erfc, erfcx at the zeros of erfc.
ERF_ZEROS = [(1.4506161632, 1.8809163586), (2.2446592738, 2.6165751407), (2.8397410469, 3.1756280996),
             (3.3354607354, 3.6461743764)]
ERFC_ZEROS = [(-1.3548809296, 1.9914668430), (-2.1770501160, 2.6912073476), (-2.7714108506, 3.2349458718),
              (-3.2721457657, 3.7029006065)]


def near_zeros(rng, fn, n_each):
    """Points at distances 10⁻¹ to 10⁻¹² from the first zeros of `fn`, in each quadrant where it
    has zeros: there the value is a difference of two far larger terms."""
    with mpmath.workprec(200):
        if fn == "erf":
            f, guesses = mpmath.erf, ERF_ZEROS
        else:
            f, guesses = mpmath.erfc, ERFC_ZEROS
        zeros = [mpmath.findroot(f, mpmath.mpc(a, b)) for a, b in guesses]
    out = []
    for z0 in zeros:
        if fn == "faddeeva":
            z0 = 1j * z0  # lower half-plane
        for _ in range(n_each):
            d = 10 ** rng.uniform(-12, -1)
            t = rng.uniform(0, 2 * math.pi)
            p = (float(z0.real) + d * math.cos(t), float(z0.imag) + d * math.sin(t))
            # the images of the zero under the function's symmetries
            if fn == "erf":
                p = rng.choice([p, (-p[0], -p[1]), (p[0], -p[1]), (-p[0], p[1])])
            elif fn in ("erfc", "erfcx"):
                p = rng.choice([p, (p[0], -p[1])])
            else:
                p = rng.choice([p, (-p[0], p[1])])
            out.append(p)
    return out


def axis_strip(rng, n):
    """Next to the real axis, above it (0 < y ≤ 10⁻²): Re w is the Voigt function there, e^(−x²)
    but a part of the size of y; up to 2026-11-21 it was dropped for 6.708 ≤ |x| < 10 (s_erf-4)."""
    out = []
    for i in range(n):
        if i % 3 == 0:
            x = rng.uniform(6.0, 10.5)
        elif i % 3 == 1:
            x = rng.uniform(0, 30)
        else:
            x = math.exp(rng.uniform(math.log(30), math.log(1e4)))
        y = math.exp(rng.uniform(math.log(1e-300), math.log(1e-2)))
        out.append((x * rng.choice([-1, 1]), y))
    return out


def imag_strip(rng, n):
    """Next to the imaginary axis (|Re z| ≤ 10⁻⁶, 1 ≤ |Im z| < 26.5): erfc(z) = 1 − x·(…) there,
    whose 1 was lost for 6.708 ≤ |Im z| < 10 (s_erf-4)."""
    out = []
    for i in range(n):
        y = rng.uniform(6.6, 10.1) if i % 2 == 0 else rng.uniform(1, 26.5)
        x = math.exp(rng.uniform(math.log(1e-300), math.log(1e-6)))
        out.append((x * rng.choice([-1, 1]), y * rng.choice([-1, 1])))
    return out


def near_imag(rng, n):
    """Next to the imaginary axis where erf and erfc are taken part by part, |x|·max(|y|, 1) < 2^-14:
    every size of x, |y| up to 38.3 (where the real part is infinite)."""
    out = []
    for i in range(n):
        y = (rng.uniform(1, 26.5), rng.uniform(26.5, 38.3), math.exp(rng.uniform(math.log(1e-3), 0.0)))[i % 3]
        x = 2.0 ** -14 / max(y, 1.0) * math.exp(rng.uniform(math.log(1e-320), 0.0))
        out.append((max(x, 5e-324) * rng.choice([-1, 1]), y * rng.choice([-1, 1])))
    return out


def near_neg_imag(rng, n):
    """Below the real axis next to the imaginary one, |x|·max(-y, 1) < 2^-30: every size of x,
    -y up to 38.3 (where the real part is infinite)."""
    out = []
    for i in range(n):
        a = (rng.uniform(1, 26.0), rng.uniform(26.0, 38.3), math.exp(rng.uniform(math.log(1e-3), 0.0)))[i % 3]
        x = 2.0 ** -30 / max(a, 1.0) * math.exp(rng.uniform(math.log(1e-320), 0.0))
        out.append((max(x, 5e-324) * rng.choice([-1, 1]), -a))
    return out


# The worst Re w next to the real axis of the review and of a search (hunt_rew.py).
AXIS_WORST = [(-8.363252259655319, 2.25184482664022e-07)]

# The worst parts next to the axes of the review of 2026-11-22: Re w at the thresholds of the pole
# correction, Im erf and Im erfc next to the imaginary axis, Im w below the real axis next to it.
STRIP_WORST3 = {"faddeeva": [(8.551442249917864, 1.6563031099832787e-12)],
                "erf": [(1.05813843253742e-09, -14.59963235221551)],
                "erfc": [(8.650640207098632e-07, 0.7265722094810152)],
                "faddeeva_imag": [(-4.998581875467942e-216, -4.2544938692689565)]}


# Fixed arguments of the fixes of 2026-11-21: where the audit found the defects, with exact or
# infinite values that the library got wrong (the first block of each), then the worst cases of
# the bounds (s_erf-7).
FIXED = {
    "faddeeva": [(-0.21784117298601746, 8.442636479838443), (-0.1509360353553535, 0.5475049014826214),
                 (1438.7111121821595, 6619.206251242638),
                 (8.9e307, 8.9e307), (9.1e307, 9.1e307), (1.2e308, 1.2e308), (1.5e308, 9e307),
                 (-1.2e308, 1.2e308), (1.7976931348623157e308, 1.7976931348623157e308),
                 (1.0e308, 1e-300), (2.0 ** 1020, 2.0 ** 1020), (2.0 ** 1020 * 0.999, 1.0)],
    "faddeeva_lhp": [(30000.7, -30000.7), (123456.789, -123456.789), (1000000.3, -1000000.3),
                     (12345678.9, -12345678.9), (1.1e12, -1.1e12), (0.0, -1e200), (1.0, -1e200),
                     (1e300, -1e8), (1e300, -1e7), (1e308, -1.0), (1e160, -1.0), (1.0, -1.3e154),
                     (1.0, -1.4e154), (1e154, -1e154), (1.7976931348623157e308, -1.7976931348623157e308),
                     (0.0, -0.5), (0.0, -3.0), (-0.0, -10.0), (0.0, -26.6), (0.0, -27.0)],
    "erf": [(1.1e12, 1.1e12), (1.1e15, 1.1e15), (1.1e20, 1.1e20), (1.0, 1e200), (0.0, 1e200),
            (1.0, 1.3e154), (1.0, 1.4e154), (1e300, 1e8), (1e300, 1e7), (-1e300, -1e8),
            (1.2e154, 1e154), (-1e200, 3e200), (1e154, 1e154)],
    "erfc": [(0.5645016766659705, 0.07046203812070084), (12345678.9, 12345678.9), (1.1e12, 1.1e12),
             (1.0, 1e200), (0.0, 1e200), (0.0, 30.0), (0.0, -27.5), (1e300, 1e8), (-1e300, 1e8),
             (1e154, 1e154), (-1e154, -1e154)],
    "erfcx": [(0.8842210506881638, 0.243353986726081), (63169.32443041885, 10509.650314808161),
              (-1.1e12, 1.1e12), (-1e200, 1.0), (-1e8, 1e300), (1.2e308, -1.2e308), (-1e154, 1e154)],
}


# Added after the review of 2026-11-21: the worst arguments of the reviewer and of a search
# (hunt_c.py: a million random arguments a region and a local search), then arguments where the
# exact value overflows but the code returned finite parts (a power of 2 clamped to 2040), and
# finite parts next to infinite ones (the near-axis paths).
FIXED2 = {
    "faddeeva": [(0.7397183939392469, 5.863081434266064), (0.7397183939392467, 5.863081434266043),
                 (-1.1992951927332254, 3.8255853861227784), (-2.1182770932088215, 7.994027541184547),
                 (5147.946059092685, 257810.81971878046), (-25.78404983922919, 8116.211051703858)],
    "faddeeva_lhp": [(-5.9684828747798, -5.678659419336358), (-4.610160026835596, -0.04725147679567022),
                     (-2e-323, -5690.354600842763), (-1e-320, -30.0), (1e-300, -37.6), (2.85107016464e-313, -37.604284273223065),
                     (-6.1361773503e-313, -37.41680426258388), (5e-324, -1e300), (-1e-200, -1.4e154), (3e-320, -26.7),
                     (1e-300, -38.2), (1.0, -38.3), (1e-10, -37.0)],
    "erf": [(-0.7407857919299866, 4.478517758225923), (-4.131741060814937, 4.486401144015711),
            (0.8163100928862023, 8.822969071072364), (-1.5968270547301837, 8.647054927727657),
            (1.0, 1e306), (4.053377871816522e-18, 2.852998099721371e306), (2.3195109627e-313, 7.16529296807247e242),
            (-1e-320, 1e300), (1e-310, 30.0), (1e-100, 30.0), (-1.3e-317, -37.5), (1e-30, 38.0), (1.0, 38.2)],
    "erfc": [(0.5568424316888304, 0.23130697906629447), (0.5568424316887816, 0.2313069790661766),
             (4.072415615515918, -0.5518245803433128), (-2.716470231401634, 3.2166514553218577),
             (5.758398163661839, -1.1940037581900185), (1.0, 1e306), (1e-320, 30.0), (-1e-200, 1e300),
             (2.0194673485008142e-94, -14.684011985237424), (1.2708717e-317, 37.50578270415098), (1.0, 38.2)],
    "erfcx": [(-0.6171450975264149, 4.39679158651839), (0.5476792406378858, -0.23809422204349834),
              (0.5476792406378665, -0.23809422204350345), (-5.688101511795715, -6.031881740659186),
              (0.5477390308534561, -0.33154687950632683), (-12076671.741680516, -2.851648e-317), (-30.0, 1e-320),
              (-37.6, -1e-300), (-1e300, 5e-324), (-38.3, -1.0), (-37.0, -1e-10)],
}


def beyond(rng, fn, n):
    """Where e^(-+z^2) is beyond any double or close to it (s_erf-2 as the review found it): a
    coordinate beyond sqrt(MAX) with the other tiny or not, |y^2 - x^2| from 1400 to 1470 (the
    exponent of e^(-z^2) exact there now), and the near-axis strips with a large exponent."""
    sg = lambda: rng.choice([-1, 1])
    lu = lambda a, b: math.exp(rng.uniform(math.log(a), math.log(b)))
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            big, small = lu(1.4e154, 1.7e308), lu(5e-324, 1e-100)
        elif k == 1:
            big = lu(1.4e154, 1.7e308)
            small = lu(1e-300, big * 0.999)
        elif k == 2:
            small = lu(5e-324, 10.0)
            big = math.sqrt(small * small + rng.uniform(1400, 1470))
        else:
            big, small = rng.uniform(26.0, 38.3), lu(5e-324, 1e-10)
        # (x, y) with |y| > |x| for erf, erfc and w below the real axis; erfcx: x = -|y|, y = x
        if fn in ("erf", "erfc"):
            out.append((sg() * small, sg() * big))
        elif fn == "faddeeva":
            out.append((sg() * small, -big))
        else:
            out.append((-big, sg() * small))
    return out


# Added after the review of 2026-11-22: where |v| is just beyond the largest double, a finite part
# next to an infinite one is accurate to the bound times eps|v| (in absolute terms) for erf and
# erfc, far less relative to itself where it is far below |v| (the review's first two arguments of
# each); below the real axis (erfcx: left of the imaginary one) it is right on its own.
FIXED3 = {
    "erf": [(-445.62434747747915, -446.42845923226344), (30.21447217934555, -40.40545301433577)],
    "erfc": [(445.62434747747915, -446.42845923226344), (21.007099575148132, 33.99852962707751)],
    "faddeeva_lhp": [(-63.955726465829976, -69.28680735394832), (8345.218844069563, -8345.261377398252)],
    "erfcx": [(-69.28680735394832, 63.955726465829976), (-8345.261377398252, -8345.218844069563)],
}


def band(rng, fn, n):
    """Where |v| is just beyond the largest double (ln|v| from 709.8 to 713), so that a part can be
    finite next to an infinite one, or both finite with |v| beyond it: |e^(-z^2)| times the factor
    it multiplies, 1/(sqrt(pi)|z|) for erf and erfc (|y| > |x|) and 2 for w below the real axis
    (erfcx: its image left of the imaginary axis); |x| from 1 to 10^4."""
    out = []
    for _ in range(n):
        x = math.exp(rng.uniform(0.0, math.log(1e4)))
        e = 709.8 + rng.uniform(0.0, 3.2)
        y = math.sqrt(x * x + e)
        for _ in range(3):
            f = 2.0 if fn in ("faddeeva", "erfcx") else 1.0 / (math.sqrt(math.pi) * math.hypot(x, y))
            y = math.sqrt(x * x + e - math.log(f))
        sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
        if fn in ("erf", "erfc"):
            out.append((sx * x, sy * y))
        elif fn == "faddeeva":
            out.append((sx * x, -y))
        else:
            out.append((-y, sx * x))
    return out


def complex_c32(rng, fn, n):
    """f32 arguments of the Complex<f32> forms; cases near a zero (scale > 100|v|) and results
    outside the normal range of f32 are left out."""
    if fn == "faddeeva":
        pts = [(x, abs(y)) for x, y in box(rng, -4, 4, -4, 4, n // 2) + polar(rng, 4, 1e6, n // 4)]
        pts += [(x, -abs(y)) for x, y in box(rng, -6, 6, -6, 6, n // 4)]
    else:
        pts = box(rng, -1, 1, -1, 1, n // 4) + box(rng, -6, 6, -6, 6, n // 2) + polar(rng, 4, 1e4, n // 4)
    pts = [(to_f32(x), to_f32(y)) for x, y in pts]
    return complex_rows(fn, pts, keep=lambda v, s: 1.2e-38 < abs(v) < 3.4e38 and s < 100 * abs(v))


def main():
    for single in (False, True):
        for name in REAL:
            rng = random.Random(f"erf {name} {single}")
            xs = gen_real(rng, name, single)
            t = "f32" if single else "f64"
            write(f"erf_{name}_{t}.txt", f"{name} of {t} arguments: the regions of gen_erf.py", real_rows(name, xs))
    # Complex: the well-conditioned sets, then those near zeros and cancellation (lower half-plane).
    rng = random.Random("erf complex")
    up = lambda pts: [(x, abs(y)) for x, y in pts]
    sets = {
        "faddeeva": up(box(rng, -1, 1, -1, 1, 120) + box(rng, -4, 4, -4, 4, 150) + box(rng, -10, 10, -10, 10, 120)
                       + axes(rng, 80) + polar(rng, 10, 1e6, 60)),
        "faddeeva_lhp": [(x, -abs(y)) for x, y in box(rng, -4, 4, -4, 4, 150) + box(rng, -10, 10, -10, 10, 60)],
        "erf": box(rng, -1, 1, -1, 1, 120) + box(rng, -4, 4, -4, 4, 200) + axes(rng, 60) + polar(rng, 4, 1e4, 60),
        "erfc": box(rng, -1, 1, -1, 1, 100) + box(rng, -4, 4, -4, 4, 200) + axes(rng, 60) + polar(rng, 4, 1e4, 60),
        "erfcx": box(rng, -1, 1, -1, 1, 100) + box(rng, -4, 4, -4, 4, 200) + axes(rng, 60) + polar(rng, 4, 1e4, 60),
    }
    # Added with the fixes of 2026-11-21, from their own generators: the diagonals at a large |z|
    # (s_erf-1), the points near zeros, a large |z| everywhere, and the fixed cases above; every
    # value is kept (infinite, subnormal, zero) but these.
    anything = lambda v, s: True
    r2 = random.Random("erf complex fixes")
    sign = lambda pts, sx, sy: [(sx * x, sy * y) for x, y in pts]
    more = {
        "faddeeva": polar(r2, 1e6, 1.7e308, 60, upper=True) + FIXED["faddeeva"],
        "faddeeva_lhp": (sign(diagonal(r2, 80), 1, -1) + sign(diagonal(r2, 40), -1, -1)
                         + near_zeros(r2, "faddeeva", 12) + [(x, -abs(y)) for x, y in polar(r2, 10, 1e6, 40)]
                         + FIXED["faddeeva_lhp"]),
        "erf": (diagonal(r2, 60) + sign(diagonal(r2, 30), -1, -1) + sign(diagonal(r2, 30), 1, -1)
                + near_zeros(r2, "erf", 12) + polar(r2, 1e4, 1.7e308, 40) + FIXED["erf"]),
        "erfc": (diagonal(r2, 60) + sign(diagonal(r2, 30), 1, -1) + sign(diagonal(r2, 30), -1, 1)
                 + near_zeros(r2, "erfc", 12) + polar(r2, 1e4, 1.7e308, 40) + FIXED["erfc"]),
        "erfcx": (sign(diagonal(r2, 60), -1, 1) + sign(diagonal(r2, 30), -1, -1) + diagonal(r2, 20)
                  + near_zeros(r2, "erfcx", 12) + polar(r2, 1e4, 1.7e308, 40) + FIXED["erfcx"]),
    }
    # Added after the review of 2026-11-21, from a third generator.
    r4 = random.Random("erf complex review")
    review = {name: FIXED2[name] + ([] if name == "faddeeva" else beyond(r4, "faddeeva" if name.startswith("faddeeva") else name, 60))
              for name in sets}
    # Added after the review of 2026-11-22, from a fourth generator.
    r6 = random.Random("erf complex review 2")
    review2 = {name: [] if name == "faddeeva" else FIXED3[name] + band(r6, "faddeeva" if name.startswith("faddeeva") else name, 40)
               for name in sets}
    for name, pts in sets.items():
        fn = "faddeeva" if name.startswith("faddeeva") else name
        rows = (complex_rows(fn, pts) + complex_rows(fn, more[name], keep=anything)
                + complex_rows(fn, review[name], keep=anything) + complex_rows(fn, review2[name], keep=anything))
        write(f"erf_{name}_c64.txt", f"{fn} of complex arguments (re im): see gen_erf.py", rows)
    # Next to the axes, where a part is far below |w| (s_erf-4).
    r3 = random.Random("erf complex axes")
    strip = [(7.0, 1e-30), (7.0, 1e-22), (7.0, 1e-19), (7.0, 1e-15), (7.0, 1e-10), (6.72, 1e-6), (9.9, 1e-30),
             (6.7, 1e-30), (-8.5, 1e-200), (6.708, 5e-324)] + axis_strip(r3, 240)
    istrip = [(1e-300, 7.0), (1e-30, 7.0), (1e-20, 7.0), (1e-10, 6.72), (1e-30, 6.7), (1e-30, 10.0),
              (-1e-300, -7.0), (5e-324, 8.0)] + imag_strip(r3, 160)
    # After the review: its worst Re w and the worst of a search (hunt_rew.py), and erf, erfc in all
    # of the strip where they are taken part by part (|x|·max(|y|, 1) < 2^-14; erf from |z| = 1).
    r5 = random.Random("erf complex axes review")
    strip += AXIS_WORST + axis_strip(r5, 90) + STRIP_WORST3["faddeeva"]
    write("erf_faddeeva_axis_c64.txt", "faddeeva next to the real axis (re im): Re w on its own; see gen_erf.py",
          complex_rows("faddeeva", strip, keep=anything, parts=True))
    wide = near_imag(r5, 160)
    extra = {"erf": [(5.514051685907627e-10, -9.051472484477799)] + [p for p in wide if p[0] ** 2 + p[1] ** 2 >= 1]
             + STRIP_WORST3["erf"],
             "erfc": [(5.93267420938868e-22, 7.031199451248784), (2.0194673485008142e-94, -14.684011985237424)] + wide
             + STRIP_WORST3["erfc"]}
    for fn in ("erf", "erfc"):
        write(f"erf_{fn}_axis_c64.txt", f"{fn} next to the imaginary axis (re im): each part on its own; see gen_erf.py",
              complex_rows(fn, istrip + extra[fn], keep=anything, parts=True))
    # w below the real axis next to the imaginary one, part by part (and so erfcx left of the
    # imaginary axis next to the real one, its image).
    neg = ([p for p in FIXED2["faddeeva_lhp"] if abs(p[0]) * max(-p[1], 1) < 2.0 ** -30] + near_neg_imag(r5, 160)
           + STRIP_WORST3["faddeeva_imag"])
    write("erf_faddeeva_imag_c64.txt", "faddeeva below the real axis next to the imaginary one (re im): each part on its own; see gen_erf.py",
          complex_rows("faddeeva", neg, keep=anything, parts=True))
    # Complex<f32>.
    for name in ("faddeeva", "erf", "erfc", "erfcx"):
        rows = complex_c32(random.Random(f"erf complex f32 {name}"), name, 240)
        write(f"erf_{name}_c32.txt", f"{name} of Complex<f32> arguments (re im): see gen_erf.py", rows)


if __name__ == "__main__":
    main()
