"""Reference data of the erf area (issue #20): erf, erfc, erfcx, their inverses, Dawson's
integral, the Fresnel integrals, and the complex erf, erfc, erfcx and Faddeeva function, with
mpmath at 256 bits. Run from this directory: `python3 gen_erf.py`. Deterministic.
"""

import math
import random

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


def w(z):
    return mpmath.exp(-z * z) * mpmath.erfc(-1j * z)


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
    return xs + sp


def complex_rows(name, pts):
    rows = []
    for x, y in pts:
        z = mpmath.mpc(x, y)
        v = {"faddeeva": w, "erf": mpmath.erf, "erfc": mpmath.erfc,
             "erfcx": lambda z: mpmath.exp(z * z) * mpmath.erfc(z)}[name](z)
        v = mpmath.mpc(v)
        # The harness measures relative to |v|: keep |v| in the normal range.
        if not (1e-290 < abs(v) < F64_MAX):
            continue
        rows.append(([x, y], [v]))
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
    for name, pts in sets.items():
        fn = "faddeeva" if name.startswith("faddeeva") else name
        write(f"erf_{name}_c64.txt", f"{fn} of complex arguments (re im): see gen_erf.py", complex_rows(fn, pts))


if __name__ == "__main__":
    main()
