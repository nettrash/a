"""The reference data of the bessel area of issue #20: J, Y, I, K (plain and scaled) of real
orders, the spherical functions and the Airy functions, f64 and f32, with mpmath at 256 bits.

Files `bessel_<func>_<type>.txt`. For `j`, `y`, `sj`, `sy` each case carries two references:
the value, then the modulus `M = √(J² + Y²)` (of the order, at the argument; for the spherical
ones `√(π/(2x))·M`), the scale of the absolute error near the zeros; and so do `i`, `is` at the
negative non-integer orders, with `I_a + (2/π)|sin aπ|·K_a`, `a = −ν`, the scale of the
reflection. Run from this directory: `python3 gen_bessel.py`.

The cases of the regressions of the audit of 2026-11-21 (`regressions`) follow each file's own:
the arguments that exposed a defect and others about them, drawn from their own generator (so
the earlier cases stay as they were); their references are evaluated until two working
precisions agree (`stable`), from Debye's expansion at a high precision for the orders beyond
mpmath's reach (`debye`), and for the Airy functions far out at a precision that grows with the
phase. The cases of the reviews of 2026-11-22 (`regressions_review`, `regressions_review2`) follow
those, likewise.
"""

import math
import random
from fractions import Fraction

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


def ref_j_big(nu, x):
    """`ref_j` (for `x > 0`) of the orders whose power series cancels by more than mpmath's
    default working precision allows (about 0.77ν bits at the turning point)."""
    nu, x = mp_arg(nu), mp_arg(x)
    kw = dict(maxprec=100000, maxterms=10**6)
    j = mpmath.besselj(nu, x, **kw)
    y = mpmath.bessely(nu, x, **kw)
    return [j, mpmath.sqrt(j * j + y * y)]


def reflection_scale(nu, x):
    """`I_a + (2/π)|sin aπ|·K_a`, `a = −ν`, for a negative non-integer `ν` and `x > 0` (else None):
    the scale of the terms of `I_ν = I_a + (2/π)·sin(aπ)·K_a`, which cancel about its zero."""
    if nu >= 0 or nu == mpmath.floor(nu) or x <= 0:
        return None
    a = -nu
    return mpmath.besseli(a, x) + 2 / mpmath.pi * abs(mpmath.sin(a * mpmath.pi)) * mpmath.besselk(a, x)


def ref_i(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    if x < 0:
        return [mpmath.besseli(nu, -x) * (-1) ** int(nu)]
    m = reflection_scale(nu, x)
    v = mpmath.besseli(nu, x)
    return [v] if m is None else [v, m]


def ref_k(nu, x):
    return [mpmath.besselk(mp_arg(nu), mp_arg(x))]


def ref_is(nu, x):
    nu, x = mp_arg(nu), mp_arg(x)
    s = (-1) ** int(nu) if x < 0 else 1
    v = s * mpmath.besseli(nu, abs(x)) * mpmath.exp(-abs(x))
    m = reflection_scale(nu, x)
    return [v] if m is None else [v, m * mpmath.exp(-x)]


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


# ---------------------------------------------------------------- the regressions of 2026-11-21


def stable(f):
    """`f()` at 256 bits and on, doubling until two precisions agree to `2⁻²²⁰` (mpmath's own
    cancellations, as of `besselk` at large orders, show as a disagreement)."""
    prev = None
    for prec in (256, 512, 1024, 2048, 4096):
        with mpmath.workprec(prec):
            v = f()
            if isinstance(v, (list, tuple)):
                v = [+t for t in v]
            else:
                v = [+v]
        if prev is not None:
            with mpmath.workprec(prec):
                if all(a == b or (b != 0 and abs(a / b - 1) < mpmath.mpf(2) ** -220) or
                       (mpmath.isinf(a) and a == b) for a, b in zip(prev, v)):
                    return v
        prev = v
    raise ValueError("no agreement")


def _debye_u():
    """Debye's polynomials U_0..U_12 as {power: Fraction} (DLMF 10.41.10)."""
    def mul(a, b):
        r = {}
        for e1, c1 in a.items():
            for e2, c2 in b.items():
                r[e1 + e2] = r.get(e1 + e2, 0) + c1 * c2
        return r
    us = [{0: Fraction(1)}]
    for _ in range(12):
        u = us[-1]
        d = mul({2: Fraction(1, 2), 4: Fraction(-1, 2)}, {e - 1: c * e for e, c in u.items() if e})
        g = {e + 1: c / (e + 1) for e, c in mul({0: Fraction(1), 2: Fraction(-5)}, u).items()}
        for e, c in g.items():
            d[e] = d.get(e, 0) + c / 8
        us.append({e: c for e, c in d.items() if c != 0})
    return us


DEBYE_U = _debye_u()


def debye(kind, nu, x, scaled=False):
    """`I_ν(x)` (`kind` "i") or `K_ν(x)` by Debye's expansion to `U₁₂` at `200 + 2·log₂ max(ν, x)`
    bits (the exponent `νη` needs its absolute error far below 1), times `e^(∓x)` when `scaled`:
    the reference of the orders beyond mpmath's series (from `10⁴` on the next term is below
    `10⁻⁴⁰` of the value)."""
    prec = 200 + 2 * max(1, int(math.log2(max(nu, x))))
    with mpmath.workprec(prec):
        nu, x = mpmath.mpf(nu), mpmath.mpf(x)
        r = mpmath.sqrt(nu * nu + x * x)
        p = nu / r
        e = r + nu * mpmath.log(x / (nu + r))
        if scaled:
            e = e - x
        s = mpmath.mpf(0)
        for k, u in enumerate(DEBYE_U):
            uk = sum(mpmath.mpf(c.numerator) / c.denominator * p ** j for j, c in u.items())
            s += (uk if kind == "i" else (-1) ** k * uk) / nu ** k
        if kind == "i":
            v = mpmath.exp(e) / mpmath.sqrt(2 * mpmath.pi * r) * s
        else:
            v = mpmath.sqrt(mpmath.pi / (2 * r)) * mpmath.exp(-e) * s
        return +v


def airy_far(x, d, b):
    """`airy_ref` far on the negative axis, at a working precision that grows with the phase
    `ζ = (2/3)|x|^(3/2)` (256 bits beyond its integer part)."""
    prec = 256 + max(0, int(1.5 * math.log2(abs(x))))
    with mpmath.workprec(prec):
        return [+v for v in airy_ref(x, d, b)]


Z0 = None


def z0():
    """The root `z₀ = 0.6627…` of Debye's `η(z) = √(1 + z²) + ln(z/(1 + √(1 + z²)))` as a
    Fraction (to 1200 bits)."""
    global Z0
    if Z0 is None:
        with mpmath.workprec(1300):
            eta = lambda z: mpmath.sqrt(1 + z * z) + mpmath.log(z / (1 + mpmath.sqrt(1 + z * z)))
            r = mpmath.findroot(eta, mpmath.mpf("0.66274341934918158"))
            Z0 = Fraction(int(mpmath.floor(r * mpmath.mpf(2) ** 1200)), 2 ** 1200)
    return Z0


Z1 = None


def z1():
    """The root `z₁ = 0.4477…` of `η(z) + z` (`η` as in `z0`): about `x = z₁ν` the term
    `e^(−x)·K_ν(x)` of the scaled reflection of `I` at the order `−ν` is within the range."""
    global Z1
    if Z1 is None:
        with mpmath.workprec(200):
            eta = lambda z: mpmath.sqrt(1 + z * z) + mpmath.log(z / (1 + mpmath.sqrt(1 + z * z)))
            Z1 = mpmath.findroot(lambda z: eta(z) + z, mpmath.mpf("0.448"))
    return Z1


def is_reflection_debye(nu, x):
    """`e^(−x)·I_ν(x)` at a negative non-integer order `ν` of 1000 and more by the reflection
    `I_a + (2/π)·sin(aπ)·K_a`, `a = −ν`, both terms by `debye` (from `ν = 10³` on its next term is
    below `10⁻³⁹` of the value), then the scale of its terms (as `ref_is`)."""
    a = -nu
    with mpmath.workprec(200 + 2 * max(1, int(math.log2(max(a, x))))):
        e = mpmath.exp(-mpmath.mpf(x))
        i = debye("i", a, x, scaled=True)
        k = debye("k", a, x) * e
        s = mpmath.sin(mpmath.mpf(a) * mpmath.pi)
        c = 2 / mpmath.pi
        return [+(i + c * s * k), +(i + c * abs(s) * k)]


def f32s(v):
    return to_f32(v)


def zero_i_neg(a):
    """The zero of `I_(−a)` on `(0, 2a + 10)` for `a` in `(2m − 1, 2m)`, by a scan for the change
    of sign and bisection at 256 bits."""
    with mpmath.workprec(256):
        f = lambda t: mpmath.besseli(-a, t)
        lo = mpmath.mpf(1) / 64
        flo = f(lo)
        step = mpmath.mpf(1) / 64
        while True:
            hi = lo + step
            fhi = f(hi)
            if flo * fhi < 0:
                break
            lo, flo = hi, fhi
            if lo > 2 * a + 10:
                raise ValueError("no zero")
        for _ in range(200):
            mid = (lo + hi) / 2
            fm = f(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2


def regressions(name, single):
    """The cases `(args, refs)` that exposed the defects of the audit of 2026-11-21 in `name`
    (and others about them), with their references."""
    g = random.Random(20261121 + sum(map(ord, name)) + (7 if single else 0))
    cv = f32s if single else (lambda v: v)
    out = []
    lu = lambda a, b: math.exp(g.uniform(math.log(a), math.log(b)))

    def add(f, args):
        if single and any(abs(a) > 3.4e38 for a in args):
            return
        args = tuple(float(a) if isinstance(a, int) else cv(a) for a in args)
        out.append((args, stable(lambda: f(*args))))

    if name == "j":
        # J at the turning point (CF1 + Wronskian, x ≤ ν), at negative orders, and orders 0, 1.
        for nu, x in [(198.42046227010448, 198.42046227010448), (167.79054479453902, 167.716176233631),
                      (263.8657516185832, 263.2687179735704), (188.0, 187.54731986477245),
                      (242.93967463268035, 242.93967463268035), (297.0, 297.0),
                      (-56.00001068717319, 55.63786332239022), (-299.93048126012445, 298.37480785192275),
                      (-7.498042132965493, 1.298694258543044), (-17.689180231537414, 1.8977927229590101),
                      (-277.93974226875287, 15.690829707042946), (-287.0616857301436, 17.56951322167215),
                      (-229.00000000160273, 6.986773121870236), (1.0, 15.9275099295195)]:
            add(ref_j, (nu, x))
        for _ in range(4 if single else 40):
            nu = lu(50, 500)
            add(ref_j, (nu, nu * (1 + g.uniform(-2, 0.2) * nu ** (-2.0 / 3.0))))
        for _ in range(2 if single else 15):
            add(ref_j, (-(g.randint(2, 30) + g.uniform(0.45, 0.5)), g.uniform(0.3, 2.0)))
        for _ in range(2 if single else 20):
            add(ref_j, (1.0, g.uniform(10, 24)))
        # The turning point of larger orders, where the old error grew with ν (66 ulps of M at
        # 10³, 951 ulps of CF1's ratio at 2843).
        for nu, x in [(1000.0, 1000.0), (1000.0, 995.0), (2843.01, 2843.01), (2843.01, 2835.0)]:
            add(ref_j_big, (nu, x))
        for _ in range(1 if single else 4):
            nu = g.uniform(1000, 4000)
            add(ref_j_big, (nu, nu * (1 + g.uniform(-2, 0.2) * nu ** (-2.0 / 3.0))))
    elif name == "y":
        # Y in Temme's region (μ near ±1/2), at negative orders, and the order 0 below 24.
        for nu, x in [(7.478705843856315, 1.9096396396680124), (7.498042132965493, 1.298694258543044),
                      (17.48509524208048, 1.822712642232716), (3.488297416753122, 1.9828498727353754),
                      (26.57792635867805, 1.892720686179843), (-231.84298374018013, 7.88471649590445),
                      (0.0, 15.960183709153872), (0.0, 12.820811555411364)]:
            add(ref_y, (nu, x))
        for _ in range(4 if single else 40):
            mu = g.uniform(0.45, 0.5) * g.choice([-1, 1])
            add(ref_y, (g.randint(1, 30) + mu, g.uniform(0.3, 2.0)))
        for _ in range(2 if single else 20):
            add(ref_y, (0.0, g.uniform(10, 24)))
    elif name in ("i", "is"):
        f = ref_i if name == "i" else ref_is
        # Negative non-integer orders about the zero of I_ν (the reflection cancels), their
        # overflow edge, and the orders up to −300.
        cases = [(-1.1094411281323366, 0.6255270242630361), (-5.775809085289805, 3.8959337248979695),
                 (-1.7484091466884095, 1.2577953269632478), (-61.86156842008641, 0.19559246038540856),
                 (-134.09577780580432, 53.687984311144895), (-279.08448558930246, 15.48715033854455),
                 (-270.05283293138064, 13.585152601221168), (-259.9999786559247, 11.901606112636651)]
        for nu, x in cases:
            add(f, (nu, x))
        for a in (1.1094411281323366, 1.75, 3.3, 5.775809085289805, 9.6, 15.2, 25.5, 39.7):
            x0 = zero_i_neg(a)
            for rel in (-1e-3, -3e-6, 3e-6, 1e-3, 0.03):
                add(f, (-a, float(x0 * (1 + rel))))
        for _ in range(2 if single else 10):
            add(f, (-g.uniform(50, 300), lu(0.1, 100)))
        if name == "is":
            # Hankel's former branch for ν ≥ 50, and x at the top of the range.
            for nu, x in [(121.0, 9435.513583819933), (58.26984711067225, 2037.2250493806714),
                          (121.4064565944238, 9576.796377323044), (44.17951191884695, 7.843151846169141e307),
                          (31.903985698962728, 1.4990873830527484e308), (2.5, 1.7976931348623157e308)]:
                add(f, (nu, x))
            for _ in range(4 if single else 30):
                nu = g.uniform(50, 300)
                add(f, (nu, 0.6 * nu * nu * g.uniform(1, 4)))
            for _ in range(2 if single else 10):
                add(f, (g.uniform(2, 49), lu(1e300, 1.7e308)))
            if not single:
                # The scaled reflection at large negative orders about x = z₁a, the only place
                # where e^(−x)·K_a is within the range, and either side of it (∞, then 0).
                for a in (1000.5, 5000.3, 65535.5, 100000.5):
                    xb = float(z1() * a)
                    for d in (-20, -5, 0, 5, 20):
                        x = xb + d / (1 + 2.2)
                        out.append(((-a, x), is_reflection_debye(-a, x)))
                    for z in (0.4, 0.6):
                        out.append(((-a, z * a), is_reflection_debye(-a, z * a)))
    elif name in ("sj", "sy"):
        f = ref_sj if name == "sj" else ref_sy
        if name == "sj":
            for n, x in [(10, 20.46963057069963), (12, 24.84679075699926), (550, 549.3427161079844)]:
                add(f, (n, x))
            for _ in range(4 if single else 40):
                n = g.randint(4, 12)
                add(f, (n, g.uniform(2 * n, 0.5 * n * (n + 1))))
            for _ in range(2 if single else 15):
                n = g.randint(100, 1500)
                add(f, (n, n * g.uniform(0.98, 1.0)))
        else:
            for _ in range(4 if single else 30):
                n = g.randint(2, 4)
                add(f, (n, g.uniform(n, 0.5 * n * (n + 1))))
            if not single:
                # y_n at tiny x: −∞ where it overflows, finite just above.
                for n, x in [(0, 1.92292464353327e-310), (1, 7.0e-155), (2, 2.0e-103),
                             (12, 3.7516323695037054e-187), (0, 6.0e-309), (1, 7.5e-155),
                             (2, 2.6e-103), (1, 2.0e-151)]:
                    add(f, (n, x))
    elif name == "si":
        cases = [(10, 37.66115967953328), (44, 83.74903644760914), (970, 517.7423795885412),
                 (951, 524.5549309075323), (1712, 980.1541774563506), (100, 720.0000000000001),
                 (23312, 10568.118521549368)]
        if single:
            cases = [(1091, 681.6755981445313), (1091, 743.9862060546875)]
        for n, x in cases:
            add(ref_si, (n, x))
        for _ in range(4 if single else 40):
            n = g.randint(30, 3000)
            add(ref_si, (n, g.uniform(20 + n / 2, min(0.5 * n * (n + 1), 1500.0))))
        for _ in range(2 if single else 20):
            n = g.randint(7, 49)
            add(ref_si, (n, g.uniform(20 + n / 2, 0.5 * n * (n + 1))))
        if not single:
            # Orders above 10⁷ (Debye's I in constant time) about x = z₀(n + 1/2), where in range.
            for _ in range(10):
                n = g.randint(10_000_001, 4_000_000_000)
                x = float(Fraction(n) * z0() + Fraction(1, 2) * z0() + Fraction(g.uniform(-500, 500)))
                v = debye("i", n + 0.5, x) * mpmath.sqrt(mpmath.pi / (2 * mpmath.mpf(x)))
                out.append(((float(n), x), [v]))
    elif name == "ks":
        # Hankel's former branch for ν ≥ 50 (x ≥ 0.6ν²), Debye's expansion now.
        add(ref_ks, (276.48848009974523, 47476.65908253519))
        for _ in range(4 if single else 30):
            nu = g.uniform(50, 300)
            add(ref_ks, (nu, 0.6 * nu * nu * g.uniform(1, 4)))
    elif name == "sk":
        if not single:
            for n, x in [(0, 2.74826685869973e-310), (1, 9.0e-155), (57, 3.4870340111965286e-285),
                         (0, 9.0e-309), (1, 1.3e-154), (1, 2.0e-151)]:
                add(ref_sk, (n, x))
    elif name in ("ai", "aip", "bi", "bip"):
        d, b = int(name.endswith("p")), name.startswith("b")
        fa = lambda x: airy_ref(x, d, b)
        if name == "bip":
            for x in (-0.17609077357260405, -0.2472783109455814):
                add(fa, (x,))
            for _ in range(4 if single else 40):
                add(fa, (g.uniform(-1.2, -0.05),))
        if name == "aip":
            for x in (-9.001175304519322, -9.001679052653774):
                add(fa, (x,))
            for _ in range(4 if single else 40):
                add(fa, (g.uniform(-9.1, -8.4),))
        # Far on the negative axis: the value within the modulus (the phase is lost there).
        far = [-6.419482215080373e21, -1.0e23, -5.613731440517933e37, -9.047088488815489e40, -1.0e100]
        if single:
            far = [-6.209673445466199e27, -7.941560825561911e24]
        for x in far:
            x = cv(x)
            out.append(((x,), airy_far(x, d, b)))
        for _ in range(4 if single else 20):
            x = cv(-lu(1e9, 1e38 if single else 1e200))
            out.append(((x,), airy_far(x, d, b)))
    return out


def regressions_huge(name, single):
    """Debye's I, K at orders beyond mpmath's series: plain about x = z₀ν (where they are in
    range), with x chosen within 600 of z₀ν; scaled for x ≫ ν (f32: the scaled forms only, at
    the arguments rounded to f32). References by `debye`."""
    g = random.Random(20261122 + sum(map(ord, name)) + (7 if single else 0))
    out = []
    lu = lambda a, b: math.exp(g.uniform(math.log(a), math.log(b)))
    if single:
        if name in ("is", "ks"):
            cases = [(889990506610688.0, 1.0155131778383523e29)]
            for _ in range(10):
                nu = to_f32(lu(1e4, 1e18))
                x = to_f32(nu * nu / lu(1 / 50, 80))
                if x < 3.4e38:
                    cases.append((nu, x))
            for nu, x in cases:
                nu, x = to_f32(nu), to_f32(x)
                out.append(((nu, x), [debye(name[0], nu, x, scaled=True)]))
        return out
    if name in ("i", "k"):
        cases = [(889990503241027.4, 589835349306017.2)]
        for _ in range(40):
            nu = lu(1e4, 4e18)
            x = float(Fraction(nu) * z0() + Fraction(g.uniform(-600, 600)))
            cases.append((nu, x))
        for nu, x in cases:
            out.append(((nu, x), [debye(name, nu, x)]))
    else:
        kind = name[0]
        cases = [(5.650345951006096e37, 2.647916945341862e72), (889990503241027.4, 1.0155131731998175e29),
                 (798114799932366.6, 1.5682455084283833e29)]
        for _ in range(30):
            nu = lu(1e4, 1e150)
            x = nu * nu / lu(1 / 50, 1300)
            if x < 1.7e308:
                cases.append((nu, x))
        for _ in range(10):
            nu = lu(1e4, 1e15)
            cases.append((nu, nu * lu(0.5, 20)))
        for nu, x in cases:
            out.append(((nu, x), [debye(kind, nu, x, scaled=True)]))
    return out


def regressions_review(name, single):
    """The cases of the review of 2026-11-22: `I_n` and `K_n` of the integer orders 2 to 49 where
    `e^(−x)` leaves the range (`K₀`, `K₁` subnormal from `x ≈ 705.5`, and `I_n` from its Wronskian
    with `K_n` up to Hankel's region `x = 0.6n²`: `∞` beyond its overflow at about 715), and the
    scaled Debye forms at the top of the range (`x` from `5·10³⁰⁷`, `ν` from `10¹⁴⁶` on, where
    `(ν/x)²` is subnormal). References by `stable` and, for the huge orders, by `debye`."""
    g = random.Random(20261123 + sum(map(ord, name)) + (7 if single else 0))
    cv = f32s if single else (lambda v: v)
    out = []
    lu = lambda a, b: math.exp(g.uniform(math.log(a), math.log(b)))

    def add(f, args):
        args = tuple(float(a) if isinstance(a, int) else cv(a) for a in args)
        out.append((args, stable(lambda: f(*args))))

    if name == "i":
        if single:
            # I overflows f32 long before; beyond x = 742 it was NaN.
            add(ref_i, (46, 1059.405))
            for _ in range(6):
                n = g.randint(36, 49) * g.choice([-1, 1])
                add(ref_i, (n, g.uniform(745, 0.6 * n * n)))
            return out
        for n, x in [(39, 715.0), (40, 713.0), (43, 705.5), (39, 826.4037582841073),
                     (-39, 826.4037582841073)]:
            add(ref_i, (n, x))
        for _ in range(16):
            add(ref_i, (g.randint(35, 49) * g.choice([-1, 1]), g.uniform(700, 714)))
        for _ in range(8):
            n = g.randint(35, 49) * g.choice([-1, 1])
            add(ref_i, (n, g.uniform(716, 0.6 * n * n)))
    elif name == "k" and not single:
        # K_n near its underflow (subnormal results).
        add(ref_k, (49, 740.0))
        for _ in range(20):
            add(ref_k, (g.randint(2, 49), g.uniform(700, 745)))
    elif name in ("is", "ks") and not single:
        kind = name[0]
        cases = {"is": [(9.022384834968027e149, 1.64819653969638e308),
                        (1.1109232872259081e153, 1.1685702408781258e308)],
                 "ks": [(5.653783006465378e153, 9.867269418738659e307)]}[name]
        cases += [(1.0e150, 1.7976931348623157e308), (3.0e154, 1.7976931348623157e308)]
        for _ in range(24):
            cases.append((lu(1e146, 4e154), lu(5e307, 1.7976931348623157e308)))
        for _ in range(6):
            # Up to the order where the value leaves the normal range: e^(∓ν²/(2x))/√(2πx) and
            # √(π/(2x))·e^(ν²/(2x)), ν²/(2x) up to about 350 (I) and 700 (K).
            x = lu(5e307, 1.7976931348623157e308)
            cases.append((lu(4e154, math.sqrt(700 if kind == "i" else 1400) * math.sqrt(x)), x))
        for nu, x in cases:
            out.append(((nu, x), [debye(kind, nu, x, scaled=True)]))
    return out


def regressions_review2(name, single):
    """The cases of the second review of 2026-11-22 (f64 only): `jₙ` and `J_ν` below their turning
    points (`x` from `ν/20` to `ν/2`, above the power series), where `M` is the size of `Y` and
    the error is measured in ulps of the value; `Y₁` about `x = 3.75`, where the term `g/x` of its
    form about its first zeros had no error term without a fused multiply-add; the scaled `I` at
    negative non-integer orders from `2¹⁶` to `4.4·10¹⁵` about `x = z₁a` (`a = −ν`), and either side
    of that band (`±∞`, then 0). References by `stable`, the last by `is_reflection_debye`."""
    out = []
    if single:
        return out
    g = random.Random(20261124 + sum(map(ord, name)))
    lu = lambda a, b: math.exp(g.uniform(math.log(a), math.log(b)))

    def add(f, args):
        args = tuple(float(a) for a in args)
        out.append((args, stable(lambda: f(*args))))

    if name == "sj":
        for n, x in [(29, 10.82000673849002), (112, 43.04742460079839), (220, 94.12078164038805)]:
            add(ref_sj, (n, x))
        while len(out) < 45:
            n = g.randint(2, 1000)
            x = n * g.uniform(0.05, 0.5)
            if x * x > n + 1.5:
                add(ref_sj, (n, x))
    elif name == "j":
        while len(out) < 30:
            nu = g.choice([float(g.randint(2, 1000)), lu(2, 1000)])
            x = nu * g.uniform(0.05, 0.5)
            if x * x > nu + 1:
                add(ref_j, (nu, x))
    elif name == "y":
        for x in (3.727922896042429, 3.7613239609299063):
            add(ref_y, (1.0, x))
        for _ in range(20):
            add(ref_y, (1.0, g.uniform(3.70, 3.80)))
    elif name == "is":
        z = z1()
        for _ in range(36):
            while True:
                a = float(math.floor(lu(65536, 4.4e15)) + g.choice([0.5, 0.25, 0.75, 0.1, g.random()]))
                if a != math.floor(a):
                    break
            x = float(mpmath.mpf(a) * z + g.uniform(-250, 250))
            out.append(((-a, x), is_reflection_debye(-a, x)))
        # The second review's points (+∞ at 3.9·10⁷ where the value is 9·10⁻²¹²; 1.82 ulps at
        # 1.65·10⁷), and either side of the band.
        cases = [(38732692.75, 17342438.238260992), (16543289.1, 7407301.514842674)]
        for a in (70000.5, 3.3e7 + 0.25, 1.0e12 + 0.5):
            for d in (-1000.0, 1000.0):
                cases.append((a, float(mpmath.mpf(a) * z + d)))
            for zz in (0.3, 0.6):
                cases.append((a, zz * a))
        for a, x in cases:
            out.append(((-a, x), is_reflection_debye(-a, x)))
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
            rows += regressions(name, single) + regressions_review2(name, single)
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
            rows += regressions(name, single) + regressions_huge(name, single)
            rows += regressions_review(name, single) + regressions_review2(name, single)
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
            rows = [((n, x), ref(n, x)) for n, x in rows] + regressions(name, single)
            rows += regressions_review2(name, single)
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
            rows = [((x,), ref(x)) for x in xs] + regressions(name, single)
            write(f"bessel_{name}_{t}.txt", f"airy_{name}(x), {t}: the value, then the modulus", rows)


if __name__ == "__main__":
    gen_jy()
    gen_ik()
    gen_sph()
    gen_airy()
