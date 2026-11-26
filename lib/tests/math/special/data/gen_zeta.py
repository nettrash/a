"""The zeta area's reference data (mpmath, 256 bits): the Riemann zeta function (real and complex),
the Hurwitz zeta function, the dilogarithm and the polylogarithm, both real branches of the
Lambert W function and the exponential integrals E1, Ei and En.

Run from this directory: `python3 gen_zeta.py [only…]` (names of the files' functions to make
only those). `ZETA_SCALE=k ZETA_OUT=dir` multiplies the counts by `k` and writes to `dir` (for the
larger checks outside the repository).

Each file ends with the regression sets of the audit of 2026-11-21 (defects s_zeta-1 to 23), and
some with those of the first and second reviews of the fixes (2026-11-22): the arguments that exposed a defect
and points around them, with their own seeds, so that the rows before them are unchanged. Their
references avoid the methods the audit found unreliable: the
polylogarithm of a negative integer order by exact rational arithmetic, of other negative orders
by the Hurwitz zeta function of a complex argument (x < 0) or Jonquière's formula (0 < x < 1),
huge orders by their series, the complex zeta function past sigma = 20 by its Dirichlet series.
"""

import math
import os
import random
import sys
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "a-special-zeta-pycache")

import mpmath  # noqa: E402

from gen_common import F64_MAX, log_uniform, mp_arg, to_f32, uniform, write

SCALE = int(os.environ.get("ZETA_SCALE", "1"))
OUT = os.environ.get("ZETA_OUT", ".")
ONLY = set(sys.argv[1:])


def n(k):
    return k * SCALE


def out(name, header, rows):
    write(os.path.join(OUT, "zeta_" + name + ".txt"), header, rows)


def wants(*names):
    return not ONLY or any(x in ONLY for x in names)


F32_MAX = 3.4028234663852886e38


def f32s(xs):
    """The f32 values of the floats `xs` within the f32 range."""
    return [to_f32(x) for x in xs if abs(x) <= F32_MAX]


INV_E = -1 / mpmath.e
MINUS_INV_E_F64 = float(INV_E)
MINUS_INV_E_F32 = to_f32(MINUS_INV_E_F64)


def lambert(x, k):
    if x in (MINUS_INV_E_F64, MINUS_INV_E_F32):
        return mpmath.mpf(-1)
    v = mp_arg(x)
    if k == -1 and v == 0:
        return -mpmath.inf
    if v < INV_E or (k == -1 and v > 0):
        return mpmath.nan
    return mpmath.lambertw(v, k).real


def lambert_args(rng, k, count, single):
    """Arguments of W_k: near the branch point (p = √(2(1 + e x)) log-uniform), the middle, and
    large (W0) or tiny negative (W-1) ones."""
    xs = []
    ps = log_uniform(rng, 1e-9, 0.7, count // 4)
    xs += [float((mpmath.mpf(p) ** 2 / 2 - 1) / mpmath.e) for p in ps]
    if k == 0:
        xs += uniform(rng, -0.36, 3.0, count // 4)
        xs += log_uniform(rng, 1e-300, 1e-3, count // 8)
        xs += [-x for x in log_uniform(rng, 1e-300, 0.3, count // 8)]
        xs += log_uniform(rng, 3.0, 1e308, count // 4)
    else:
        xs += uniform(rng, -0.36, -0.01, count // 4)
        xs += [-x for x in log_uniform(rng, 1e-320, 1e-2, count // 2)]
    if single:
        xs = f32s(xs)
        xs = [x for x in xs if x >= INV_E]
    return xs


def gen_lambert():
    for k, name in [(0, "w0"), (-1, "wm1")]:
        for single, ty in [(False, "f64"), (True, "f32")]:
            rng = random.Random(100 + k + (7 if single else 0))
            xs = lambert_args(rng, k, n(600 if not single else 300), single)
            if k == 0:
                xs += [MINUS_INV_E_F64, MINUS_INV_E_F32, 0.0, -0.0, 1.0, math.e]
                xs += [] if single else [1e308]
            else:
                xs += [MINUS_INV_E_F64, MINUS_INV_E_F32, -1e-300, -5e-324]
            if single:
                xs = sorted(set(f32s(xs)))
            # regression (s_zeta-12): the pieces near the branch point, p ≤ 0.66 for W0 and
            # x ≤ −0.2 for W−1, where c₁t and s were rounded
            rr = random.Random(150 + k + (7 if single else 0))
            if k == 0:
                extra = [-0.35213442446884474, -0.2891798588814022, -0.3542219199038835]
                extra += [float((mpmath.mpf(p) ** 2 / 2 - 1) / mpmath.e) for p in uniform(rr, 0.2, 0.66, n(120))]
            else:
                extra = [-0.20606477410531593, -0.2739251904605389]
                extra += uniform(rr, -0.3, -0.2, n(120))
            extra = f32s(extra) if single else extra
            rows = [((x,), [lambert(x, k)]) for x in xs]
            rows += [((x,), [lambert(x, k)]) for x in extra if x >= INV_E]
            out(f"lambert{name}_{ty}", f"Lambert W{'0' if k == 0 else '-1'}(x), {ty}", rows)


if wants("lambertw0", "lambertwm1", "lambert"):
    gen_lambert()


def zeta_real(x):
    v = mp_arg(x)
    if v == 1:
        return mpmath.inf
    if mpmath.isinf(v):
        return mpmath.mpf(1) if v > 0 else mpmath.nan
    return mpmath.zeta(v)


def zeta_args(rng, count, single):
    xs = []
    xs += [1 + d for d in log_uniform(rng, 1e-15, 2.0, count // 8)]
    xs += [1 - d for d in log_uniform(rng, 1e-15, 1.0, count // 10)]
    xs += uniform(rng, 3.0, 60.0, count // 8)
    xs += uniform(rng, -0.5, 3.0, count // 8)
    xs += uniform(rng, -20.0, -0.5, count // 8)
    xs += uniform(rng, -170.0, -20.0, count // 10)
    xs += uniform(rng, -270.0, -170.0, count // 20)
    # near the trivial zeros
    for _ in range(count // 10):
        k = rng.randint(1, 60)
        xs.append(-2 * k + rng.choice([-1, 1]) * math.exp(rng.uniform(math.log(1e-14), math.log(0.1))))
    xs += log_uniform(rng, 60.0, 1e300, count // 40)
    xs += [-x for x in log_uniform(rng, 280.0, 1e300, count // 40)]
    if single:
        xs = f32s(xs)
    return xs


def gen_zeta_real():
    for single, ty in [(False, "f64"), (True, "f32")]:
        rng = random.Random(200 + single)
        xs = zeta_args(rng, n(800 if not single else 500), single)
        xs += [0.0, -0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, 0.5, -0.5, 56.0, 1e308 if not single else 1e38]
        if single:
            xs = sorted(set(f32s(xs)))
        # regression (s_zeta-7): [−1/2, 0), where 1/(s − 1) + R(s) cancels by 3, and the
        # reflection past 1 − s = 171 (the duplication formula before)
        rr = random.Random(250 + single)
        extra = [-0.46592206046310214, -0.42614200244195655, -0.45565129015029016, -228.35535356976317]
        extra += uniform(rr, -0.5, 0.0, n(150)) + uniform(rr, -280.0, -170.0, n(60))
        if single:
            extra = f32s(extra)
        rows = [((x,), [zeta_real(x)]) for x in xs + extra]
        out(f"zeta_{ty}", f"Riemann zeta(s), real s, {ty}", rows)


if wants("zeta"):
    gen_zeta_real()


EI_X0 = mpmath.findroot(mpmath.ei, 0.37)


def e1_ref(x):
    v = mp_arg(x)
    if v < 0 or mpmath.isnan(v):
        return mpmath.nan
    if v == 0:
        return mpmath.inf
    return mpmath.e1(v)


def ei_ref(x):
    v = mp_arg(x)
    if v == 0:
        return -mpmath.inf
    return mpmath.ei(v)


def en_ref(k, x):
    v = mp_arg(x)
    if v < 0:
        return mpmath.nan
    if v == 0:
        return mpmath.inf if k <= 1 else mpmath.mpf(1) / (k - 1)
    # mpmath's expint cancels at 256 bits for large n and x: x^(n-1) Γ(1 - n, x) at 1024.
    with mpmath.workprec(1024):
        r = mpmath.gammainc(1 - k, v) * v ** (k - 1)
    return +r


def gen_expint():
    for single, ty in [(False, "f64"), (True, "f32")]:
        c = 1 if not single else 0.6
        rng = random.Random(300 + single)
        xs = log_uniform(rng, 1e-300 if not single else 1e-40, 1.0, n(int(150 * c)))
        xs += uniform(rng, 1.0, 10.0, n(int(150 * c)))
        xs += log_uniform(rng, 10.0, 746.0, n(int(150 * c)))
        xs += [0.0, 1.0, 2.0, 700.0, 745.0, 746.0, 1e10]
        if single:
            xs = sorted(set(f32s(xs)))
        # regression (s_zeta-21, -22): (1/2, 1], where F(x) − ln x cancels by 4, the left end of
        # the rational on (1, 2.5], and Boost's rational past 2.5
        rr = random.Random(305 + single)
        extra = [0.9513045267373101, 0.988406775477793, 0.8356864752387235, 0.5906746809370671]
        extra += [1.0061049075258708, 2.912476427726344, 2.5650136142422393]
        extra += uniform(rr, 0.5, 1.0, n(int(150 * c))) + uniform(rr, 1.0, 1.2, n(int(40 * c)))
        extra += uniform(rr, 2.5, 4.0, n(int(60 * c)))
        xs += f32s(extra) if single else extra
        out(f"e1_{ty}", f"E1(x), {ty}", [((x,), [e1_ref(x)]) for x in xs])
        rng = random.Random(310 + single)
        xs = log_uniform(rng, 1e-300 if not single else 1e-40, 6.0, n(int(150 * c)))
        xs += [float(EI_X0 + d) for d in uniform(rng, -0.05, 0.05, n(int(60 * c)))]
        xs += [float(EI_X0) * (1 + d) for d in log_uniform(rng, 1e-16, 1e-3, n(int(40 * c)))]
        xs += [float(EI_X0) * (1 - d) for d in log_uniform(rng, 1e-16, 1e-3, n(int(40 * c)))]
        xs += uniform(rng, 6.0, 50.0, n(int(150 * c)))
        xs += uniform(rng, 50.0, 717.0, n(int(80 * c)))
        xs += [-x for x in log_uniform(rng, 1e-300 if not single else 1e-40, 746.0, n(int(80 * c)))]
        xs += [0.0, float(EI_X0), 6.0, 10.0, 20.0, 48.0, 709.0, 716.0, 717.0, 1e10]
        if single:
            xs = sorted(set(f32s(xs)))
        # regression (s_zeta-21, -22): Ei below x₀ − 0.1 and on [4.5, 6] (the rounded ln and R),
        # and −E₁(−x) on [−1, −1/2)
        rr = random.Random(315 + single)
        extra = [0.270472298656838, 0.26047213412092807, 4.597758146111274, -0.9981899953583173, -0.988406775477793]
        extra += uniform(rr, 0.1, 0.27, n(int(60 * c))) + uniform(rr, 0.47, 1.5, n(int(60 * c)))
        extra += uniform(rr, 4.5, 6.0, n(int(40 * c))) + uniform(rr, -1.0, -0.5, n(int(60 * c)))
        extra += [-x for x in uniform(rr, 2.5, 4.0, n(int(40 * c)))] + [-x for x in log_uniform(rr, 4.0, 700.0, n(int(30 * c)))]
        xs += f32s(extra) if single else extra
        # review round 1 of the fixes: x − x₀ rounded far from x₀ (1.5 ulps on [4.5, 4.75])
        rr = random.Random(318 + single)
        extra = [4.54621112948109, 4.534823789348766, 4.520135107645411]
        extra += uniform(rr, 4.5, 4.75, n(int(60 * c))) + uniform(rr, 1.5, 6.0, n(int(40 * c)))
        xs += f32s(extra) if single else extra
        # review round 2 of the fixes: past 6 the rationals in u = (x − c)/h, whose polynomials
        # alternate and cancel at the left end of each piece (by up to 58 and 95: 2.13 ulps at
        # x = 10.29, 1.70 at 30.009); the left ends of the four pieces and their switch points
        rr = random.Random(330 + single)
        extra = [10.291004169856471, 10.200679756196376, 10.008133437940838, 30.39794466765631,
                 10.000000000000167, 30.00866057064067, 10.070128483640373, 30.33569548827923]
        extra += uniform(rr, 10.0, 10.6, n(int(80 * c))) + uniform(rr, 30.0, 30.5, n(int(40 * c)))
        extra += uniform(rr, 6.0, 6.5, n(int(20 * c))) + uniform(rr, 20.0, 20.5, n(int(20 * c)))
        for e in (6.0, 10.0, 20.0, 30.0, 48.0):
            extra += [math.nextafter(e, 0.0), math.nextafter(e, 100.0)]
        xs += f32s(extra) if single else extra
        out(f"ei_{ty}", f"Ei(x), {ty}", [((x,), [ei_ref(x)]) for x in xs])
        rng = random.Random(320 + single)
        rows = []
        for _ in range(n(int(500 * c))):
            k = rng.choice([0, 1, 2, 2, 3, 3, 4, 5, 7, 10, 15, 16, 20, 30, 50, 100, 1000])
            r = rng.random()
            if r < 0.4:
                x = math.exp(rng.uniform(math.log(1e-20 if not single else 1e-30), 0.0))
            elif r < 0.8:
                x = rng.uniform(0.0, 20.0)
            else:
                x = math.exp(rng.uniform(math.log(20.0), math.log(746.0)))
            if single:
                x = to_f32(x)
            rows.append(((float(k), x), [en_ref(k, x)]))
        rows += [((float(k), 0.0), [en_ref(k, 0.0)]) for k in (2, 3, 10)]
        # regression (s_zeta-22, -23): the recurrence from E₁ near x = 1, and E₀ where e⁻ˣ/x
        # overflows (+∞)
        rr = random.Random(325 + single)
        extra = [(2.0, 1.143073742542361), (2.0, 1.1179508001575749), (3.0, 1.0010928798726808), (1.0, 0.9513045267373101)]
        extra += [(float(rr.choice([2, 3, 4, 5])), rr.uniform(0.9, 1.3)) for _ in range(n(int(80 * c)))]
        if not single:
            extra += [(0.0, 4e-309), (0.0, 5e-324), (0.0, 5.5e-309), (0.0, 1e-308)]
        for k, x in extra:
            x = to_f32(x) if single else x
            rows.append(((k, x), [en_ref(int(k), x)]))
        out(f"en_{ty}", f"E_n(x), {ty}: arguments n (an integer), x", rows)


if wants("e1", "ei", "en", "expint"):
    gen_expint()


DILOG_X0 = mpmath.findroot(lambda t: mpmath.polylog(2, t).real, 12.6)


def dilog_ref(x):
    v = mp_arg(x)
    if mpmath.isinf(v):
        return -mpmath.inf
    return mpmath.polylog(2, v).real


def gen_dilog():
    for single, ty in [(False, "f64"), (True, "f32")]:
        c = 1 if not single else 0.6
        rng = random.Random(400 + single)
        xs = uniform(rng, -1.0, 1.0, n(int(150 * c)))
        xs += [s * x for s in (1, -1) for x in log_uniform(rng, 1e-300 if not single else 1e-40, 1e-2, n(int(20 * c)))]
        xs += [1 - x for x in log_uniform(rng, 1e-16, 0.25, n(int(25 * c)))]
        xs += [1 + x for x in log_uniform(rng, 1e-16, 1.0, n(int(25 * c)))]
        xs += uniform(rng, 1.0, 20.0, n(int(60 * c)))
        xs += [float(DILOG_X0) * (1 + s * x) for s in (1, -1) for x in log_uniform(rng, 1e-16, 1e-2, n(int(15 * c)))]
        xs += log_uniform(rng, 20.0, 1e300 if not single else 1e38, n(int(30 * c)))
        xs += [-x for x in log_uniform(rng, 1.0, 1e300 if not single else 1e38, n(int(50 * c)))]
        xs += [0.0, 0.5, 0.75, 1.0, 2.0, -1.0, float(DILOG_X0)]
        if single:
            xs = sorted(set(f32s(xs)))
        # regression (s_zeta-11): just below −1 (the inversion), near 3/4 (the series and the
        # reflection)
        rr = random.Random(405 + single)
        extra = [-1.0847038041673585, -1.037443400130222, 0.7367823889183898, 0.7538040576567853]
        extra += uniform(rr, -1.3, -1.0, n(int(60 * c))) + uniform(rr, 0.65, 0.85, n(int(60 * c)))
        xs += f32s(extra) if single else extra
        out(f"dilog_{ty}", f"dilog(x) = Re Li2(x), {ty}", [((x,), [dilog_ref(x)]) for x in xs])


BERN = [mpmath.bernoulli(2 * k) / mpmath.factorial(2 * k) for k in range(1, 41)]


def hurwitz_em(S, A, extra):
    N = max(0, int(mpmath.ceil(S + 60 - A))) + extra
    tot = mpmath.fsum((k + A) ** -S for k in range(N))
    B = A + N
    t = B ** (1 - S) / (S - 1) + B ** -S / 2
    for k in range(1, 41):
        t += BERN[k - 1] * mpmath.rf(S, 2 * k - 1) * B ** (-S - 2 * k + 1)
    return tot + t


def hurwitz_ref(s, a):
    S, A = mp_arg(s), mp_arg(a)
    if S < 1 or A <= 0:
        return mpmath.nan
    if S == 1:
        return mpmath.inf
    # far below the least subnormal: ζ(s, a) < A^(−s) (1 + A/(s − 1)) (and at 256 bits the
    # exponent −s ln a of a huge s would not carry the digits of the value)
    if A >= 1 and S * mpmath.log(A) - mpmath.log(1 + A / (S - 1)) > 1200:
        return mpmath.mpf(0)
    # mpmath's zeta(s, a) loses digits for large a: the Euler–Maclaurin sum here, its N and
    # its number of Bernoulli terms such that the remainder is far below 2^-200, checked
    # against a second N.
    r = hurwitz_em(S, A, 0)
    r2 = hurwitz_em(S, A, 25)
    assert abs(r - r2) <= abs(r) * mpmath.mpf(2) ** -150, (s, a)
    return r


def gen_hurwitz():
    for single, ty in [(False, "f64"), (True, "f32")]:
        c = 1 if not single else 0.6
        rng = random.Random(500 + single)
        rows = []
        for _ in range(n(int(400 * c))):
            r = rng.random()
            s = 1 + math.exp(rng.uniform(math.log(1e-12), math.log(1.0))) if r < 0.25 else rng.uniform(1.0, 30.0) if r < 0.8 else math.exp(rng.uniform(math.log(30.0), math.log(1000.0)))
            ra = rng.random()
            a = math.exp(rng.uniform(math.log(1e-10), 0.0)) if ra < 0.3 else rng.uniform(0.0, 10.0) if ra < 0.8 else math.exp(rng.uniform(math.log(10.0), math.log(1e10)))
            if single:
                s, a = to_f32(s), to_f32(a)
            if a == 0 or s < 1:
                continue
            rows.append(((s, a), [hurwitz_ref(s, a)]))
        rows += [((s, a), [hurwitz_ref(s, a)]) for s, a in [(2.0, 1.0), (2.0, 0.5), (3.0, 2.0), (1.0, 1.0), (0.5, 1.0), (2.0, 0.0), (2.0, -1.0)]]
        # regression (s_zeta-8, -9, -10): tiny a near s = 1 (the cut dropped the rest), huge a
        # (A^(−s) underflowed, A/(s − 1) overflowed), a just below 7 + s/2 (the rounded a + N)
        rr = random.Random(505 + single)
        extra = []
        if not single:
            extra += [(1.0 + 2.0 ** -52, 7e-17), (1.0 + 2.0 ** -52, 1.78546142370102e-17), (1.0 + 2.0 ** -52, 1e-20),
                      (1.0000001, 5e-17), (1.0 + 2.0 ** -10, 7e-17), (1.01, 5e-17),
                      (2.0, 1e170), (1.1, 1e302), (3.0, 1e120), (1.211518622195213, 1.99254842185827e279),
                      (1.5, 1e300), (1.001, 1e306), (1.0000000001, 1e299), (1.0 + 2.0 ** -52, 1e300),
                      (1.5, 1.7976931348623157e308), (1e300, 1e300),
                      (48.97104698503809, 31.026265782404277), (50.98787121532069, 31.639519903362004),
                      (113.07602174402705, 63.53796526710132), (114.31405384841739, 63.38774898980862)]
            extra += [(1 + math.exp(rr.uniform(math.log(2.0 ** -52), math.log(1e-2))), math.exp(rr.uniform(math.log(1e-20), math.log(1e-12)))) for _ in range(n(60))]
            extra += [(1 + math.exp(rr.uniform(math.log(1e-12), math.log(2.0))), math.exp(rr.uniform(math.log(1e100), math.log(1e308)))) for _ in range(n(40))]
        else:
            extra += [(to_f32(1 + math.exp(rr.uniform(math.log(2.0 ** -23), math.log(1e-2)))), to_f32(math.exp(rr.uniform(math.log(1e-30), math.log(1e-10))))) for _ in range(n(30))]
            extra += [(to_f32(1 + math.exp(rr.uniform(math.log(1e-6), math.log(2.0)))), to_f32(math.exp(rr.uniform(math.log(1e20), math.log(3e38))))) for _ in range(n(30))]
        for _ in range(n(int(60 * c))):
            sv = rr.uniform(20.0, 200.0)
            k = rr.randint(1, 3)
            a = 7 + sv / 2 - k + rr.uniform(-0.01, 0.0)
            if single:
                sv, a = to_f32(sv), to_f32(a)
            extra.append((sv, a))
        if not single:
            # subnormal results (their bound is in units of 2⁻¹⁰⁷⁴): the largest errors of the
            # survey on each path, and s ln a between 709 and 744 (the direct sum, the sum with
            # the tail, the tail alone)
            ru = random.Random(508)
            extra += [(163.99658126606357, 77.16836814430557), (177.18022112488345, 57.941252871833534),
                      (157.11551721618514, 91.30314542131406), (11.865465360994254, 3.0445078835736806e28),
                      (161.64151964430755, 84.85888244738561)]
            for _ in range(n(40)):
                sv = ru.uniform(20.0, 200.0)
                extra.append((sv, math.exp(ru.uniform(709.0, 744.0) / sv)))
        rows += [((sv, a), [hurwitz_ref(sv, a)]) for sv, a in extra]
        out(f"hurwitz_{ty}", f"Hurwitz zeta(s, a), {ty}: arguments s, a", rows)


if wants("dilog"):
    gen_dilog()
if wants("hurwitz"):
    gen_hurwitz()


def zeta_c(a, b):
    z = mpmath.mpc(mp_arg(a), mp_arg(b))
    if z.real >= 20:
        # the Dirichlet series (mpmath's own method would sum |t| terms): its tail past 400 is
        # below 400^(−19)/19 < 10⁻⁵¹
        return mpmath.fsum(mpmath.power(k, -z) for k in range(1, 401))
    with mpmath.workprec(320):
        r2 = mpmath.zeta(z)
    r = mpmath.zeta(z)
    assert abs(r - r2) <= abs(r) * mpmath.mpf(2) ** -120 + mpmath.mpf(2) ** -200, (a, b)
    return r


def zetac_regression(rng, single):
    """The regression set of the complex zeta function (s_zeta-1 to -6): σ ≥ 56 at any t, σ past
    60 and 290, the Riemann–Siegel range t > 10⁴ (to 10¹⁰), the left half-plane past σ = −170
    and with t > 200, tiny t at the trivial zeros, the strip near the real axis, and the points of
    the audit. Not filtered by |ζ|."""
    pts = [(293.0, 1.0), (294.0, 1.0), (300.0, 1.0), (500.0, 1.0), (1000.0, 10.0), (2000.0, 0.5),
           (246.0310990849004, -826.7268708653353), (274.7846597675654, 867.1033531760229),
           (1505.0965075416852, -0.04153163439071301), (60.0, 1e15), (80.0, 3e11), (56.0, 1.0),
           (-171.0, 1.0), (-180.5, 10.0), (-200.5, 0.5), (-206.1145403378245, 138.07210375404588),
           (-250.5, 0.5), (-259.5, 0.01), (-300.0, 1.0),
           (-10.5, 250.0), (-20.5, 250.0), (-29.81567501707877, -234.47181272117908), (-40.5, 250.0),
           (-59.949315628777335, 205.38159248655947), (-80.5, 300.0), (-2.9, 100.0),
           (0.5869055696932023, -0.007637103133296675), (0.5, -0.5768671268520098),
           (-2.744088748256996, 105.60919529628768), (0.4995828708345811, 0.5339368497233771),
           (-154.86927619082192, -141.8197913713972), (-171.4369661212803, 177.7086910688032),
           (-62.0, 9.498931719573825e-280), (-170.0, 1e-300), (-100.0, 1e-300), (-20.0, 1e-320),
           (-50.0, 1e-300), (-240.0, 1.9208691529757967e-283),
           # the largest errors of the survey in the strip (the Euler–Maclaurin formula for
           # 0 ≤ σ < 1/2, the reflection beyond the series about 1)
           (0.08345517569682681, 6.072450952390156), (0.48287845846660926, 3.0305072254480407)]
    for _ in range(n(40)):
        pts.append((rng.uniform(56.0, 3000.0), rng.choice((-1, 1)) * math.exp(rng.uniform(math.log(1e-3), math.log(1e12)))))
    for _ in range(n(30)):
        pts.append((rng.uniform(10.0, 56.0), rng.choice((-1, 1)) * math.exp(rng.uniform(math.log(1e3), math.log(1e9)))))
    for t in (1.0123e4, 3.1e4, 1.07e5, 1.03e6, 1.01e7, 1.012e8, 3.037e8, 1.0123e9, 1.0e10):
        for sg in (0.5, 0.75, 2.0, -0.5, 5.5):
            pts.append((sg, t * rng.uniform(1.0, 1.001)))
    for _ in range(n(60)):
        pts.append((rng.uniform(-260.0, -3.0), rng.choice((-1, 1)) * rng.uniform(200.0, 3000.0)))
    for _ in range(n(40)):
        pts.append((rng.uniform(-262.0, -170.0), rng.choice((-1, 1)) * rng.uniform(0.0, 200.0)))
    for _ in range(n(30)):
        pts.append((-2.0 * rng.randint(1, 120), math.exp(rng.uniform(math.log(1e-320), math.log(1e-20)))))
    for _ in range(n(80)):
        pts.append((rng.uniform(-0.5, 1.5), rng.choice((-1, 1)) * math.exp(rng.uniform(math.log(1e-6), math.log(2.9)))))
    for _ in range(n(40)):
        pts.append((rng.uniform(-3.0, 0.0), rng.choice((-1, 1)) * rng.uniform(13.5, 200.0)))
    # the Euler–Maclaurin formula left of σ = 1/2 (5.8 ε in the strip, 7.5 ε of |ζ| for σ < 0,
    # found by the dense surveys of the fixes; the reflection since): its worst points, the band
    # −0.3 ≤ σ < 0 with 5 − 30σ ≤ t < 13.5, and 0 ≤ σ < 1/2 to t = 10⁴
    ru = random.Random(607 + single)
    pts += [(0.031382427816565306, 6.319235549807937), (0.00720612791452202, 5.678854259294242),
            (0.022128619072456834, 5.10076965385676), (0.12675733641193665, 169.22603322447625),
            (0.09347747799042971, 2668.573313688294), (0.30263866368029724, 7712.906636051734),
            (0.28770120607928024, 1415.5289849077737), (-0.2666091888971492, 13.42307514981129),
            (-0.15223527293236033, 13.454571962393107)]
    for _ in range(n(30)):
        sg = ru.uniform(-0.3, 0.0)
        pts.append((sg, ru.choice((-1, 1)) * ru.uniform(5.0 - 30.0 * sg, 13.5)))
    for _ in range(n(30)):
        pts.append((ru.uniform(0.0, 0.5), ru.choice((-1, 1)) * ru.uniform(5.0, 13.5)))
    for _ in range(n(30)):
        pts.append((ru.uniform(0.0, 0.5), ru.choice((-1, 1)) * math.exp(ru.uniform(math.log(13.5), math.log(1e4)))))
    # the largest errors of the final surveys, one or two a region of the bounds
    pts += [(0.2816284461778709, 9649.782598404516), (0.32052769592341684, 9811.833375027594),
            (0.4686376817225372, -6083.6003987267595), (0.35283997195043604, 269944.703057189),
            (0.30687227862371785, -12.265226434704605), (-1.1299714043699003, 6.24222044096562),
            (0.5756257397856813, 6765.21985639323), (0.5014153596548816, 49575.246086105384)]
    if single:
        pts = [(to_f32(a), to_f32(b)) for a, b in pts if abs(a) < 3e38 and abs(b) < 3e38]
    return [((a, b), [zeta_c(a, b)]) for a, b in pts]


def zetac_review1(rng, single):
    """The regression set of review round 1 of the fixes: the Riemann–Siegel range near the
    critical line to t = 2⁴⁰ (its terms rounded to doubles, and 1 − σ rounded in the reflection,
    reached 22 ε), the trivial zeros with a tiny t past n = 120 (an infinity for the finite
    t ζ′(−2n)) or with an f32 t (real parts of the wrong sign), and the left half-plane below
    t = 13.5 past σ = −12 (2.57 ε, the sine of the logarithmic form in double)."""
    pts = [(0.4968503801267799, 2794537398.715366), (0.4995476007645195, 775350.6075987144),
           (0.5019328089495687, 24225852502.355785), (0.5, -434944171877.1923),
           (0.5010340135446789, 66754.84202495475), (0.5, 52321.103999338324),
           (-400.0, 1e-300), (-420.0, 1e-300), (-350.0, 1e-280), (-300.0, 1e-300),
           (-256.0, 6.335842e-36), (-218.0, -1.2600015e-31), (-210.0, 2.1425854e-42)]
    for d in range(4, 12):
        for _ in range(n(3)):
            t = min(10 ** rng.uniform(d, d + 1), 1.0995e12)
            pts.append((0.5, rng.choice((-1, 1)) * t))
            pts.append((rng.uniform(0.49, 0.5), t * (1 + rng.uniform(-1e-3, 1e-3))))
            pts.append((rng.uniform(0.5, 0.51), t * (1 + rng.uniform(-1e-3, 1e-3))))
            pts.append((rng.uniform(0.0, 0.49), t * (1 + rng.uniform(-1e-3, 1e-3))))
    for _ in range(n(20)):
        pts.append((-2.0 * rng.randint(121, 220), math.exp(rng.uniform(math.log(1e-320), math.log(1e-150)))))
    for _ in range(n(20)):
        pts.append((-2.0 * rng.randint(60, 130), rng.choice((-1, 1)) * math.exp(rng.uniform(math.log(1e-44), math.log(1e-20)))))
    # left of the strip below t = 13.5 in the logarithmic form (σ ≤ −12), whose sin(πs/2) and
    # its product with ζ(1 − s) in double reached 2.57 ε against the stated 2.5 (drawn after the
    # rows above, which stay as they were)
    pts += [(-256.04017305436486, 10.613788266298485), (-26.436303203669695, -0.0018593799805256264)]
    for _ in range(n(25)):
        pts.append((rng.uniform(-260.0, -12.0), rng.choice((-1, 1)) * rng.uniform(5.0, 13.5)))
    if single:
        pts = [(to_f32(a), to_f32(b)) for a, b in pts if abs(a) < 3e38 and abs(b) < 3e38]
        pts = [(a, b) for a, b in pts if b != 0.0]
    return [((a, b), [zeta_c(a, b)]) for a, b in pts]


def zetac_review2(single):
    """The regression set of review round 2 of the fixes (f64 only): σ = 1 with a tiny t, where
    the reciprocal of the series about the pole squared t (subnormal below t = 1.5·10⁻¹⁵⁴: a NaN
    real part) and the scaling below t = 2⁻⁶⁰⁰ took the real part from ζ(1) = ∞; the value is
    γ − i/t + O(t), −i∞ past the range (t < 2⁻¹⁰²⁴), both signs of t."""
    if single:
        return []
    ts = [1e-154, 1e-155, 1e-160, 1e-162, 1e-170, 1e-200, 2.0 ** -500, math.nextafter(2.0 ** -500, 0.0),
          2.0 ** -600, 1e-300, 1e-308, 1e-310, 5e-324]
    rows = []
    for t in ts:
        for b in (t, -t):
            v = zeta_c(1.0, b)
            assert abs(v.real - mpmath.euler) < mpmath.mpf(2) ** -200 and abs(v.imag * b + 1) < mpmath.mpf(2) ** -200, b
            rows.append(((1.0, b), [v]))
    return rows


def gen_zeta_complex():
    for single, ty in [(False, "c64"), (True, "c32")]:
        c = 1 if not single else 0.5
        rng = random.Random(600 + single)
        pts = []
        for _ in range(n(int(500 * c))):
            r = rng.random()
            if r < 0.35:
                sg, t = rng.uniform(0.5, 6.0), rng.uniform(-60.0, 60.0)
            elif r < 0.5:
                sg, t = rng.uniform(-20.0, 0.5), rng.uniform(-40.0, 40.0)
            elif r < 0.6:
                sg, t = 1 + rng.uniform(-1e-3, 1e-3), rng.uniform(-1e-3, 1e-3)
            elif r < 0.75:
                sg, t = rng.uniform(6.0, 80.0), rng.uniform(-200.0, 200.0)
            elif r < 0.9:
                sg, t = rng.uniform(-2.0, 3.0), math.copysign(math.exp(rng.uniform(math.log(60.0), math.log(1000.0))), rng.random() - 0.5)
            else:
                sg, t = rng.uniform(-5.0, 5.0), rng.uniform(-1e-6, 1e-6)
            if single:
                sg, t = to_f32(sg), to_f32(t)
            v = zeta_c(sg, t)
            # away from the zeros: |zeta| >= 1e-3 (near them only an absolute bound holds)
            if abs(v) < 1e-3:
                continue
            pts.append(((sg, t), [v]))
        pts += [((sg, t), [zeta_c(sg, t)]) for sg, t in [(2.0, 0.0), (0.5, 14.0), (-1.0, 1.0), (3.0, -4.0)]]
        pts += zetac_regression(random.Random(605 + single), single)
        pts += zetac_review1(random.Random(615 + single), single)
        pts += zetac_review2(single)
        out(f"zetac_{ty}", f"Riemann zeta(s), complex s, {ty}: arguments re, im; |zeta| >= 1e-3 but in the regression set", pts)
        # near the non-trivial zeros on the critical line, for an absolute bound
        rng = random.Random(610 + single)
        rows = []
        zs = [mpmath.zetazero(k).imag for k in range(1, 30)]
        for _ in range(n(int(80 * c))):
            g = float(rng.choice(zs)) + rng.uniform(-0.01, 0.01)
            sg = 0.5 + rng.uniform(-0.01, 0.01)
            if single:
                sg, g = to_f32(sg), to_f32(g)
            rows.append(((sg, g), [zeta_c(sg, g)]))
        out(f"zetaczeros_{ty}", f"Riemann zeta(s) near its first zeros, {ty}: arguments re, im", rows)


if wants("zetac"):
    gen_zeta_complex()


def polylog_ref(s, x):
    S, X = mp_arg(s), mp_arg(x)
    if X > 1:
        return mpmath.nan
    if X == 1:
        return mpmath.zeta(S) if S > 1 else mpmath.inf
    if X == 0:
        return mpmath.mpf(0)
    with mpmath.workprec(320):
        r2 = mpmath.polylog(S, X)
    r = mpmath.polylog(S, X)
    r, r2 = mpmath.re(r), mpmath.re(r2)
    assert abs(r - r2) <= abs(r) * mpmath.mpf(2) ** -110 + mpmath.mpf(2) ** -1000, (s, x, r, r2)
    return r


def li_negative_int(m, x):
    """Li_{−m}(x) for an integer m ≥ 0 by exact rational arithmetic: x A_m(x)/(1 − x)^(m+1), A_m
    the Eulerian polynomial."""
    from fractions import Fraction

    X = Fraction(x)
    a = [1]
    for k in range(2, m + 1):
        b = [0] * k
        for j in range(k):
            b[j] = (j + 1) * (a[j] if j < len(a) else 0) + (k - j) * (a[j - 1] if 0 < j <= len(a) else 0)
        a = b
    if m <= 1:
        a = [1]
    num = sum(c * X ** j for j, c in enumerate(a))
    v = X * num / (1 - X) ** (m + 1)
    return mpmath.mpf(v.numerator) / v.denominator


def li_hurwitz(S, X):
    """Li_S(X) for S < 0, X < 0: 2 Γ(1 − S)(2π)^(S−1) Re[i^(1−S) ζ(1 − S, 1/2 − iμ/2π)], μ = ln(−X)."""
    mu = mpmath.log(-X)
    a = mpmath.mpf(1) / 2 - 1j * mu / (2 * mpmath.pi)
    z = mpmath.zeta(1 - S, a)
    return 2 * mpmath.gamma(1 - S) * mpmath.power(2 * mpmath.pi, S - 1) * mpmath.re(mpmath.expjpi((1 - S) / 2) * z)


def li_jonquiere(S, X):
    """Li_S(X) for S < 0, 0 < X < 1: Γ(1 − S)[(−μ)^(S−1) + 2 Re((2π)^(S−1) e^(iπ(S−1)/2)
    ζ(1 − S, 1 + iμ/2π))], μ = ln X."""
    mu = mpmath.log(X)
    tw = 2 * mpmath.pi
    z = mpmath.zeta(1 - S, 1 + 1j * mu / tw)
    rest = 2 * mpmath.re(mpmath.power(tw, S - 1) * mpmath.expjpi((S - 1) / 2) * z)
    return mpmath.gamma(1 - S) * (mpmath.power(-mu, S - 1) + rest)


def li_inverse_int(m, X):
    """Li_m(X) for an integer m ≥ 1 and X < −1: −(−1)^m Li_m(1/X) − Σ 2η(2j) μ^(m−2j)/(m − 2j)!."""
    mu = mpmath.log(-X)
    inner = mpmath.polylog(m, 1 / X)
    p = mpmath.fsum((1 if j == 0 else 2 * mpmath.altzeta(2 * j)) * mu ** (m - 2 * j) / mpmath.factorial(m - 2 * j) for j in range(m // 2 + 1))
    return -((-1) ** m) * inner - p


def li_large_order(S, X):
    """Li_S(X) for a huge order: its series x + x²2^(−S) + …, asymptotic past |x| = 1, to a term
    below 2^(−400) of it."""
    t = [mpmath.power(X, k) * mpmath.power(k, -S) for k in range(1, 6)]
    assert abs(t[-1]) < abs(t[0]) * mpmath.mpf(2) ** -400, (S, X)
    return mpmath.fsum(t)


def polylog_ref2(s, x):
    """The references of the regression set (see the module's note)."""
    S, X = mp_arg(s), mp_arg(x)
    if X == 1:
        return mpmath.zeta(S) if S > 1 else mpmath.inf
    # an integer order past −1 by the inversion, exact for every order (the series of a large order
    # is only asymptotic there: its terms grow again past k ≈ S/ln|x|)
    if S >= 1 and S == int(S) and X < -1:
        return li_inverse_int(int(S), X)
    if S > 60 + abs(mpmath.log(abs(X), 2)) + 1:
        return li_large_order(S, X)
    if S < 0 and S == int(S):
        return li_negative_int(int(-S), x)
    if 0 < S < mpmath.mpf(10) ** -60:
        # likewise a positive order this small (subnormal), |S ∂ₛLi| < 40 |S| |Li|
        return X / (1 - X)
    if -mpmath.mpf(10) ** -60 < S < 0:
        # Li_0 = x/(1 − x), within |S| ln stuff < 10⁻⁵⁸ (the Hurwitz form's pole would need more
        # than 256 bits to cancel)
        return X / (1 - X)
    if S < 0 and X < 0:
        return li_hurwitz(S, X)
    if S < 0 and X > 0:
        return li_jonquiere(S, X)
    return polylog_ref(s, x)


def polylog_cond2(s, x, v):
    """The condition number of the regression set: for an integer order |x ∂ₓLi|/|Li| =
    |Li_(s−1)/Li_s| (s is exact), else as polylog_cond."""
    S, X = mp_arg(s), mp_arg(x)
    if S >= 0 or X >= 1 or X == 0:
        return mpmath.mpf(1)
    if mpmath.isinf(v) or abs(v) > F64_MAX:
        return mpmath.mpf(1)
    if v == 0:
        return mpmath.inf
    if S == int(S):
        return max(mpmath.mpf(1), abs(li_negative_int(int(-S) + 1, x) / v))
    f = li_hurwitz if X < 0 else li_jonquiere
    dx = f(S - 1, X)
    ds = mpmath.diff(lambda t: f(t, X), S)
    return max(mpmath.mpf(1), (abs(dx) + abs(S * ds)) / abs(v))


def polylog_regression(rng, single):
    """The regression set of the polylogarithm (s_zeta-13 to -20)."""
    pts = [(-0.01565104796801646, -0.12260566708241478), (-0.010809728094889434, -0.3992904455234516),
           (-1e-6, -0.6), (-1e-10, -0.6), (-1e-12, -0.6), (-1.4726550341833386e-12, -0.3889832237751385),
           (-1.3742577956848978e-12, -416072549.4671279), (-1e-15, -0.6), (-1e-16, -0.6), (-1e-300, -0.6),
           (-3.0, -1.1565307818297381), (-12.0, -0.37783947202014767), (-16.0, -0.8636639127197592),
           (-30.0, -1.7861725404150706), (-46.0, -0.04284267695459443), (-64.0, -0.8406290647845185),
           (-64.0, -1.5), (-64.0, -1.3143001729010633), (-30.0, -1.0), (-64.0, -1.0),
           (-50.0, -980661.9085843436), (-64.0, -29161.390265617098), (-64.0, -1e6), (-30.0, -1e12),
           (-5.0, -1e80), (-3.0, -1e200), (-29.0, -173114823056.5652), (-32.0, -2869947085.3346896),
           (-62.0, 0.4794832072060483), (-64.0, 0.7385758003399454), (-64.0, 0.6), (-58.0, 0.9839157370338123),
           (-16.0, 0.7602550206226898), (-65.0, -0.5), (-100.0, -0.3), (-100.0, -2.5), (-150.0, -0.5),
           (-150.0, 0.5), (-200.0, -0.3), (-170.5, -1e30), (-171.5, -3e16), (-180.5, -0.5), (-200.5, -1e60),
           (-20.5, 1 - 2.0 ** -53), (-100.0, 0.99), (-1000.0, 0.5), (-25.818325412678536, 0.9999999999999957),
           (2023.0, 0.5), (2024.0, 0.5), (2030.0, 0.5), (2050.5, 0.5), (4073.0, 0.5), (1e12, 0.5),
           (1e20, 0.5), (1e300, 0.5), (2023.9, 0.999), (2030.0, -3.0), (2030.5, -3.0), (170.0, -1e100),
           (171.0, -1e60), (171.0, -1e100), (200.0, -1e100), (200.0, -1e200), (300.0, -1e300), (171.5, -1e300),
           (1.5040116181876138, 0.6645143587084922), (1.7053943041941062, 0.6424025084212109),
           (0.5, 0.9999999999999962), (0.6586512592567728, 0.9999999999866493), (52.0, -62193392441.619095),
           (48.0, -183480659018525.4), (8.589008705791079, -31.551964101894708),
           (-15.734274606313944, 0.9996269291847417), (-7.608167006876511, 0.5056402371926697)]
    for _ in range(n(50)):
        pts.append((-math.exp(rng.uniform(math.log(1e-16), math.log(1.0))), -rng.uniform(0.005, 3.0)))
    for _ in range(n(60)):
        pts.append((float(-rng.randint(2, 80)), -math.exp(rng.uniform(math.log(1e-6), math.log(1e12)))))
    for _ in range(n(30)):
        pts.append((float(-rng.randint(2, 80)), rng.uniform(0.0, 0.999)))
    for _ in range(n(30)):
        pts.append((-rng.uniform(1.0, 250.0), -math.exp(rng.uniform(math.log(1e-3), math.log(1e40)))))
    for _ in range(n(30)):
        pts.append((-rng.uniform(1.0, 250.0), rng.uniform(0.01, 0.999)))
    for _ in range(n(40)):
        pts.append((rng.uniform(1.0, 3.0), rng.uniform(0.6, 0.75)))
    for _ in range(n(20)):
        pts.append((rng.uniform(0.0, 1.0), 1 - math.exp(rng.uniform(math.log(1e-15), math.log(1e-2)))))
    for _ in range(n(20)):
        pts.append((float(rng.randint(100, 1000)), -math.exp(rng.uniform(math.log(1e30), math.log(1e300)))))
    rows = []
    for sv, x in pts:
        if single:
            if abs(sv) >= 3e38 or abs(x) >= 3e38:
                continue
            sv, x = to_f32(sv), to_f32(x)
            if x > 1:
                continue
        v = polylog_ref2(sv, x)
        rows.append(((sv, x), [v, polylog_cond2(sv, x, v)]))
    return rows


def polylog_review1(rng, single):
    """The regression set of review round 1 of the fixes (f64 only): Sommerfeld's expansion with
    s just below a power of 2 (1 + s rounded: 620 ulps at s = 127.7), the integral of Fermi and
    Dirac for large orders (ln Γ(s) and the powers at the nodes in double: 1974 ulps at s = 555),
    negative orders with |x| near 2^(±|s|) (where the Hurwitz form's terms are 50 times the
    result: 76 κ at Li₋₇₅(−1.6·10⁻²³)), and s > 0 with −3/4 ≤ x < −1/2 (the expansion about −1:
    3.25 ulps for s ≥ 1, 6.4 below)."""
    if single:
        return []
    pts = [(127.68723671408164, -1.1616108079044661e288), (63.03077463081349, -5.72825412477934e264),
           (31.97527985785239, -1.4279018720273414e94), (15.909421190274935, -3.033933841057965e67),
           (7.640655877911983, -1.4251902969985922e134), (3.4507088353539443, -2.463165342427401e130),
           (554.8953966753616, -1.6458745875901356e252), (150.3, -1e100), (126.66070522077007, -2.655840421665018e173),
           (299.73697352425245, -3.288610153239495e213),
           (-75.0, -1.6287527279532503e-23), (-80.0, -1.790174677039718e24), (-73.0, -1.3569088928787636e22),
           (-39.0, -872808674474.3723), (-74.96842749393772, -1.3928269449385304e-24)]
    for lo in (3, 7, 15, 31, 63, 127):
        for _ in range(n(6)):
            sv = rng.uniform(lo, lo + 1)
            pts.append((sv, -math.exp(rng.uniform(max(4 * sv + 1, math.log(2.4e16) + 1), 708.0))))
    for _ in range(n(20)):
        sv = rng.uniform(90.0, 400.0)
        lo, hi = max(sv / 2, (sv - 60) * math.log(2.0)), min(700.0, 4 * sv)
        if lo < hi:
            pts.append((sv, -math.exp(rng.uniform(lo, hi))))
    for _ in range(n(30)):
        k = rng.randint(2, 80)
        pts.append((float(-k), -(2.0 ** -k) * 10 ** rng.uniform(-1.5, 1.5)))
        pts.append((float(-k), -(2.0 ** k) * 10 ** rng.uniform(-1.5, 1.5)))
    for _ in range(n(15)):
        sv = -rng.uniform(2.0, 80.0)
        pts.append((sv, -(2.0 ** sv) * 10 ** rng.uniform(-1.5, 1.5)))
    # s ≥ 1 with −3/4 ≤ x < −0.6, where the expansion about −1 starts a binade above Li_s and
    # cancels by up to 1.6 (3.25 ulps against the stated 3), drawn after the rows above
    pts += [(1.3145706295314235, -0.6060701734582992), (1.0809748360786056, -0.6147514755415665),
            (1.1755338912748965, -0.6185927250348958)]
    for _ in range(n(30)):
        pts.append((rng.uniform(1.0, 3.0), rng.uniform(-0.75, -0.6)))
    # and 0 < s < 1 with −3/4 ≤ x < −1/2, the same expansion (6.4 ulps against the stated 4.5)
    pts += [(0.9352841750778765, -0.5499183699102135), (0.9725780370015147, -0.533264343207155),
            (0.3743143069696747, -0.5761043902418217), (0.03314492225781459, -0.596955207470814)]
    for _ in range(n(40)):
        pts.append((rng.uniform(0.0, 1.0), rng.uniform(-0.75, -0.5)))
    rows = []
    for sv, x in pts:
        v = polylog_ref2(sv, x)
        rows.append(((sv, x), [v, polylog_cond2(sv, x, v)]))
    return rows


def polylog_review2(rng, single):
    """The regression set of review round 2 of the fixes (f64 only): non-integer −1 < s < 0 with
    x < 0 by decade of |x| to 10³⁰⁷, the Hurwitz form, whose pole pair (taken out for every
    s > −1) is of size |a|^(1−s) against a result of |a| sin(π|s|)/|s| and whose remainder, pole
    parts and Re[i^(1−s) Z] were in double (84 κ at Li₋₀.₉₉₅(−10²⁸⁷), 5.9 κ at |x| = 6·10¹⁶,
    4.3 κ near s = 0 with κ = 1); and positive subnormal orders (NaN past x = −2 from Γ(s))."""
    if single:
        return []
    pts = [(-0.9458970006505468, -5.988621018848669e16), (-0.991471151410543, -1.5933782317048956e16),
           (-0.9332302193921705, -8.455014040183486e27), (-0.9951507284315685, -1.0522533332726049e287),
           (-0.009837771536763178, -7.147819010802523e29), (-0.9899687718062896, -1.560144256178833e19),
           (-0.936607539467426, -1.2036680089663435e30), (-0.9722302485390166, -9.080100114215104e282),
           (-0.9264723208297508, -2.430463154541963e28), (-0.2208185156403754, -1.9963599814698023e140)]
    for d in range(8, 308, 5):
        for lo, hi in ((-1.0, -0.85), (-1.0, -0.85), (-0.85, -0.25), (-0.25, 0.0)):
            pts.append((rng.uniform(lo, hi), -10.0 ** rng.uniform(d, min(d + 5.0, 307.5))))
    # either side of the switch of the pole pair at s = −1/4
    pts += [(-0.25, -1e100), (math.nextafter(-0.25, 0.0), -1e100), (math.nextafter(-0.25, -1.0), -1e100)]
    # positive subnormal orders, where Li_s = x/(1 − x) to far below an ulp
    pts += [(5e-324, -3.0), (1e-310, -100.0), (5e-324, -2.3e16), (2.225073858507e-308, -50.0),
            (1e-320, -1e10), (5e-324, 0.5), (1e-315, -1e300), (1e-310, -0.9)]
    rows = []
    for sv, x in pts:
        v = polylog_ref2(sv, x)
        rows.append(((sv, x), [v, polylog_cond2(sv, x, v)]))
    return rows


def polylog_fits(rng, single):
    """The fits of the speed pass (f64 only): s = 1/2, 3/2, 5/2 and 3 on −2.4·10¹⁶ ≤ x < 0, that is
    x·G(x) on [−3/4, 0) (uniform and log-uniform to 10⁻³⁰⁰), −F(μ) on [−2, −3/4) and
    −μˢR(μ)/Γ(s + 1) on the nine pieces of v = ln μ past −2 (μ = ln(−x)), with the ends of those
    pieces (v = −3/8 + k/2) and of the three forms."""
    if single:
        return []
    pts = []
    for sv in (0.5, 1.5, 2.5, 3.0):
        pts += [(sv, rng.uniform(-0.75, 0.0)) for _ in range(6)]
        pts += [(sv, -math.exp(rng.uniform(math.log(1e-300), math.log(0.75)))) for _ in range(3)]
        pts += [(sv, rng.uniform(-2.0, -0.75)) for _ in range(6)]
        for k in range(1, 9):
            pts.append((sv, -math.exp(math.exp(-0.375 + 0.5 * k))))
        lo, hi = math.log(math.log(2.0)), math.log(math.log(2.4e16))
        pts += [(sv, -math.exp(math.exp(rng.uniform(lo, hi)))) for _ in range(12)]
        pts += [(sv, -0.75), (sv, math.nextafter(-0.75, 0.0)), (sv, -2.0), (sv, math.nextafter(-2.0, 0.0)),
                (sv, -2.4e16), (sv, math.nextafter(-2.4e16, 0.0))]
    rows = []
    for sv, x in pts:
        try:
            v = polylog_ref2(sv, x)
        except AssertionError:
            continue
        rows.append(((sv, x), [v, polylog_cond2(sv, x, v)]))
    return rows


def polylog_cond(s, x):
    """The condition number of Li_s(x) in its arguments, |x ∂ₓLi| + |s ∂ₛLi| over |Li|
    (x ∂ₓ Li_s = Li_(s−1)), for s < 0, where Li_s has real zeros; for an integer s < 0, whose
    order is exact to every method, |x ∂ₓLi|/|Li| alone (by exact rational arithmetic); 1
    elsewhere."""
    S, X = mp_arg(s), mp_arg(x)
    if S >= 0 or X >= 1 or X == 0:
        return mpmath.mpf(1)
    if S == int(S):
        v = li_negative_int(int(-S), x)
        return mpmath.inf if v == 0 else max(mpmath.mpf(1), abs(li_negative_int(int(-S) + 1, x) / v))
    v = mpmath.re(mpmath.polylog(S, X))
    if v == 0:
        return mpmath.inf
    dx = mpmath.re(mpmath.polylog(S - 1, X))
    ds = mpmath.diff(lambda t: mpmath.re(mpmath.polylog(t, X)), S)
    return max(mpmath.mpf(1), (abs(dx) + abs(S * ds)) / abs(v))


def gen_polylog():
    for single, ty in [(False, "f64"), (True, "f32")]:
        c = 1 if not single else 0.6
        rng = random.Random(700 + single)
        rows = []
        for _ in range(n(int(700 * c))):
            r = rng.random()
            if r < 0.3:
                s = rng.uniform(-4.0, 12.0)
            elif r < 0.5:
                s = float(rng.choice([3, 4, 5, 6, 7, -2, -3, 10]))
            elif r < 0.7:
                s = rng.choice([1, 2, 3, 4, 5]) + rng.choice([-1, 1]) * math.exp(rng.uniform(math.log(1e-12), math.log(0.4)))
            elif r < 0.85:
                s = rng.uniform(12.0, 60.0)
            else:
                s = rng.uniform(-10.0, -2.0)
            rx = rng.random()
            if rx < 0.25:
                x = rng.uniform(-0.5, 0.5)
            elif rx < 0.45:
                x = rng.uniform(0.5, 1.0)
            elif rx < 0.55:
                x = 1 - math.exp(rng.uniform(math.log(1e-12), math.log(0.5)))
            elif rx < 0.75:
                x = rng.uniform(-2.0, -0.5)
            else:
                x = -math.exp(rng.uniform(math.log(2.0), math.log(1e12)))
            if single:
                s, x = to_f32(s), to_f32(x)
            try:
                v = polylog_ref(s, x)
            except AssertionError:
                continue
            rows.append(((s, x), [v, polylog_cond(s, x)]))
        extra = [(2.0, 0.5), (3.0, 1.0), (1.0, 0.5), (0.0, 0.5), (-1.0, 0.5), (3.0, -1.0), (0.5, 1.0), (2.5, 2.0)]
        rows += [((s, x), [polylog_ref(s, x), polylog_cond(s, x)]) for s, x in extra]
        rows += polylog_regression(random.Random(705 + single), single)
        rows += polylog_review1(random.Random(715 + single), single)
        rows += polylog_review2(random.Random(725 + single), single)
        rows += polylog_fits(random.Random(735 + single), single)
        out(
            f"polylog_{ty}",
            f"polylog Li_s(x), {ty}: arguments s, x; references Li_s(x), then its condition number "
            "(|x d/dx| + |s d/ds|)/|Li| for s < 0, |x d/dx|/|Li| for an integer s < 0 (1 elsewhere)",
            rows,
        )


if wants("polylog"):
    gen_polylog()
