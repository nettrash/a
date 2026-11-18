"""The zeta area's reference data (mpmath, 256 bits): the Riemann zeta function (real and complex),
the Hurwitz zeta function, the dilogarithm and the polylogarithm, both real branches of the
Lambert W function and the exponential integrals E1, Ei and En.

Run from this directory: `python3 gen_zeta.py [only…]` (names of the files' functions to make
only those). `ZETA_SCALE=k ZETA_OUT=dir` multiplies the counts by `k` and writes to `dir` (for the
larger checks outside the repository).
"""

import math
import os
import random
import sys
import tempfile

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "a-special-zeta-pycache")

import mpmath  # noqa: E402

from gen_common import log_uniform, mp_arg, to_f32, uniform, write

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
            rows = [((x,), [lambert(x, k)]) for x in xs]
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
        rows = [((x,), [zeta_real(x)]) for x in xs]
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
        out(f"hurwitz_{ty}", f"Hurwitz zeta(s, a), {ty}: arguments s, a", rows)


if wants("dilog"):
    gen_dilog()
if wants("hurwitz"):
    gen_hurwitz()


def zeta_c(a, b):
    z = mpmath.mpc(mp_arg(a), mp_arg(b))
    with mpmath.workprec(320):
        r2 = mpmath.zeta(z)
    r = mpmath.zeta(z)
    assert abs(r - r2) <= abs(r) * mpmath.mpf(2) ** -120 + mpmath.mpf(2) ** -200, (a, b)
    return r


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
        out(f"zetac_{ty}", f"Riemann zeta(s), complex s, {ty}: arguments re, im; |zeta| >= 1e-3", pts)
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


def polylog_cond(s, x):
    """The condition number of Li_s(x) in its arguments, |x ∂ₓLi| + |s ∂ₛLi| over |Li|
    (x ∂ₓ Li_s = Li_(s−1)), for s < 0, where Li_s has real zeros; 1 elsewhere."""
    S, X = mp_arg(s), mp_arg(x)
    if S >= 0 or X >= 1 or X == 0:
        return mpmath.mpf(1)
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
        out(
            f"polylog_{ty}",
            f"polylog Li_s(x), {ty}: arguments s, x; references Li_s(x), then its condition number "
            "(|x d/dx| + |s d/ds|)/|Li| for s < 0 (1 elsewhere)",
            rows,
        )


if wants("polylog"):
    gen_polylog()
