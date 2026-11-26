"""The elliptic area's reference data (mpmath, 256 bits): the elliptic integrals of the parameter
m, Carlson's integrals, the Jacobi elliptic functions and the orthogonal polynomials.

Principal values (Pi with n sin^2(phi) > 1, n > 1, R_J with p < 0, R_C with y < 0) are the real
parts of mpmath's values, checked against Cauchy principal-value quadrature when this was written.

The orthogonal-polynomial files carry a second reference, the scale S of the case: the largest
|P| on the interval of orthogonality (Legendre, Chebyshev, Jacobi, Gegenbauer, associated
Legendre), or the weighted largest value e^(x^2/2) max|H_n e^(-t^2/2)| (Hermite, e^(x^2/4) for He)
and e^(x/2) max|L e^(-t/2)| (Laguerre) inside the oscillatory region; 0 elsewhere. The tests
measure |got - ref| in ulps of max(|ref|, S), relative error away from the zeros and an
absolute error relative to the polynomial's scale near them.

Each file ends with the edge cases of the audit of 2026-11-21 and of the second pass of its fixes
(the `*_edges` functions below; `huge_edges`, `pv_edges`, `rc_sub_edges` and
`jacobi_near_one_edges` are the second pass's, after all the others), then those of the two
reviews of the fixes (`review1_*`, then `review2_*`), and for the Jacobi functions those of the
confirmation of the second round (`final_jacobi_edges`, last): the arguments at which the defects
showed and neighbours of them, drawn from generators of their own so that the cases before them
stay as they were. Their references come at precisions
chosen for them (`RP`), and for the parameters at which mpmath's polynomials fail, from exact sums.

Run from this directory: python3 gen_elliptic.py
"""

import math
import random
import struct
from fractions import Fraction

import mpmath

from gen_common import log_uniform, mp_arg, to_f32, uniform, write

mp = mpmath.mp
rng = random.Random(2020)
M = mp_arg


def R(fn, *args):
    """fn(*args) of float arguments, evaluated at 256 and 768 bits (and 2048 if those disagree
    beyond 2^-150): mpmath's Carlson integrals lose precision for arguments spanning 10^+-300."""
    def at(prec):
        with mpmath.workprec(prec):
            v = fn(*[mpmath.mpf(a) for a in args])
            v = re(v)
        return v
    a, b = at(256), at(768)
    if a == b or (mpmath.isfinite(b) and b != 0 and abs(a - b) <= abs(b) * mpmath.mpf(2) ** -150):
        return b
    c = at(2048)
    assert c == b or abs(b - c) <= abs(c) * mpmath.mpf(2) ** -150, (fn, args, a, b, c)
    return c


def re(v):
    return mpmath.re(v) if isinstance(v, mpmath.mpc) else v


def RP(fn, *args, prec=None):
    """fn(*args) of float arguments at `prec` bits (default: 400 plus twice the largest binary
    exponent of the arguments, for spreads up to 10^+-300 and huge arguments), checked against
    `prec + 512` bits to 2^-150."""
    if prec is None:
        e = max([abs(math.frexp(a)[1]) for a in args if a != 0 and math.isfinite(a)] + [0])
        prec = 400 + 2 * e

    def at(p):
        with mpmath.workprec(p):
            return re(fn(*[mpmath.mpf(a) for a in args]))
    a, b = at(prec), at(prec + 512)
    ok = a == b or (mpmath.isfinite(b) and b != 0 and abs(a - b) <= abs(b) * mpmath.mpf(2) ** -150)
    assert ok, (fn, args, a, b)
    return b


def fb(h):
    """The f64 of the bits `h`."""
    return struct.unpack("<d", struct.pack("<Q", h))[0]


def edge_rng(part, single):
    """The generator of the edge cases of a part, apart from the random cases' `rng`."""
    return random.Random("elliptic-edges-" + part + ("-f32" if single else "-f64"))


def edge_args(xs, single):
    """The arguments as f32 values for an f32 file (beyond its range an infinity, which the
    callers drop)."""
    def f32(v):
        return to_f32(v) if abs(v) <= 3.4028234663852886e38 else math.copysign(math.inf, v)
    return [tuple(f32(v) for v in x) for x in xs] if single else [tuple(x) for x in xs]


def jacobi_exact(n, a, b, x):
    """P_n^(a,b)(x) by DLMF 18.5.8, sum_s C(n+a, n-s) C(n+b, s) ((x-1)/2)^s ((x+1)/2)^(n-s), at the
    working precision: exact in effect at 3n + 200 bits, for every parameter (mpmath.jacobi is NaN
    at some of the degenerate ones)."""
    a, b, x = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(x)
    y, z = (x - 1) / 2, (x + 1) / 2
    return mpmath.fsum(mpmath.binomial(n + a, n - s) * mpmath.binomial(n + b, s) * y ** s * z ** (n - s)
                       for s in range(n + 1))


def gegenbauer_exact(n, a, x):
    """C_n^(a)(x) by DLMF 18.5.10, sum_k (-1)^k (a)_(n-k)/(k! (n-2k)!) (2x)^(n-2k)."""
    a, x = mpmath.mpf(a), mpmath.mpf(x)
    return mpmath.fsum((-1) ** k * mpmath.rf(a, n - k) / (mpmath.factorial(k) * mpmath.factorial(n - 2 * k))
                       * (2 * x) ** (n - 2 * k) for k in range(n // 2 + 1))


def laguerre_exact(n, a, x):
    """L_n^(a)(x) = sum_k C(n + a, n - k) (-x)^k / k! (DLMF 18.5.12)."""
    a, x = mpmath.mpf(a), mpmath.mpf(x)
    return mpmath.fsum(mpmath.binomial(n + a, n - k) * (-x) ** k / mpmath.factorial(k) for k in range(n + 1))


def jacobi_rational(n, a, b, x):
    """P_n^(a,b)(x) of float arguments by DLMF 18.5.7 in exact rationals, rounded to 400 bits: for
    |x| > 2^400 with a vanishing or small leading coefficient, where the terms of the sums above
    are of the size x^n and cancel to a much smaller value (and for the parameters next to a
    degenerate a + b, where they cancel to the polynomial's scale)."""
    a, b, x = Fraction(a), Fraction(b), Fraction(x)
    y = (x - 1) / 2
    # The coefficient of y^j: (a + j + 1)_(n - j)/(n - j)! times (n + a + b + 1)_j/j!.
    lo = [Fraction(1)] * (n + 1)
    for j in range(n - 1, -1, -1):
        lo[j] = lo[j + 1] * (a + j + 1) / (n - j)
    s = Fraction(0)
    up = Fraction(1)
    yj = Fraction(1)
    for j in range(n + 1):
        if j > 0:
            up = up * (n + a + b + j) / j
        s += up * lo[j] * yj
        yj *= y
    if s == 0:
        return mpmath.mpf(0)
    with mpmath.workprec(400):
        return mpmath.mpf(s.numerator) / s.denominator


def exact_at(fn, n, *args):
    """An exact sum at 4n + 300 bits (its terms cancel by up to about 2^(3n))."""
    with mpmath.workprec(4 * n + 300):
        return +fn(n, *args)


def scale_exact(fn, n, *pars):
    """max |P| on [-1, 1] (600 points and the ends) by an exact sum, to a few digits."""
    with mpmath.workprec(3 * n + 80):
        ts = [mpmath.mpf(-1) + 2 * (k + mpmath.mpf(1) / 2) / 600 for k in range(600)] + [mpmath.mpf(-1), mpmath.mpf(1)]
        return max(abs(fn(n, *pars, t)) for t in ts)


def pi_complete(n, m):
    """Pi(n, m); for n > 1 the principal value K(m) - Pi(m/n, m) (DLMF 19.6.5), as mpmath's
    complex continuation is slow there (its real part is the same)."""
    if n > 1:
        return mpmath.ellipk(m) - mpmath.ellippi(m / n, m)
    return mpmath.ellippi(n, m)


def pi_incomplete(n, phi, m):
    """Pi(n; phi, m) for |phi| <= pi/2: for m > 1 by the reciprocal modulus (DLMF 19.7.4), for
    n > 1 by A&S 17.7.7 (both checked against mpmath's real parts when this was written)."""
    if m > 1:
        r = mpmath.sqrt(m)
        return pi_incomplete(n / m, mpmath.asin(r * mpmath.sin(phi)), 1 / m) / r
    if n > 1:
        nn = m / n
        p1 = mpmath.sqrt((n - 1) * (1 - nn))
        d = mpmath.sqrt(1 - m * mpmath.sin(phi) ** 2)
        t = mpmath.tan(phi)
        return (mpmath.ellipf(phi, m) - mpmath.ellippi(nn, phi, m)
                + mpmath.log(abs((d + p1 * t) / (d - p1 * t))) / (2 * p1))
    return re(mpmath.ellippi(n, phi, m))


# ------------------------------------------------------------------------------------------
# Complete integrals.


def m_args(n, single):
    xs = uniform(rng, -1.0, 1.0, n // 2, single)
    xs += [1.0 - 10.0 ** -rng.uniform(1, 16 if not single else 7) for _ in range(n // 8)]
    xs += [-x for x in log_uniform(rng, 1.0, 1e300 if not single else 1e30, n // 8, single)]
    xs += [s * x for s, x in zip([1, -1] * n, log_uniform(rng, 1e-300 if not single else 1e-30, 1e-2, n // 8, single))]
    xs += uniform(rng, 0.7, 1.0, n // 8, single)
    if single:
        xs = [to_f32(x) for x in xs]
    return [x for x in xs if x < 1.0]


def ke_edges(single):
    """K and E: the transformation m < -1 (m' and ln(1 - m) were rounded once) and the logarithmic
    form m >= 0.8 (ln m1 rounded once), where the audit found 1.42 and 0.90 ulps."""
    g = edge_rng("ke", single)
    xs = [fb(0xf5d6ed90ec3f1493), fb(0xd7311baca701c08c), fb(0xc00d1cdc19e15525), fb(0x3febc6bc3d224062),
          fb(0x3feffffff9a505b0), fb(0xc01f0999f281498f), fb(0x3fec8435019f5aec)]
    top = 30 if single else 307
    xs += [-(10.0 ** g.uniform(0.61, top)) for _ in range(40)]
    xs += [-g.uniform(1, 4) for _ in range(40)]
    xs += [g.uniform(0.8, 0.9) for _ in range(30)]
    xs += [1 - 10.0 ** -g.uniform(1, 7 if single else 15.9) for _ in range(30)]
    return [x for (x,) in edge_args([(x,) for x in xs], single) if x < 1.0 and math.isfinite(x)]


def pi_edges(single):
    """Pi(n, m): m <= n < 0 (the direct form cancelled by ln|m| and R_J underflowed), n > 1 with
    m << -1 (0), n << 0 with m -> 1 (a subnormal p), n -> 1- (above the 2-ulp bound)."""
    g = edge_rng("pi", single)
    if single:
        cs = [(-1e4, -1e4), (-1e20, -1e20), (-3e30, -3e38), (2.0, -1e30), (-3e38, 1 - 2.0 ** -24)]
    else:
        cs = [(-1e4, -1e4), (-1e16, -1e16), (-1e100, -1e100), (-1e205, -1e205), (-1e220, -1e220),
              (-1e300, -1e300), (2.0, -1e250), (10.0, -1e300), (fb(0xffb8b3635ef2cc87), fb(0x3fefffffffffffff)),
              (-1e300, fb(0x3fefffffffffe000)), (fb(0x3feffffffff586ae), fb(0x3fda2f60c07f708d)),
              (fb(0x3fefffff574744dd), fb(0x3febf480bef14e99))]
    top = 38 if single else 300
    for _ in range(50):
        m = -(10.0 ** g.uniform(0, top))
        cs.append((m * g.choice([1.0, g.uniform(0, 1), 1 - 10.0 ** -g.uniform(0, 8)]), m))
    for _ in range(25):
        cs.append((g.choice([g.uniform(1, 100), 10.0 ** g.uniform(2, top)]), -(10.0 ** g.uniform(1, top))))
    for _ in range(20):
        cs.append((-(10.0 ** g.uniform(top / 3, top)), 1 - 10.0 ** -g.uniform(1, 7 if single else 15.9)))
    for _ in range(25):
        cs.append((1 - 10.0 ** -g.uniform(1, 7 if single else 12), g.uniform(0, 0.99)))
    return [(n, m) for n, m in edge_args(cs, single) if n != 1.0 and m < 1.0 and math.isfinite(n) and math.isfinite(m)]


def complete(single):
    sfx = "f32" if single else "f64"
    n = 120 if single else 350
    xs = m_args(n, single)
    edges = ke_edges(single)
    write(f"elliptic_k_{sfx}.txt", f"K(m), {sfx}", [((x,), [R(mpmath.ellipk, x)]) for x in xs]
          + [((x,), [RP(mpmath.ellipk, x)]) for x in edges])
    xs = m_args(n, single)
    write(f"elliptic_e_{sfx}.txt", f"E(m), {sfx}", [((x,), [R(mpmath.ellipe, x)]) for x in xs]
          + [((x,), [RP(mpmath.ellipe, x)]) for x in edges])
    rows = []
    cnt = 80 if single else 350
    for i in range(cnt):
        kind = i % 6
        if kind == 0:
            nn = rng.uniform(-10, 1)
        elif kind == 1:
            nn = 1 + rng.choice([-1, 1]) * 10.0 ** -rng.uniform(1, 12)
        elif kind == 2:
            nn = rng.uniform(1, 20)
        elif kind == 3:
            nn = rng.choice([-1, 1]) * 10.0 ** rng.uniform(1, 12)
        elif kind == 4:
            nn = rng.uniform(-1, 1)
        else:
            nn = -(10.0 ** rng.uniform(-6, 3))
        if i % 3 == 0:
            m = 1 - 10.0 ** -rng.uniform(1, 14)
        elif i % 3 == 1:
            m = rng.uniform(-1, 1)
        else:
            m = -(10.0 ** rng.uniform(-3, 8))
        if single:
            nn, m = to_f32(nn), to_f32(m)
        if nn == 1.0 or m >= 1.0:
            continue
        rows.append(((nn, m), [R(pi_complete, nn, m)]))
    rows += [((nn, m), [RP(pi_complete, nn, m)]) for nn, m in pi_edges(single)]
    write(f"elliptic_pi_{sfx}.txt", f"Pi(n, m), {sfx}; the principal value for n > 1", rows)


# ------------------------------------------------------------------------------------------
# Incomplete integrals.


def phi_m(i, single):
    kind = i % 5
    if kind == 0:
        phi = rng.uniform(-1.6, 1.6)
    elif kind == 1:
        phi = rng.uniform(-20, 20)
    elif kind == 2:
        phi = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-10, 0)
    elif kind == 3:
        phi = rng.choice([-1, 1]) * 10.0 ** rng.uniform(1.3, 8)
    else:
        phi = rng.choice([-1, 1]) * (math.pi / 2 * rng.randint(1, 7) + rng.uniform(-1e-3, 1e-3))
    j = (i // 5) % 6
    if j == 0:
        m = rng.uniform(0, 1)
    elif j == 1:
        m = 1 - 10.0 ** -rng.uniform(1, 15)
    elif j == 2:
        m = rng.uniform(-5, 0)
    elif j == 3:
        m = -(10.0 ** rng.uniform(1, 10))
    elif j == 4:
        m = 10.0 ** rng.uniform(-12, -2)
    else:
        # m > 1 with |sin phi| <= 1/sqrt(m): the real domain.
        m = 10.0 ** rng.uniform(0.001, 6)
        phi = rng.choice([-1, 1]) * math.asin(rng.uniform(0, 1) / math.sqrt(m))
    if single:
        phi, m = to_f32(phi), to_f32(m)
    return phi, m


def periodic(f, complete_f, phi, *a):
    """f(phi) by the periodicity phi = j pi + r, |r| <= pi/2, for references of large phi."""
    p = mpmath.mpf(phi)
    j = mpmath.nint(p / mp.pi)
    r = p - j * mp.pi
    if abs(r) > mp.pi / 2:
        j += mpmath.sign(r)
        r = p - j * mp.pi
    v = re(f(r, *a))
    if j != 0:
        v += 2 * j * re(complete_f(*a))
    return v


def real_domain(phi, m):
    if m <= 1:
        return True
    return abs(math.sin(phi)) * math.sqrt(m) <= 1 and abs(phi) <= math.pi / 2


def fe_edges(single):
    """F and E: F(phi, 1) next to pi/2 (atanh of the rounded sine: inf and 3e-4 off), F(phi, 0) =
    phi, the cases above the bounds at ordinary arguments, huge phi (j = round(phi/pi) inexact),
    and m > 1 next to the edge of the real domain."""
    g = edge_rng("fe", single)
    cs = [(1.4681898451769606, 1.0), (1.5697664749281566, 1.0), (fb(0x3ff921fb3748bfac), 1.0),
          (fb(0x3ff921fb5173e808), 1.0), (fb(0x3ff921fb54442d18), 1.0), (-1.5697664749281566, 1.0)]
    if single:
        cs += [(fb(0x3ff921fb40000000), 1.0), (fb(0x3ff921fb20000000), 1.0), (fb(0x3ff921fa00000000), 1.0)]
    cs += [(g.choice([-1, 1]) * (math.pi / 2 - 10.0 ** -g.uniform(0, 7 if single else 16)), 1.0) for _ in range(30)]
    cs += [(0.030236235303514835, 0.0), (1.0, 0.0), (-7.0, 0.0), (100.0, 0.0), (1e10, 0.0)]
    cs += [(0.8849593685741823, 0.4547545860480865), (0.44982593853599084, 1 - 2.0 ** -53),
           (0.9695491222622425, 0.238310933353711), (1.2949168839446277, 0.9999975702548638),
           (1.1639810355173312, 1.185631818720106)]
    big = []
    if not single:
        big = [(2.697036017939415e159, -3.8093325253644044)]
        big += [(g.choice([-1, 1]) * 10.0 ** g.uniform(16, 300), g.choice([g.uniform(-5, 1), g.uniform(0, 1)])) for _ in range(30)]
    edge = [(0.009226445537398174, 11747.446571201079), (0.11927222986469045, 70.62878004813031)]
    for _ in range(30):
        m = 10.0 ** g.uniform(0.5, 6)
        edge.append((g.choice([-1, 1]) * math.asin((1 - 10.0 ** -g.uniform(5, 7 if single else 15)) / math.sqrt(m)), m))
    norm = [x for x in edge_args(cs + edge, single) if real_domain(x[0], x[1]) and not (x[1] == 1.0 and abs(x[0]) >= math.pi / 2)]
    return norm, edge_args(big, single)


def piinc_edges(single):
    """Pi(n; phi, m): n < -1.34e154 ((m - n)(1 - n) overflowed), the pole taken for n > 1, m << 0
    at tiny phi, m <= n < 0 (cancellation and underflow), n > 1 with m << -1, the plain region
    above 2 ulps, m > 1 next to the edge."""
    g = edge_rng("piinc", single)
    cs = []
    if not single:
        cs = [(-1.4e154, 1.0, 0.5), (-1e200, 1.0, 0.5), (-1e300, 0.3, -2.0),
              (fb(0xfb84277828ec48b1), 0.09882693020365307, 0.6737772720830277),
              (fb(0x58a3d36620337efd), -2.813572394067796e-10, -6.2008343425823615e267),
              (-3.0026094379355506, 0.5821696649741286, 0.9999997125800087),
              (-1e10, 1.0, -1e10), (-1e16, 1.0, -1e16), (-1e50, 1.0, -1e50), (-1e200, 1.0, -1e200), (-1e250, 1.0, -1e250),
              (5.953594985648745, -1.570727412138047, -8.088877702198222e266)]
    top = 38 if single else 300
    for _ in range(50):
        m = -(10.0 ** g.uniform(0, top))
        n = m * g.choice([1.0, g.uniform(0, 1), 1 - 10.0 ** -g.uniform(0, 8)])
        cs.append((n, g.uniform(-1.5, 1.5), m))
    for _ in range(25):
        cs.append((-(10.0 ** g.uniform(10, top)), g.uniform(-1.5, 1.5), g.uniform(-5, 1)))
    for _ in range(25):
        cs.append((g.choice([g.uniform(1, 100), 10.0 ** g.uniform(2, top / 2)]), g.choice([1, -1]) * 10.0 ** -g.uniform(0, 12),
                   -(10.0 ** g.uniform(1, top))))
    for _ in range(20):
        cs.append((g.uniform(-5, 1), g.uniform(-1.5, 1.5), 1 - 10.0 ** -g.uniform(1, 7)))
    for _ in range(15):
        m = 10.0 ** g.uniform(0.5, 4)
        phi = g.choice([-1, 1]) * math.asin((1 - 10.0 ** -g.uniform(5, 7 if single else 15)) / math.sqrt(m))
        cs.append((g.uniform(-5, 0.9), phi, m))
    out = []
    for n, phi, m in edge_args(cs, single):
        if not all(math.isfinite(v) for v in (n, phi, m)) or not real_domain(phi, m) or n * math.sin(phi) ** 2 == 1.0:
            continue
        out.append((n, phi, m))
    return out


def huge_edges(single):
    """F, E and Pi at |phi| > 2^52, where the reduction goes without j = round(phi/pi), found by
    the second pass of the fixes: results at and beyond the largest float (NaN where the
    double-double product (2C/pi) phi overflowed), and Pi(1; phi, m), whose complete integral is
    infinite (NaN there too). F of 0 <= m < 1 (2K/pi >= 1) and E of m < 0 near the top of the range;
    phi of E chosen around MAX/(2E(m)/pi)."""
    g = edge_rng("huge", single)
    big = 3.4028234663852886e38 if single else 1.7976931348623157e308
    if single:
        fe = [(3e38, -1e30), (3.4e38, 0.9), (-3.4e38, 0.999), (3.4e38, -0.5)]
        fe += [(g.choice([-1, 1]) * big * g.uniform(0.5, 1), g.uniform(-1, 1)) for _ in range(10)]
        pi = [(1.0, 1e20, 0.5), (1.0, -3e30, -2.0), (0.9, 3.4e38, 0.5), (1.0, 1e10, 0.3)]
        pi += [(1.0, g.choice([-1, 1]) * 10.0 ** g.uniform(8, 38), g.uniform(-3, 1)) for _ in range(6)]
    else:
        fe = [(-3.2099911516551386e239, -5.351869541041468e219), (4.288610545264101e298, -1.3191806702289343e132),
              (-3.4186556848142217e189, -6.47846742445255e299), (1.7e308, -1.0), (1e300, -1e20), (1.7e308, 0.9),
              (-1.7e308, 0.5), (big, 1e-4), (1e308, 0.9999999), (1.5e308, -0.1)]
        fe += [(g.choice([-1, 1]) * big * g.uniform(0.3, 1), g.uniform(-1, 1)) for _ in range(15)]
        for _ in range(15):
            m = -(10.0 ** g.uniform(0, 300))
            ge = float(2 * mpmath.ellipe(m) / mpmath.pi)
            phi = min(big, big / ge * 2.0 ** g.uniform(-2, 2))
            if phi > 2.0 ** 53:
                fe.append((g.choice([-1, 1]) * phi, m))
        pi = [(1.0, 1e16, 0.5), (1.0, -1e16, 0.5), (1.0, 1e300, -3.0), (0.9, 1.7e308, 0.5), (-0.5, -1.7e308, -2.0),
              (0.999999999, 1e306, 0.9)]
        pi += [(1.0, g.choice([-1, 1]) * 10.0 ** g.uniform(16, 308), g.uniform(-3, 1)) for _ in range(8)]
        pi += [(g.uniform(0, 1), g.choice([-1, 1]) * big * g.uniform(0.3, 1), g.uniform(0, 1)) for _ in range(8)]
    fe = [x for x in edge_args(fe, single) if all(math.isfinite(v) for v in x)]
    pi = [x for x in edge_args(pi, single) if all(math.isfinite(v) for v in x)]
    return fe, pi


def inside_gt1(phi, m):
    """Whether (phi, m), m > 1, is inside the real domain (1 - m sin^2 phi >= 0) at 300 bits: the
    float test of real_domain is not exact next to the edge."""
    with mpmath.workprec(300):
        return 1 - mpmath.mpf(m) * mpmath.sin(mpmath.mpf(phi)) ** 2 >= 0


def review1_inc_edges(single):
    """Found by the review of the fixes (round 1), after all the other rows: F for m > 1 next to
    the edge as m -> 1+, where the rounding of sin(phi) moved the edge by eps tan(phi) (11.6 (1 +
    kappa) at (1.558, 1.00016)); E for m > 1 there (s R_F - (m/3) s^3 R_D cancelled: 9.75 ulps
    at (1.539, 1.001)), for m < 0 (4.56 ulps at (0.553, -43155): the factor (-m/3) s^3 rounded
    step by step), and for tiny phi with huge -m (s^3 underflowed: 30% off); Pi for n << -1
    past a period (2j Pi + Pi(r) cancelled: 7.39 ulps at (-7.7e14; -2.96, -2.54)). Inside the
    real domain only (the edge itself is a special value of the tests). Returns the rows (phi, m)
    of F and E and (n, phi, m) of Pi."""
    g = edge_rng("review1-inc", single)
    fe = [(1.5580331674262458, 1.0001623607178978), (1.4781097127276868, 1.0086402502548162),
          (1.5388592166738542, 1.0010206729751223), (1.131041710276835, 1.2213521765706306),
          (1.2043103948119291, 1.1473200577234237), (1.2380362140272425, 1.1194474684521418),
          (1.5286148656290393, 1.0017813876253971), (0.553147177896637, -43154.989512059175),
          (0.9326418656869799, -1118.5121193039329), (-0.6798475573834707, -19160.639518983415)]
    if not single:
        fe += [(1e-150, -1e300), (3e-160, -1.7e308), (-2e-100, -1e200), (1e-20, -1e40)]
    # m > 1: m - 1 from 1e-1 to 1e-6 (1e-9 in f64) next to the edge, and m in (1, 3) across it.
    for _ in range(40):
        m = 1 + 10.0 ** -g.uniform(1, 6 if single else 9)
        e = math.asin(1 / math.sqrt(m))
        fe.append((g.choice([-1, 1]) * e * (1 - 10.0 ** -g.uniform(2, 6 if single else 12)), m))
    for _ in range(20):
        m = g.uniform(1, 3)
        fe.append((g.choice([-1, 1]) * math.asin(1 / math.sqrt(m)) * g.uniform(0.3, 1), m))
    # m < 0: uniform phi in (0, pi/2), -m to 1e8.
    for _ in range(25):
        fe.append((g.choice([-1, 1]) * g.uniform(0, math.pi / 2), -(10.0 ** g.uniform(0, 8))))
    fe = [x for x in edge_args(fe, single)
          if all(math.isfinite(v) for v in x) and (x[1] <= 1 or inside_gt1(x[0], x[1]))]
    pi = [(-768834530296985.2, -2.962973998308219, -2.5363981024125755), (-3.2, 3.5, 0.5), (-1e6, -1.8, 0.9)]
    if not single:
        pi.append((-3.0437272439429716e286, -2.66486964849815, -3.117815390554317))
    for _ in range(30):
        n = -(10.0 ** g.uniform(0, 300 if not single else 30))
        j = g.choice([-3, -2, -1, 1, 2, 3])
        r = -math.copysign(1, j) * g.uniform(0.05, math.pi / 2)
        pi.append((n, j * math.pi + r, g.uniform(-5, 1)))
    pi = [x for x in edge_args(pi, single) if all(math.isfinite(v) for v in x) and x[2] < 1.0]
    return fe, pi


def review2_inc_edges(single):
    """Found by the review of the fixes (round 2), after all the other rows: F(phi, 1) for |phi|
    in [0.3, 0.9), where s R_C(1, c^2) added the errors of R_C, of the product and of c^2 to those
    of the sine (3.04 ulps at 0.456); Pi for m > 1 next to 1 with n at or above m and phi next to
    the pole n sin^2 phi = 1, where the kernel at N = n/m and mu = 1/m formed N - 1, 1 - mu/N and
    N s_beta^2 - 1 from the rounded N and mu (903 (1 + kappa) at (1 + 1.6e-8; 1.5707, 1 + 1.0e-8)),
    and n next to m on either side. Returns the rows (phi, m) of F and E and (n, phi, m) of Pi."""
    g = edge_rng("review2-inc", single)
    fe = [(0.45609311495447225, 1.0), (-0.4628593830017078, 1.0), (-0.44427969932221667, 1.0),
          (-0.36416885731778037, 1.0), (0.3750497686943257, 1.0), (0.8360758843213079, 1.0),
          (0.7983540726687103, 1.0), (-0.7838354860573052, 1.0), (-0.856052152228683, 1.0)]
    for _ in range(40):
        fe.append((g.choice([-1, 1]) * g.uniform(0.3, 0.9), 1.0))
    for _ in range(10):
        fe.append((g.choice([-1, 1]) * g.uniform(0, math.pi / 2 * (1 - 1e-9)), 1.0))
    fe = [x for x in edge_args(fe, single) if abs(x[0]) < math.pi / 2]
    pi = [] if single else [(1.0000000164781473, 1.5706532227277117, 1.0000000101535529),
                            (1.0000283455859615, 1.554431755857697, 1.0000000028005664),
                            (1.0007614571586023, -1.5402790260070482, 1.0000115336589224)]
    for k in range(60):
        if single:
            m = 1 + 10.0 ** -g.uniform(0, 6)
            n = 1 + 10.0 ** -g.uniform(-1, 6)
            s2 = g.uniform(0, 1) / m
        elif k < 30:
            # n >= m, phi next to the pole.
            m = 1 + 10.0 ** -g.uniform(1, 12)
            n = m * (1 + 10.0 ** -g.uniform(0.5, 12))
            s2 = (1 / n) * (1 + g.choice([1, -1]) * 10.0 ** -g.uniform(1, 9))
        elif k < 45:
            # n next to m on either side, phi anywhere inside.
            m = 1 + 10.0 ** -g.uniform(1, 12)
            n = m * (1 + g.choice([1, -1]) * 10.0 ** -g.uniform(1, 14))
            s2 = g.uniform(0, 1) / m
        else:
            m = 1 + 10.0 ** -g.uniform(-1, 6)
            n = 1 + 10.0 ** -g.uniform(-1, 6)
            s2 = g.uniform(0, 1) / m
        if not 0 < s2 <= 1 / m:
            continue
        pi.append((n, g.choice([-1, 1]) * math.asin(math.sqrt(s2)), m))
    pi = [x for x in edge_args(pi, single) if all(math.isfinite(v) for v in x) and x[2] > 1
          and inside_gt1(x[1], x[2]) and x[0] * math.sin(x[1]) ** 2 != 1.0]
    return fe, pi


def incomplete(single):
    sfx = "f32" if single else "f64"
    cnt = 100 if single else 350
    rows_f, rows_e, rows_p = [], [], []
    for i in range(cnt):
        phi, m = phi_m(i, single)
        if not real_domain(phi, m) or (m == 1.0 and abs(phi) >= math.pi / 2):
            continue
        if m > 1:
            vf = R(mpmath.ellipf, phi, m)
            ve = R(mpmath.ellipe, phi, m)
        else:
            vf = R(lambda p, mm: periodic(lambda r, q: mpmath.ellipf(r, q), mpmath.ellipk, p, mm), phi, m)
            ve = R(lambda p, mm: periodic(lambda r, q: mpmath.ellipe(r, q), mpmath.ellipe, p, mm), phi, m)
        rows_f.append(((phi, m), [vf]))
        rows_e.append(((phi, m), [ve]))
    for i in range(cnt + cnt // 5):
        phi, m = phi_m(i, single)
        if not real_domain(phi, m) or abs(phi) > 1e4 or (m == 1.0 and abs(phi) >= math.pi / 2):
            continue
        kind = i % 7
        if kind == 0:
            n = rng.uniform(-5, 1)
        elif kind == 1:
            n = rng.uniform(1, 10)
        elif kind == 2:
            n = -(10.0 ** rng.uniform(-3, 8))
        elif kind == 3:
            n = 10.0 ** rng.uniform(1, 8)
        elif kind == 4:
            n = rng.uniform(0, 1)
        elif kind == 5:
            n = 1 + rng.choice([-1, 1]) * 10.0 ** -rng.uniform(1, 10)
        else:
            n = rng.uniform(-1, 0)
        if single:
            n = to_f32(n)
        if n * math.sin(phi) ** 2 == 1.0:
            continue
        if m > 1:
            v = R(pi_incomplete, n, phi, m)
        else:
            if n == 1.0 and abs(phi) > math.pi / 2:
                continue
            v = R(lambda nn, p, mm: periodic(lambda r, a, b: pi_incomplete(a, r, b), pi_complete, p, nn, mm), n, phi, m)
        rows_p.append(((n, phi, m), [v]))
    # The edges, at the precisions of RP (the reduction of a huge phi takes its bits).
    norm, big = fe_edges(single)
    for phi, m in norm + big:
        if m > 1:
            vf, ve = RP(mpmath.ellipf, phi, m), RP(mpmath.ellipe, phi, m)
        else:
            vf = RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipf(r, q), mpmath.ellipk, p, mm), phi, m)
            ve = RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipe(r, q), mpmath.ellipe, p, mm), phi, m)
        rows_f.append(((phi, m), [vf]))
        rows_e.append(((phi, m), [ve]))
    for n, phi, m in piinc_edges(single):
        if m > 1:
            v = RP(pi_incomplete, n, phi, m)
        else:
            v = RP(lambda nn, p, mm: periodic(lambda r, a, b: pi_incomplete(a, r, b), pi_complete, p, nn, mm), n, phi, m)
        rows_p.append(((n, phi, m), [v]))
    # The huge-phi edges of the second pass (last, so that the rows before stay as they were).
    fe, pi = huge_edges(single)
    for phi, m in fe:
        rows_f.append(((phi, m), [RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipf(r, q), mpmath.ellipk, p, mm), phi, m)]))
        rows_e.append(((phi, m), [RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipe(r, q), mpmath.ellipe, p, mm), phi, m)]))
    for n, phi, m in pi:
        v = RP(lambda nn, p, mm: periodic(lambda r, a, b: pi_incomplete(a, r, b), pi_complete, p, nn, mm), n, phi, m)
        rows_p.append(((n, phi, m), [v]))
    # The review's edges (round 1), last.
    fe, pi = review1_inc_edges(single)
    for phi, m in fe:
        if m > 1:
            vf, ve = RP(mpmath.ellipf, phi, m), RP(mpmath.ellipe, phi, m)
        else:
            vf = RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipf(r, q), mpmath.ellipk, p, mm), phi, m)
            ve = RP(lambda p, mm: periodic(lambda r, q: mpmath.ellipe(r, q), mpmath.ellipe, p, mm), phi, m)
        rows_f.append(((phi, m), [vf]))
        rows_e.append(((phi, m), [ve]))
    for n, phi, m in pi:
        v = RP(lambda nn, p, mm: periodic(lambda r, a, b: pi_incomplete(a, r, b), pi_complete, p, nn, mm), n, phi, m)
        rows_p.append(((n, phi, m), [v]))
    # The review's edges (round 2), last.
    fe, pi = review2_inc_edges(single)
    for phi, m in fe:
        rows_f.append(((phi, m), [RP(lambda p, mm: mpmath.ellipf(p, mm), phi, m)]))
        rows_e.append(((phi, m), [RP(lambda p, mm: mpmath.ellipe(p, mm), phi, m)]))
    for n, phi, m in pi:
        rows_p.append(((n, phi, m), [RP(pi_incomplete, n, phi, m)]))
    write(f"elliptic_f_{sfx}.txt", f"F(phi, m), {sfx}", rows_f)
    write(f"elliptic_einc_{sfx}.txt", f"E(phi, m), {sfx}", rows_e)
    write(f"elliptic_piinc_{sfx}.txt", f"Pi(n; phi, m), {sfx}; principal values past a pole", rows_p)


# ------------------------------------------------------------------------------------------
# Carlson's integrals.


def carlson_arg(single):
    k = rng.random()
    if k < 0.6:
        x = 10.0 ** rng.uniform(-3, 3)
    elif k < 0.85:
        x = 10.0 ** rng.uniform(-30, 30)
    elif k < 0.95:
        x = 10.0 ** rng.uniform(-300, 300) if not single else 10.0 ** rng.uniform(-35, 35)
    else:
        x = 0.0
    return to_f32(x) if single else x


def rj_ref(x, y, z, p):
    """R_J; for p < 0 the principal value by Carlson's formula (the form of Boost.Math), from
    R_J of a positive p (mpmath's complex continuation is slow; its real part agrees)."""
    if p > 0:
        return mpmath.elliprj(x, y, z, p)
    x, y, z = sorted([x, y, z])
    q = -p
    pn = (z * (x + y + q) - x * y) / (z + q)
    v = (pn - z) * mpmath.elliprj(x, y, z, pn) - 3 * mpmath.elliprf(x, y, z)
    v += 3 * mpmath.sqrt(x * y * z / (x * y + pn * q)) * mpmath.elliprc(x * y + pn * q, pn * q)
    return v / (z + q)


def rc_ref(x, y):
    """R_C; for y < 0 the principal value sqrt(x/(x - y)) R_C(x - y, -y) (DLMF 19.2.20)."""
    if y > 0:
        return mpmath.elliprc(x, y)
    return mpmath.sqrt(x / (x - y)) * mpmath.elliprc(x - y, -y)


def carlson_edges(single):
    """The arguments at which the audit found NaN or wrong values: all subnormal, near f64::MAX,
    results beyond f64::MAX (inf), R_J with p far above x, y, z or x, y, z tiny with p = O(1),
    the principal values with tiny |p| or tiny x, and R_J above its bound; neighbours of each."""
    g = edge_rng("carlson", single)
    M = 3.4028234663852886e38 if single else 1.7976931348623157e308
    if single:
        sub = [1e-40, 1e-44, 1.4e-45]
        rf = [(s, s, s) for s in sub] + [(0.0, 1e-40, 1e-40), (M, M, M), (M, M, 1.0)]
        rd = [(1.4e-45, 1.0, 1.4e-45), (1e-40, 1e-40, 1e-40)]
        rj = [(1e-30, 1e-30, 1e-30, 1e30), (1e-40, 9.7e-41, 9e-41, 168.26), (1e-40, 1e-40, 1e-40, 1e-40),
              (2.4677303e-18, 0.77427024, 0.3849121, 9.4054e-20)]
        rc = [(1e-40, 1e-40), (M, -M), (M, 1e-38), (1e-44, -1e10)]
    else:
        sub = [1e-308, 1e-310, 5e-324, 2.2e-308, 1e-320]
        rf = [(s, s, s) for s in sub] + [(0.0, 1e-310, 1e-310), (M, M, M), (M, M, 1.0), (M, 1.0, 1.0), (1e308, 1e308, 1e-300)]
        rf += [(g.choice(sub) * g.uniform(1, 4), g.choice(sub) * g.uniform(1, 4), g.choice([0.0, 1e-315])) for _ in range(10)]
        rf += [(M * g.uniform(0.5, 1), M * g.uniform(0.5, 1), 10.0 ** g.uniform(-300, 308)) for _ in range(10)]
        rd = [(5e-324, 1.0, 5e-324), (1e-310, 1e-310, 1e-310), (0.0, 5e-28, 2.08e-299), (M, M, M), (1e-320, 1.0, 1e-300)]
        rd += [(10.0 ** g.uniform(-323, -300), 10.0 ** g.uniform(-30, 30), 10.0 ** g.uniform(-323, -300)) for _ in range(10)]
        rj = [(1e-200, 1e-200, 1e-200, 1e200), (1e-300, 1e-300, 1e-300, 1e300), (1e-150, 1e-150, 1e-150, 1e150),
              (1e-310, 1e-310, 1e-310, 1e-310), (1e-250, 9.7e-251, 9e-251, 168.26197780983787),
              (fb(0x06e104ad756abe47), 1.482e-275, 1.392e-275, 168.26), (6.705207097837259e-275, 6.705207098076587e-275, 0.0, -122.52229384949973),
              (4.89e-311, 4.89e-311, 4.89e-311, 0.0033), (1.2957424016149786e18, 4.1263908360237943e21, 1.1048122852954912e257, -7.253235289612251e-307),
              (374.9, 22.06, 947.5, -1e-323), (2.467730310254843e-18, 0.7742702565559161, 0.38491209908293345, 9.405419401014528e-20),
              (0.010283565351136153, 4.385185900893153e51, 0.3607479546816487, 19.786045495366213), (M, M, M, M)]
        for _ in range(15):
            e = g.uniform(-300, 0)
            x = 10.0 ** e
            p = 10.0 ** min(e + g.uniform(12, 600), 308.0)
            rj.append((x, x * g.uniform(0.5, 2), x * g.uniform(0.5, 2), g.choice([1, -1]) * p))
        for _ in range(10):
            rj.append((10.0 ** g.uniform(-3, 3), 10.0 ** g.uniform(-3, 3), 10.0 ** g.uniform(-3, 300), -(10.0 ** g.uniform(-323, -250))))
        rj = [(x, y, z, p) for x, y, z, p in rj if math.isfinite(p) and p != 0.0]
        rc = [(1e-310, 1e-310), (M, -M), (M, 1e-308), (1.53e-322, -1004047202555.493), (1e-300, -1e10), (5e-324, 5e-324)]
        rc += [(10.0 ** g.uniform(-323, -290), -(10.0 ** g.uniform(0, 30))) for _ in range(10)]
    return {k: edge_args(v, single) for k, v in (("rf", rf), ("rd", rd), ("rj", rj), ("rc", rc))}


def pv_edges(single):
    """R_J principal values found by the second pass of the fixes, where xy is far below
    max(y, -p)^2 (the rho = p'q/(xy) form): rho or sqrt(z)/(sqrt(x) sqrt(y)) beyond the range (NaN),
    rho < 2^-60 with ln q - ln x cancelling, V = p'q < 2^-60 W with ln q - ln s cancelling, and
    (z + q) R_J beyond the range with R_J inside it (spreads beyond 2^1500). f64 only: the f32
    arguments, evaluated in f64, cannot reach these forms."""
    if single:
        return []
    g = edge_rng("carlson-pv", single)

    def lu(a, b):
        return 10.0 ** g.uniform(a, b)
    rj = [(2.556170558e-314, 2.5570641875e-314, 0.0033016743001422644, -0.0033016743001422644),
          (2.494427078406156e-278, 1.0041192759264211e-135, 5.105809125734171e228, -5.105809111706431e228),
          (1e-310, 1e-310, 1.0, -1.0), (1e-300, 1e-200, 1e200, -1e200),
          (6.927506611993012e-304, 0.11121623492539896, 2.4082611739525288e134, -3.2e-322),
          (2.164225721491554e-273, 1.7069439469956117e-271, 1.5246847312690459e212, -6.883237039464134e-294)]
    for _ in range(8):
        x, z = lu(-323.3, -308), lu(-10, 10)
        rj.append((x, x * g.uniform(0.5, 2), z, -z * g.choice([1.0, g.uniform(0.5, 2)])))
    for _ in range(8):
        z = lu(150, 300)
        rj.append((lu(-300, -200), lu(-200, -100), z, -z * g.choice([1 - 10.0 ** -g.uniform(1, 12), g.uniform(0.5, 2)])))
    for _ in range(8):
        rj.append((lu(-323, -295), lu(-3, 3), lu(30, 300), -lu(-250, -10)))
    for _ in range(8):
        x = lu(-323, -300)
        rj.append((x, lu(-5, 5), lu(-5, 300), -(x * lu(-15, 15))))
    for _ in range(8):
        rj.append((lu(-305, -295), lu(-3, 3), lu(-3, 300), -lu(-323.3, -320)))
    for _ in range(8):
        x = lu(-290, -230)
        y = x * lu(0, 20)
        rj.append((x, y, 10.0 ** g.uniform(math.log10(y), 300), -(x * lu(-40, -20))))
    return [r for r in rj if all(math.isfinite(v) for v in r) and r[0] != 0.0 and r[3] != 0.0]


def rc_sub_edges(single):
    """R_C principal values whose quotient x/(x - y) is below the normal range, found by the second
    pass of the fixes (sqrt(x)/sqrt(x - y) rounded three times: 2.46 ulps at (5e-323, -0.1087)), and
    the ordinary one at 2.41 ulps. f64 only (f32 arguments cannot reach the subnormal quotient)."""
    if single:
        return []
    g = edge_rng("rc-sub", single)
    cs = [(5e-323, -0.10867995258102492), (471.013362889908, -441.016219541716)]
    for _ in range(20):
        cs.append((10.0 ** g.uniform(-323.3, -300), -(10.0 ** g.uniform(-15, 25))))
    return [c for c in cs if c[0] != 0.0]


def review1_carlson_edges(single):
    """Found by the review of the fixes (round 1), after all the other rows: R_D with x, y tiny and
    z huge (spans beyond 2^1600, no scaling helps: subnormal results 20-99% off) and across the
    switch to its expansion for z > 2^120 max(x, y); R_J with x, y tiny and z huge, p next to y
    or |p| far above z (the duplication overflowed: inf or NaN for finite values, also inside
    DLMF 19.21.10); principal values with V = p'q far below W where two large logarithms
    cancelled (5.98 scaled ulps at (7.5e-215, 1.5e100, 9.5e66, -3.0e-285)). f64 only: f32
    arguments, evaluated in f64, span far less. Returns the rows of R_D and R_J."""
    if single:
        return [], []
    g = edge_rng("review1-carlson", single)

    def lu(a, b):
        return 10.0 ** g.uniform(a, b)
    rd = [(1e-316, 1e-316, 1e212), (2.67953697e-316, 6.219496945784042e-290, 3.858395444207167e210),
          (1e-316, 1e-300, 1e210), (1.0, 1.0, 1e40), (2.0, 0.0, 1e300)]
    for _ in range(12):
        rd.append((lu(-323.3, -290), lu(-323.3, -250), lu(150, 308.2)))
    for _ in range(12):
        z = lu(-280, 300)
        rd.append((z * lu(-40, -33), z * lu(-40, -30) if g.random() < 0.7 else 0.0, z))
    rj = [(1.7474934e-317, 1.974386793004844e-309, 5.7516244720367435e194, 7.674401530947324e206),
          (1.7474934e-317, 1.974386793004844e-309, 5.7516244720367435e194, -7.674401530947324e206),
          (0.0, 1e-310, 1e200, 1e300), (0.0, 1e-310, 1e200, -1e300), (0.0, 1e-310, 1e200, 1e-310),
          (1e-317, 2e-309, 1e195, 2e-309),
          (7.506553953259573e-215, 1.5166929453536658e100, 9.529002958020896e66, -2.96941174047871e-285)]
    for _ in range(12):
        x = 0.0 if g.random() < 0.3 else lu(-323.3, -295)
        y, z = lu(-323, -290), lu(150, 308)
        p = y * g.uniform(0.5, 2) if g.random() < 0.5 else lu(math.log10(z) + 13, 308.2) * g.choice([-1, 1])
        rj.append((x, y, z, p))
    for _ in range(10):
        rj.append((lu(-300, -150), lu(50, 150), lu(30, 100), -lu(-300, -250)))
    rd = [r for r in rd if all(math.isfinite(v) for v in r) and r[2] > 0 and (r[0] > 0 or r[1] > 0)]
    rj = [r for r in rj if all(math.isfinite(v) for v in r) and r[3] != 0.0 and [r[0], r[1], r[2]].count(0.0) <= 1]
    return rd, rj


def review2_carlson_edges(single):
    """Found by the review of the fixes (round 2), after all the other rows: R_J by DLMF 19.21.10
    (|p| above 2^40 max(x, y, z)) with a subnormal middle argument y, where y - q of the rounded q
    was 0 and the term (y - q) R_J(q) was lost (18.8 ulps at (0, 2e-323, 2.69, 3.24e12)), either
    sign of p, and spans beyond 2^1600 with p > 0 (6.3 ulps at (0, 1.3e-316, 1.1e176, 1.5e188));
    and of normal arguments, where its terms rounded step by step took it to 2.9 ulps. f64 only.
    Rows (x, y, z, p)."""
    if single:
        return []
    g = edge_rng("review2-carlson", single)

    def lu(a, b):
        return 10.0 ** g.uniform(a, b)
    rj = [(0.0, 2e-323, 2.6869540121305318, 3240599136426.942),
          (1.7814088875e-314, 1.6820941e-316, 0.014950759359958696, 19753887867.89231),
          (0.0, 5.5534632734e-314, 0.1816639417048735, -219544660138.4224),
          (1.897202915e-315, 2e-323, 2761150751779.036, -3.408389090440692e24),
          (0.0, 1.30068695e-316, 1.10663688445477e176, 1.505687356031855e188),
          (1.455391127574754e-288, 3.3881311940567326e-92, 1.7639579891747828e-212, 5.57104378653199e-80),
          (2.424809104861835e63, 7.027309540128917e-234, 1.7592752459851372e-118, 4.578648402381279e75)]
    for k in range(50):
        if k < 40:
            y = lu(-323.3, -308)
            x = g.choice([0.0, lu(-323.3, -308), y * g.uniform(0, 1)])
            z = lu(-3, 300) if k < 30 else lu(150, 290)
        else:
            x, y, z = sorted([lu(-300, 300), lu(-300, 300), lu(-300, 300)])
        top = max(x, y, z)
        lp = min(math.log2(top) + (g.uniform(40, 60) if 30 <= k < 40 else g.uniform(40, 41)), 1023.9)
        rj.append((x, y, z, (1.0 if 30 <= k < 40 else g.choice([1.0, -1.0])) * 2.0 ** lp))
    return [r for r in rj if all(math.isfinite(v) for v in r) and r[3] != 0.0 and [r[0], r[1], r[2]].count(0.0) <= 1]


def carlson(single):
    sfx = "f32" if single else "f64"
    cnt = 80 if single else 250
    a = carlson_arg
    edges = carlson_edges(single)
    rows = []
    for _ in range(cnt):
        x, y, z = a(single), a(single), a(single)
        if [x, y, z].count(0.0) > 1:
            continue
        rows.append(((x, y, z), [R(mpmath.elliprf, x, y, z)]))
    rows += [((x, y, z), [RP(mpmath.elliprf, x, y, z)]) for x, y, z in edges["rf"]]
    write(f"elliptic_rf_{sfx}.txt", f"R_F(x, y, z), {sfx}", rows)
    rows = []
    for _ in range(cnt):
        x, y, z = a(single), a(single), a(single)
        if z == 0.0 or (x == 0.0 and y == 0.0):
            continue
        rows.append(((x, y, z), [R(mpmath.elliprd, x, y, z)]))
    rows += [((x, y, z), [RP(mpmath.elliprd, x, y, z)]) for x, y, z in edges["rd"]]
    r1_rd, r1_rj = review1_carlson_edges(single)
    rows += [((x, y, z), [RP(mpmath.elliprd, x, y, z)]) for x, y, z in r1_rd]
    write(f"elliptic_rd_{sfx}.txt", f"R_D(x, y, z), {sfx}", rows)
    rows = []
    for i in range(cnt + cnt // 3):
        x, y, z = a(single), a(single), a(single)
        p = a(single)
        if i % 4 == 0:
            p = -p
        if p == 0.0 or [x, y, z].count(0.0) > 1:
            continue
        rows.append(((x, y, z, p), [R(rj_ref, x, y, z, p)]))
    rows += [((x, y, z, p), [RP(rj_ref, x, y, z, p)]) for x, y, z, p in edges["rj"]]
    rows += [((x, y, z, p), [RP(rj_ref, x, y, z, p)]) for x, y, z, p in pv_edges(single)]
    rows += [((x, y, z, p), [RP(rj_ref, x, y, z, p)]) for x, y, z, p in r1_rj]
    rows += [((x, y, z, p), [RP(rj_ref, x, y, z, p)]) for x, y, z, p in review2_carlson_edges(single)]
    write(f"elliptic_rj_{sfx}.txt", f"R_J(x, y, z, p), {sfx}; the principal value for p < 0", rows)
    rows = []
    for i in range(cnt):
        x, y = a(single), a(single)
        if i % 5 == 0:
            y = -y
        if i % 7 == 0:
            y = x * (1 + rng.uniform(-0.1, 0.1))
            y = to_f32(y) if single else y
        if y == 0.0:
            continue
        rows.append(((x, y), [R(rc_ref, x, y)]))
    rows += [((x, y), [RP(rc_ref, x, y)]) for x, y in edges["rc"]]
    rows += [((x, y), [RP(rc_ref, x, y)]) for x, y in rc_sub_edges(single)]
    write(f"elliptic_rc_{sfx}.txt", f"R_C(x, y), {sfx}; the principal value for y < 0", rows)


# ------------------------------------------------------------------------------------------
# Jacobi elliptic functions: references sn, cn, dn.


def jacobi_edges(single):
    """The Jacobi functions next to odd multiples of K (where the reduction clamped |r| > K to K),
    at huge u (where the quotient u/(2K) misrounded j), and where the transformed parameter is
    next to 1 (above the bounds of the regions m < 0 and m > 1)."""
    g = edge_rng("jacobi", single)
    if single:
        cs = [(1e20, 0.5), (1e12, 0.5), (3e9, -3.0), (2e10, 4.0)]
    else:
        cs = [(fb(0x404377c0b3152fe1), 0.5), (fb(0x4067685e8e2ecee7), 0.5), (fb(0x40acfc01c17b8b49), 0.5),
              (fb(0x414c4a7b9ab5f115), 0.5), (3e16, 0.5), (1e18, 0.5), (1e20, 0.5), (1e30, 0.5), (1e16, 0.1),
              (fb(0x4073bbae55298768), 1e-300), (fb(0x40a88e53afb51c8f), 1e-300),
              (-0.045572976033048926, -50043.80632143492), (11.72980681435471, 1.0000000000000384),
              (-7.054141916822147, 1.0000000000026692), (91.80193029643138, 1.0000000000000508),
              (-27.385242763772183, 0.9999999999999978), (2.935842863778815, 7.6698146068500055),
              (8.987754054247182, 0.4281333962490431)]
    for _ in range(40):
        m = g.choice([g.uniform(0.01, 0.9), -g.uniform(0.1, 10), g.uniform(1.1, 10), 1 - 10.0 ** -g.uniform(1, 6)])
        with mpmath.workprec(120):
            # The quarter period in u: K(m), K(mu)/sqrt(1 - m) or K(1/m)/sqrt(m).
            if m < 0:
                kq = mpmath.ellipk(-m / (1 - m)) / mpmath.sqrt(1 - m)
            elif m > 1:
                kq = mpmath.ellipk(1 / m) / mpmath.sqrt(m)
            else:
                kq = mpmath.ellipk(m)
            u = float((2 * g.randint(0, 3000 if single else 10000) + 1) * kq * (1 + g.uniform(-1, 1) * 1e-15))
        cs.append((g.choice([-1, 1]) * u, m))
    for _ in range(25):
        m = g.choice([g.uniform(0.01, 0.9), -g.uniform(0.1, 10), g.uniform(1.1, 10)])
        cs.append((g.choice([-1, 1]) * 10.0 ** g.uniform(9 if single else 12, 30), m))
    return edge_args(cs, single)


def jacobi_agm(f, u, m):
    """sn, cn or dn (f = 0, 1, 2) of mpf arguments by the AGM (A&S 16.4) after the reduction of u
    modulo 4K at the working precision, with A&S 16.10-11 for m < 0 and m > 1: for the parameters
    next to 1 (and the transformed ones beyond -1e6), where the theta functions converge slowly."""
    def agm(u, m, m1):
        a, b, c = mpmath.mpf(1), mpmath.sqrt(m1), mpmath.sqrt(m)
        aa, cc = [a], [c]
        while abs(c) > mpmath.mpf(2) ** (8 - mp.prec) * a:
            a, b, c = (a + b) / 2, mpmath.sqrt(a * b), (a - b) / 2
            aa.append(a)
            cc.append(c)
        n = len(aa) - 1
        k4 = 2 * mp.pi / aa[n]
        u = u - k4 * mpmath.nint(u / k4)
        phi = mpmath.mpf(2) ** n * aa[n] * u
        for i in range(n, 0, -1):
            phi = (phi + mpmath.asin(cc[i] * mpmath.sin(phi) / aa[i])) / 2
        s, c = mpmath.sin(phi), mpmath.cos(phi)
        # dn = sqrt(m1 + m cn^2), without the cancellation of 1 - m sn^2 next to the zero of cn.
        return s, c, mpmath.sqrt(m1 + m * c * c)
    if m < 0:
        d = 1 - m
        s, c, dn = agm(u * mpmath.sqrt(d), -m / d, 1 / d)
        r = (mpmath.sqrt(1 / d) * s / dn, c / dn, 1 / dn)
    elif m > 1:
        s, c, dn = agm(u * mpmath.sqrt(m), 1 / m, (m - 1) / m)
        r = (s / mpmath.sqrt(m), dn, c)
    else:
        r = agm(u, m, 1 - m)
    return r[f]


def jacobi_near_one_edges(single):
    """Found by the second pass of the fixes: the transformed parameter mu within 2^-40 of 1, where
    the Landen steps lost up to 8K ulps of dn and the test |w| <= K/2 took K of the rounded mu
    (mu = 1 itself for m < -2^53): m < -1e6 by decades, m within 2^-40 of 1 on either side; the
    transformed argument uniform in [0, 3K], next to (j + 1/2)K, and huge."""
    g = edge_rng("jacobi-near-one", single)
    top = 38 if single else 308
    cs = [] if single else [
        (1.5871178057016533e-12, -4.615612169578259e26), (-7.0112966348461805e131, -5.9379406967016225e214),
        (-1.9386284872855772e-06, -4220362791379119.5), (1.195767012075013e-19, -2.152446584174481e42),
        (-2.530870797838623e-25, -1.463282907362075e53), (5.226925197899624e-25, -8.892131862233403e52),
        (1e300, -1e300), (1.7e308, -1.7e308), (-1e200, -1e250), (1e160, -1e160), (3e15, -1e300)]
    for _ in range(24 if single else 60):
        k = g.randrange(3)
        if k == 0:
            m = -(10.0 ** g.uniform(6, top))
        elif k == 1:
            m = 1 - 10.0 ** -g.uniform(12.1, 15.9)
        else:
            m = 1 + 10.0 ** -g.uniform(12.1, 15.6)
        if single:
            m = to_f32(m)
            if m == 1.0:
                continue
        with mpmath.workprec(200):
            # The quarter period of the transformed parameter from its complement mu1 (mu itself
            # is 1 at this precision for m < -1e60): K = pi/(2 AGM(1, sqrt(mu1))).
            mm = mpmath.mpf(m)
            if m < 0:
                mu1, s = 1 / (1 - mm), mpmath.sqrt(1 - mm)
            elif m > 1:
                mu1, s = (mm - 1) / mm, mpmath.sqrt(mm)
            else:
                mu1, s = 1 - mm, mpmath.mpf(1)
            kk = mp.pi / (2 * mpmath.agm(1, mpmath.sqrt(mu1)))
            f = g.randrange(3)
            if f == 0:
                w = g.uniform(0, 3) * kk
            elif f == 1:
                w = (g.randint(0, 6) + 0.5) * kk * (1 + g.choice([0.0, g.uniform(-0.02, 0.02), 10.0 ** -g.uniform(3, 15)]))
            else:
                w = 10.0 ** g.uniform(0, 30 if single else 300) * s
            u = g.choice([-1, 1]) * float(w / s)
        if math.isfinite(u) and u != 0.0:
            cs.append((u, m))
    return [x for x in edge_args(cs, single) if all(math.isfinite(v) for v in x) and x[0] != 0.0]


def review2_jacobi_edges(single):
    """Found by the review of the fixes (round 2), after all the other rows (references by
    jacobi_agm): m < 0 with mu = -m/(1 - m) below 0.9 and u over several periods, where K for the
    reduction came from the rounded mu (8.3 ulps of cn at (-7.20, -3.80)); 0 <= m < 0.9 and m > 1,
    where dn of the last Landen step and of the step past K/2 took up to 4.3 ulps (at (7.5e6,
    0.886)); and the upper edge of the band 0.9 <= mu < 1 - 1e-6 (mu1 next to 1e-6)."""
    g = edge_rng("review2-jacobi", single)
    cs = [] if single else [
        (-7.203330143485626, -3.8043648081832555), (-7.495788943841146, -3.750152495182783),
        (-6.7582004122797965, -3.7987915842552744), (5.315718105664146, -3.867453072587876),
        (6.302067205650962, -8.0955571780849), (7510347.962705025, 0.8856009115923995),
        (-46.33576218429675, 0.8524635061857085), (12778.638986752856, 0.03052301511462128),
        (-1.0195159722885176, 0.8643798452600362), (69.316135528668, 0.8902771398338722),
        (-0.3823371115808824, -966628.1619364037), (0.054607656425301895, -963183.8628587329)]
    for k in range(20 if single else 90):
        c = k % 4
        if c == 0:
            mu = g.uniform(0.5, 0.9)
            m = -mu / (1 - mu)
        elif c == 1:
            m = mu = g.uniform(0.0, 0.9)
        elif c == 2:
            mu = g.uniform(0.3, 0.9)
            m = 1 / mu
        else:
            mu = 1 - 10.0 ** -g.uniform(5, 6)
            m = -mu / (1 - mu)
        with mpmath.workprec(120):
            sc = mpmath.sqrt(1 - mpmath.mpf(m)) if m < 0 else (mpmath.sqrt(mpmath.mpf(m)) if m > 1 else 1)
            kq = mpmath.ellipk(mu) / sc
            u = float(g.uniform(-8, 8) * kq) if g.random() < 0.6 else float((g.randint(-20, 20) + 0.5) * kq * (1 + g.uniform(-0.05, 0.05)))
        cs.append((u, m))
    return [x for x in edge_args(cs, single) if all(math.isfinite(v) for v in x) and x[0] != 0.0 and x[1] != 0.0]


def final_jacobi_edges(single):
    """Found by the confirmation of the round-2 fixes and the survey after it, after all the other
    rows (references by jacobi_agm): the Landen ascent of round 2 multiplied each step's sn by the
    rounded 1 + s against the exactly divided argument (3.68 ulps of dn at (-69.3, 0.889), 5.19 at
    (-4.65, -7.45)); the worst points of the survey of that code (band 1 up to 4.99, the band
    0.9 <= mu < 1 - 1e-6 up to 34.7, mu next to 1 up to 70.0) and of the code after it; and mu next
    to the largest errors of each band: u next to (j + 1/2)K, mu in [0.84, 0.9) of the three forms,
    mu1 = 1 - mu in [1e-6, 1e-5] and in [1e-12, 1e-10]."""
    g = edge_rng("final-jacobi", single)
    cs = [] if single else [
        (-69.31436094126745, 0.889104578480809), (-4.65468391847437, -7.450655753855837),
        (0.9015970365798311, 0.8885886131334039), (22.037323167929518, -8.366135249993407),
        (74.30534466107116, 0.8896772368273372), (-3.411037352035361, 0.8815807976356697),
        (1.2449951838815658, -7.653414496050229), (-4.114952152788178, -6.4941363259981655),
        (-9.715581783485627, -7.9840965495695), (-29.257476402296707, 4.607495476139785),
        (-8.186095445150544, 1.1232936920150458), (-0.2742391313123753, -552329.95719887),
        (-0.0383891616975901, -935857.0365177535), (0.08039443834099821, -405565.6529063705),
        (-7.010703872379336, 0.999999999992074), (94.38530598391199, 0.9999999999962982),
        (6.136861988121666, 0.8697407983658129), (-4.61425244076607, -8.147855260221124),
        (-79.69417701827258, 1.1614488152196358), (106.24296575584876, 1.1518480810484482),
        (0.021640482224594377, -905424.1075396767), (7.639614118685428, 0.9999999999987742),
        (108.46588815653908, 0.9999999999954418)]
    for k in range(30 if single else 100):
        c = k % 5
        if c < 3:
            mu1 = 1 - g.uniform(0.84, 0.9)
        elif c == 3:
            mu1 = 10.0 ** -g.uniform(5, 6)
        else:
            mu1 = 10.0 ** -g.uniform(10, 12)
        f = k % 3
        with mpmath.workprec(200):
            mu = 1 - mpmath.mpf(mu1)
            m = float(mu if f == 0 else (-mu / (1 - mu) if f == 1 else 1 / mu))
            if single:
                m = to_f32(m)
            if m == 0.0 or m == 1.0:
                continue
            mm = mpmath.mpf(m)
            if m < 0:
                mu1, sc = 1 / (1 - mm), mpmath.sqrt(1 - mm)
            elif m > 1:
                mu1, sc = (mm - 1) / mm, mpmath.sqrt(mm)
            else:
                mu1, sc = 1 - mm, mpmath.mpf(1)
            kk = mp.pi / (2 * mpmath.agm(1, mpmath.sqrt(mu1)))
            u = float((g.randint(-20, 20) + 0.5 + g.uniform(-0.17, 0.17)) * kk / sc)
        cs.append((u, m))
    return [x for x in edge_args(cs, single) if all(math.isfinite(v) for v in x) and x[0] != 0.0 and x[1] != 0.0]


def jacobi(single):
    sfx = "f32" if single else "f64"
    cnt = 120 if single else 420
    rows = []
    for i in range(cnt):
        k = i % 4
        if k == 0:
            u = rng.uniform(-20, 20)
        elif k == 1:
            u = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-10, 0)
        elif k == 2:
            u = rng.choice([-1, 1]) * 10.0 ** rng.uniform(1.3, 6 if single else 8)
        else:
            u = rng.uniform(-4, 4)
        j = (i // 4) % 6
        if j == 0:
            m = rng.uniform(0, 1)
        elif j == 1:
            m = 1 - 10.0 ** -rng.uniform(1, 15)
        elif j == 2:
            m = 10.0 ** rng.uniform(-15, -1)
        elif j == 3:
            m = -rng.uniform(0, 10)
        elif j == 4:
            m = rng.uniform(1, 10)
        else:
            m = rng.choice([-1, 1]) * 10.0 ** rng.uniform(1, 6)
        if single:
            u, m = to_f32(u), to_f32(m)
        uu, mm = M(u), M(m)
        rows.append(((u, m), [R(lambda a, b: mpmath.ellipfun(f, a, m=b), u, m) for f in ("sn", "cn", "dn")]))
    for u, m in jacobi_edges(single):
        rows.append(((u, m), [RP(lambda a, b: mpmath.ellipfun(f, a, m=b), u, m) for f in ("sn", "cn", "dn")]))
    for u, m in jacobi_near_one_edges(single):
        rows.append(((u, m), [RP(lambda a, b, f=f: jacobi_agm(f, a, b), u, m) for f in range(3)]))
    for u, m in review2_jacobi_edges(single):
        rows.append(((u, m), [RP(lambda a, b, f=f: jacobi_agm(f, a, b), u, m) for f in range(3)]))
    for u, m in final_jacobi_edges(single):
        rows.append(((u, m), [RP(lambda a, b, f=f: jacobi_agm(f, a, b), u, m) for f in range(3)]))
    write(f"elliptic_jacobi_{sfx}.txt", f"(sn, cn, dn)(u, m), {sfx}", rows)


# ------------------------------------------------------------------------------------------
# Orthogonal polynomials: reference and scale.


def grid_max_float(f, a, b, pts=600):
    """grid_max in floats, for the polynomials whose values fit (Jacobi, Gegenbauer)."""
    best = 0.0
    for k in range(pts):
        t = a + (b - a) * (k + 0.5) / pts
        try:
            best = max(best, abs(f(t)))
        except ZeroDivisionError:
            pass
    for t in (a, b):
        try:
            best = max(best, abs(f(float(t))))
        except ZeroDivisionError:
            pass
    return mpmath.mpf(best)


def grid_max(f, a, b, pts=600):
    lo = mp.prec
    mp.prec = 64
    try:
        best = mpmath.mpf(0)
        ts = [mpmath.mpf(a) + (mpmath.mpf(b) - a) * (k + mpmath.mpf(1) / 2) / pts for k in range(pts)]
        for t in ts + [mpmath.mpf(a), mpmath.mpf(b)]:
            try:
                best = max(best, abs(f(t)))
            except (ValueError, ZeroDivisionError):
                pass  # at a zero mpmath's hypergeometric series may not converge
    finally:
        mp.prec = lo
    return best


def rec_value(kind, n, t, a=0, b=0):
    """The polynomial by its three-term recurrence in mpf at the working precision: for the
    scales only (a few digits), where mpmath's hypergeometric series can be slow near zeros."""
    if kind == "plm":
        m = a
        p = (-1) ** m * mpmath.fac2(2 * m - 1) * (1 - t * t) ** (mpmath.mpf(m) / 2)
        if n == m:
            return p
        q, p = p, (2 * m + 1) * t * p
        for k in range(m + 1, n):
            q, p = p, ((2 * k + 1) * t * p - (k + m) * q) / (k - m + 1)
        return p
    if kind == "jac":
        q, p = 1.0, (a + 1) + (a + b + 2) * (t - 1) / 2
        if n == 0:
            return q
        for k in range(2, n + 1):
            s = 2 * k + a + b
            c = 2 * k * (k + a + b) * (s - 2)
            q, p = p, ((s - 1) * (s * (s - 2) * t + a * a - b * b) * p
                       - 2 * (k + a - 1) * (k + b - 1) * s * q) / c
        return p
    if kind == "geg":
        q, p = 1.0, 2 * a * t
        if n == 0:
            return q
        for k in range(1, n):
            q, p = p, (2 * (k + a) * t * p - (k + 2 * a - 1) * q) / (k + 1)
        return p
    if kind == "lag":
        q, p = mpmath.mpf(1), 1 + a - t
        if n == 0:
            return q
        for k in range(1, n):
            q, p = p, ((2 * k + 1 + a - t) * p - (k + a) * q) / (k + 1)
        return p
    if kind in ("h", "he"):
        q, p = mpmath.mpf(1), (2 * t if kind == "h" else t)
        if n == 0:
            return q
        for k in range(1, n):
            q, p = p, ((2 * t * p - 2 * k * q) if kind == "h" else (t * p - k * q))
        return p
    raise ValueError(kind)


def n_of(i, top):
    k = i % 4
    if k == 0:
        return rng.randint(0, 8)
    if k == 1:
        return rng.randint(9, 40)
    if k == 2:
        return rng.randint(41, 200)
    return rng.randint(min(201, top), top)


def jacobi_scale(n, a, b):
    """max |P_n^(a,b)| on [-1, 1] by the exact sum of DLMF 18.5.8, its coefficients once."""
    with mpmath.workprec(3 * n + 80):
        a, b = mpmath.mpf(a), mpmath.mpf(b)
        cs = [mpmath.binomial(n + a, n - s) * mpmath.binomial(n + b, s) for s in range(n + 1)]
        best = mpmath.mpf(0)
        for k in range(602):
            t = mpmath.mpf(-1) + 2 * (k + mpmath.mpf(1) / 2) / 600 if k < 600 else mpmath.mpf(2 * k - 1201)
            y, z = (t - 1) / 2, (t + 1) / 2
            best = max(best, abs(mpmath.fsum(c * y ** s * z ** (n - s) for s, c in enumerate(cs))))
        return best


def poly_edges(single):
    """The degenerate parameters of the audit (Jacobi alpha + beta, alpha or beta a negative
    integer; Gegenbauer 2 alpha and Laguerre alpha negative integers), where the recurrence divided
    by zero or decoupled, with references by exact sums, and |x| > 2^400, where its first step
    overflowed: per file, the rows (arguments, [value, scale])."""
    g = edge_rng("poly", single)
    fix = to_f32 if single else (lambda v: v)
    out = {k: [] for k in ("legendre", "hermite_h", "hermite_he", "laguerre", "assoc_laguerre",
                           "chebyshev_t", "chebyshev_u", "jacobi_p", "gegenbauer")}
    ns = [3, 5, 10, 20, 50] if single else [3, 5, 10, 20, 50, 100, 300]
    cfgs = [(-1.0, -1.0), (-0.5, -1.5), (-2.0, 0.0), (-1.5, -1.5), (-2.5, -0.5), (0.5, -3.5), (-4.5, -4.5),
            (-1.0, -3.0), (-3.0, 1.0)]
    for a, b in cfgs:
        for n in ns:
            S = jacobi_scale(n, a, b)
            xs = [fix(g.uniform(-1, 1)) for _ in range(3)] + [1.0, -1.0, fix(g.choice([-1, 1]) * g.uniform(1, 1.5))]
            if n == 3 and (a, b) == (-1.0, -1.0):
                xs += [0.3]
            if n in (10, 20, 50) and (a, b) == (-1.0, -1.0):
                xs += [0.5]
            if n == 100 and (a, b) == (-1.0, -1.0):
                xs += [0.0]
            if n == 300 and (a, b) == (-0.5, -1.5):
                xs += [0.25]
            for x in xs:
                out["jacobi_p"].append(((float(n), a, b, x), [exact_at(jacobi_exact, n, a, b, x), S if abs(x) <= 1 else 0]))
    ngs = [5, 10, 20, 50, 128] if single else [5, 10, 20, 50, 128, 200, 300, 400]
    for a in (-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0):
        for n in ngs:
            S = grid_max_float(lambda t: rec_value("geg", n, t, a), -1.0, 1.0)
            xs = [fix(g.uniform(-1, 1)) for _ in range(2)] + [fix(0.999999999), fix(-0.9999), 1.0, fix(1 + g.uniform(0, 0.4))]
            if (n, a) == (200, -1.5):
                xs += [0.999999999]
            if (n, a) == (400, -2.5) and not single:
                xs += [0.9999999990686774]
            if (n, a) == (300, -3.0) and not single:
                xs += [-0.06099916351786194]
            if (n, a) == (128, -3.0):
                xs += [fix(1.3612104441547808)]
            for x in xs:
                out["gegenbauer"].append(((float(n), a, x), [exact_at(gegenbauer_exact, n, a, x), S if abs(x) <= 1 else 0]))
    nls = [3, 10, 50, 200] if single else [3, 10, 50, 200, 400]
    for a in (-1.0, -2.0, -3.0, -5.0):
        for n in nls:
            if n < -a:
                continue
            edge = 4 * n + 2 * a + 2
            with mpmath.workprec(64):
                sc = grid_max(lambda t: rec_value("lag", n, t, mpmath.mpf(a)) * mpmath.exp(-t / 2), 0, max(edge, 1))
            xs = [fix(1e-9), fix(g.uniform(0, edge)), fix(g.uniform(0, 1)), fix(-g.uniform(0, 5))]
            for x in xs:
                s_ = sc * mpmath.exp(mpmath.mpf(x) / 2) if 0 <= x <= edge else 0
                out["assoc_laguerre"].append(((float(n), a, x), [exact_at(laguerre_exact, n, a, x), s_]))
    # |x| > 2^400 (2^100 in f32, where the values pass f32's range): the leading term.
    lo, hi = (100, 127) if single else (390, 1023)
    huge = [(2, 1e154), (2, 3e192), (1, 1e300), (2, 1e200)] if not single else [(2, 1e19), (1, 3e38)]
    huge += [(g.choice([1, 2, 3, 5]), g.choice([-1, 1]) * 2.0 ** g.uniform(lo, hi)) for _ in range(12)]
    for n, x in edge_args(huge, single):
        n = int(n)
        out["legendre"].append(((float(n), x), [RP(lambda t: mpmath.legendre(n, t), x), mpmath.mpf(0)]))
        out["hermite_h"].append(((float(n), x), [RP(lambda t: mpmath.hermite(n, t), x), mpmath.mpf(0)]))
        out["hermite_he"].append(((float(n), x), [RP(lambda t: mpmath.hermite(n, t / mpmath.sqrt(2)) * mpmath.mpf(2) ** (-mpmath.mpf(n) / 2), x), mpmath.mpf(0)]))
        out["laguerre"].append(((float(n), x), [RP(lambda t: mpmath.laguerre(n, 0, t), x), mpmath.mpf(0)]))
        out["assoc_laguerre"].append(((float(n), 0.5, x), [RP(lambda t: mpmath.laguerre(n, mpmath.mpf(0.5), t), x), mpmath.mpf(0)]))
        out["chebyshev_t"].append(((float(n), x), [RP(lambda t: mpmath.chebyt(n, t), x), mpmath.mpf(0)]))
        out["chebyshev_u"].append(((float(n), x), [RP(lambda t: mpmath.chebyu(n, t), x), mpmath.mpf(0)]))
        out["jacobi_p"].append(((float(n), 0.5, 0.5, x), [RP(lambda t: mpmath.jacobi(n, mpmath.mpf(0.5), mpmath.mpf(0.5), t), x), mpmath.mpf(0)]))
        out["gegenbauer"].append(((float(n), 1.5, x), [RP(lambda t: mpmath.gegenbauer(n, mpmath.mpf(1.5), t), x), mpmath.mpf(0)]))
    if not single:
        out["chebyshev_t"].append(((1.0, 1e300), [mpmath.mpf(1e300), mpmath.mpf(0)]))
        out["chebyshev_u"].append(((1.0, 1.5e158), [mpmath.mpf(3e158), mpmath.mpf(0)]))
    return out


def review1_jacobi_edges(single):
    """Found by the review of the fixes (round 1), after all the other rows of jacobi_p, with exact
    rational references (jacobi_rational): |x| > 2^400 with alpha + beta = -l + delta and
    n < l <= 2n, whose leading coefficient has the factor delta (0 or an infinity for a finite
    value: P_1^(1/2,-5/2)(1e150) = 1.5), both signs of x, just above 2^400 and next to f64::MAX
    (f64 only: f32 arguments never reach 2^400); and alpha + beta within 1/2 of -l, l = 2..9,
    alpha, beta in (-5, 10), where two coefficients of the recurrence are about delta times their
    size (1.5e13 ulps of S at (29, -4.69012, 0.69012, 0.99999999999)). Rows (args, [value, scale])."""
    g = edge_rng("review1-jacobi", single)
    fix = to_f32 if single else (lambda v: v)
    cs = []
    if not single:
        cs += [(1, 0.5, -2.5, 1e150), (2, 0.5, -3.5, -1e200), (3, -1.0, -4.0, 1e150), (2, 1.5, -5.5, 1e150),
               (4, 0.5, -8.5, 3e120)]
        for _ in range(40):
            n = g.randint(1, 8)
            l = g.randint(n + 1, 2 * n) if n > 1 else 2
            a = g.choice([g.uniform(-5, 10), round(g.uniform(-5, 10) * 4) / 4])
            d = g.choice([0.0, 0.0, 10.0 ** -g.uniform(1, 15), -(10.0 ** -g.uniform(1, 15))])
            x = g.choice([-1, 1]) * 2.0 ** g.choice([g.uniform(400, 402), g.uniform(1000, 1023.99), g.uniform(400, 1023.99)])
            cs.append((n, a, -l - a + d, x))
    near = [(29, -4.690120023422273, 0.6901200234284552, 0.9999999999934994), (60, 0.3, -2.2999999999, 0.999),
            (7, 0.69, -4.6899999999, -0.3)]
    for _ in range(14 if single else 40):
        while True:
            l = g.randint(2, 9)
            a = g.uniform(-4.99, 9.99)
            d = g.choice([-1, 1]) * 10.0 ** -g.uniform(0.3, 15)
            b = -l - a + d
            if -5 < b < 10:
                break
        t = g.random()
        n = g.randint(l, 60) if t < 0.6 else (g.randint(1, l) if t < 0.8 else g.randint(l, 300))
        x = g.uniform(-1, 1) if g.random() < 0.6 else g.choice(
            [(1 - 10.0 ** -g.uniform(1, 15)) * g.choice([-1, 1]), g.choice([-1, 1]) * g.uniform(1, 3)])
        near.append((n, a, b, x))
    rows = []
    for n, a, b, x in cs + near:
        a, b, x = fix(a), fix(b), fix(x)
        S = jacobi_scale(n, a, b) if abs(x) <= 1 else mpmath.mpf(0)
        rows.append(((float(n), a, b, x), [jacobi_rational(n, a, b, x), S]))
    return rows


def review2_jacobi_p_edges(single):
    """Found by the review of the fixes (round 2), after all the other rows of jacobi_p, with exact
    rational references: alpha + beta within 1/8 of -l, l = 2..4, past |delta| = 1/64 (where the
    recurrence was kept) at degrees 16 to 300 and at or next to x = +-1 (1.35 ulps of S at
    (232, -0.2562, -3.7597, -1)). Rows (args, [value, scale])."""
    g = edge_rng("review2-jacobi-p", single)
    fix = to_f32 if single else (lambda v: v)
    a0, b0 = -0.25621206658540085, -3.7596785609235583
    cs = [] if single else [(232, a0, b0, -1.0), (232, a0, b0, -0.999999999925448),
                            (233, -3.847595687219682, -0.18096837542410515, 0.9999999999999999),
                            (232, a0, -4 - 0.0157 - a0, -1.0), (232, a0, -4 + 0.0159 - a0, -1.0),
                            (232, a0, -4 - 0.02 - a0, -1.0)]
    for _ in range(10 if single else 34):
        l = g.choice([2, 3, 4])
        d = g.uniform(1 / 64, 1 / 8) * g.choice([1, -1])
        a = g.uniform(max(-5, -l + d - 10), min(10, -l + d + 5))
        n = int(round(10 ** g.uniform(math.log10(16), math.log10(300))))
        r = g.random()
        x = g.choice([1.0, -1.0]) if r < 0.4 else (g.choice([1, -1]) * (1 - 10.0 ** -g.uniform(1, 13)) if r < 0.8 else g.uniform(-1, 1))
        cs.append((n, a, -l + d - a, x))
    rows = []
    for n, a, b, x in cs:
        a, b, x = fix(a), fix(b), fix(x)
        S = jacobi_scale(n, a, b) if abs(x) <= 1 else mpmath.mpf(0)
        rows.append(((float(n), a, b, x), [jacobi_rational(n, a, b, x), S]))
    return rows


def orthopoly(single):
    sfx = "f32" if single else "f64"
    cnt = 30 if single else 120
    fix = to_f32 if single else (lambda v: v)
    top = 1000
    edges = poly_edges(single)

    rows = []
    for i in range(cnt):
        n = n_of(i, top)
        x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 3))
        s = 1 if abs(x) <= 1 else 0
        rows.append(((float(n), x), [mpmath.legendre(n, M(x)), mpmath.mpf(s)]))
    write(f"elliptic_legendre_{sfx}.txt", f"P_n(x) and its scale, {sfx}", rows + edges["legendre"])

    rows = []
    cache = {}
    for i in range(cnt):
        n = n_of(i, 120)
        m = rng.randint(0, n + 1) if i % 9 else n + rng.randint(1, 3)
        x = fix(rng.uniform(-1, 1))
        if (n, m) not in cache:
            cache[(n, m)] = grid_max(lambda t: rec_value("plm", n, t, m), -1, 1) if m <= n else 0
        v = mpmath.legenp(n, m, M(x)) if m <= n else mpmath.mpf(0)
        rows.append(((float(n), float(m), x), [re(v), cache[(n, m)]]))
    write(f"elliptic_assoc_legendre_{sfx}.txt", f"P_n^m(x) (Condon-Shortley) and its scale, {sfx}", rows)

    def herm(phys):
        rows = []
        cache = {}
        for i in range(cnt):
            n = n_of(i, 300 if phys else 300)
            edge = math.sqrt(2 * n + 1) if phys else math.sqrt(4 * n + 2)
            x = fix(rng.uniform(-edge, edge) if i % 5 else rng.choice([-1, 1]) * rng.uniform(edge, 3 * edge + 3))
            if phys:
                f = lambda t: mpmath.hermite(n, t)
                w = lambda t: mpmath.exp(-t * t / 2)
            else:
                f = lambda t: mpmath.hermite(n, t / mpmath.sqrt(2)) * mpmath.mpf(2) ** (-mpmath.mpf(n) / 2)
                w = lambda t: mpmath.exp(-t * t / 4)
            kind = "h" if phys else "he"
            if n not in cache:
                cache[n] = grid_max(lambda t: rec_value(kind, n, t) * w(t), 0, edge + 1)
            s = cache[n] / w(M(x)) if abs(x) <= edge else 0
            rows.append(((float(n), x), [f(M(x)), s]))
        return rows
    write(f"elliptic_hermite_h_{sfx}.txt", f"H_n(x) and its scale, {sfx}", herm(True) + edges["hermite_h"])
    write(f"elliptic_hermite_he_{sfx}.txt", f"He_n(x) and its scale, {sfx}", herm(False) + edges["hermite_he"])

    def lag(assoc):
        rows = []
        cache = {}
        for i in range(cnt):
            n = n_of(i, 400)
            al = fix(rng.choice([rng.uniform(-0.99, 5), rng.uniform(-10, 30)])) if assoc else 0.0
            if assoc and i % 3 == 0:
                al = rng.randint(-1, 40) / 4  # on the grid of the exact-coefficient path
            edge = 4 * n + 2 * al + 2
            x = fix(rng.uniform(0, max(edge, 1)) if i % 5 else rng.choice([rng.uniform(-10, 0), rng.uniform(edge, 3 * edge + 10)]))
            f = lambda t: mpmath.laguerre(n, M(al), t)
            key = (n, al)
            if key not in cache:
                cache[key] = grid_max(lambda t: rec_value("lag", n, t, M(al)) * mpmath.exp(-t / 2), 0, max(edge, 1))
            s = cache[key] * mpmath.exp(M(x) / 2) if 0 <= x <= edge else 0
            args = (float(n), al, x) if assoc else (float(n), x)
            rows.append((args, [f(M(x)), s]))
        return rows
    write(f"elliptic_laguerre_{sfx}.txt", f"L_n(x) and its scale, {sfx}", lag(False) + edges["laguerre"])
    write(f"elliptic_assoc_laguerre_{sfx}.txt", f"L_n^(alpha)(x) and its scale, {sfx}", lag(True) + edges["assoc_laguerre"])

    for name, f, sc in [("chebyshev_t", mpmath.chebyt, lambda n: 1), ("chebyshev_u", mpmath.chebyu, lambda n: n + 1)]:
        rows = []
        for i in range(cnt):
            n = n_of(i, 5000)
            x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 1.5))
            s = sc(n) if abs(x) <= 1 else 0
            rows.append(((float(n), x), [f(n, M(x)), mpmath.mpf(s)]))
        write(f"elliptic_{name}_{sfx}.txt", f"{name}(n, x) and its scale, {sfx}", rows + edges[name])

    rows = []
    cache = {}
    for i in range(cnt):
        n = n_of(i, 300)
        a = fix(rng.choice([rng.uniform(-0.99, 3), rng.uniform(-5, 10)]))
        b = fix(rng.choice([rng.uniform(-0.99, 3), rng.uniform(-5, 10)]))
        if i % 3 == 0:
            a, b = rng.randint(-3, 40) / 4, rng.randint(-3, 40) / 4  # on the grid
        x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 2))
        key = (n, a, b)
        if key not in cache:
            cache[key] = grid_max_float(lambda t: rec_value("jac", n, t, a, b), -1.0, 1.0)
        s = cache[key] if abs(x) <= 1 else 0
        rows.append(((float(n), a, b, x), [mpmath.jacobi(n, M(a), M(b), M(x)), s]))
    write(f"elliptic_jacobi_p_{sfx}.txt", f"P_n^(alpha, beta)(x) and its scale, {sfx}",
          rows + edges["jacobi_p"] + review1_jacobi_edges(single) + review2_jacobi_p_edges(single))

    rows = []
    cache = {}
    for i in range(cnt):
        n = n_of(i, 400)
        a = fix(rng.choice([rng.uniform(-0.49, 3), rng.uniform(0, 20), rng.uniform(-5, 0)]))
        if i % 3 == 0:
            a = rng.randint(-8, 80) / 4  # on the grid
        x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 2))
        key = (n, a)
        if key not in cache:
            cache[key] = grid_max_float(lambda t: rec_value("geg", n, t, a), -1.0, 1.0)
        s = cache[key] if abs(x) <= 1 else 0
        rows.append(((float(n), a, x), [mpmath.gegenbauer(n, M(a), M(x)), s]))
    write(f"elliptic_gegenbauer_{sfx}.txt", f"C_n^(alpha)(x) and its scale, {sfx}", rows + edges["gegenbauer"])


if __name__ == "__main__":
    import sys

    parts = sys.argv[1:] or ["complete", "incomplete", "carlson", "jacobi", "orthopoly"]
    for p in parts:
        rng.seed("elliptic-" + p)
        for single in (False, True):
            globals()[p](single)
