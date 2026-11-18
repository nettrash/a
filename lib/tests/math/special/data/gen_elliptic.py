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

Run from this directory: python3 gen_elliptic.py
"""

import math
import random

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


def complete(single):
    sfx = "f32" if single else "f64"
    n = 120 if single else 350
    xs = m_args(n, single)
    write(f"elliptic_k_{sfx}.txt", f"K(m), {sfx}", [((x,), [R(mpmath.ellipk, x)]) for x in xs])
    xs = m_args(n, single)
    write(f"elliptic_e_{sfx}.txt", f"E(m), {sfx}", [((x,), [R(mpmath.ellipe, x)]) for x in xs])
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


def carlson(single):
    sfx = "f32" if single else "f64"
    cnt = 80 if single else 250
    a = carlson_arg
    rows = []
    for _ in range(cnt):
        x, y, z = a(single), a(single), a(single)
        if [x, y, z].count(0.0) > 1:
            continue
        rows.append(((x, y, z), [R(mpmath.elliprf, x, y, z)]))
    write(f"elliptic_rf_{sfx}.txt", f"R_F(x, y, z), {sfx}", rows)
    rows = []
    for _ in range(cnt):
        x, y, z = a(single), a(single), a(single)
        if z == 0.0 or (x == 0.0 and y == 0.0):
            continue
        rows.append(((x, y, z), [R(mpmath.elliprd, x, y, z)]))
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
    write(f"elliptic_rc_{sfx}.txt", f"R_C(x, y), {sfx}; the principal value for y < 0", rows)


# ------------------------------------------------------------------------------------------
# Jacobi elliptic functions: references sn, cn, dn.


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


def orthopoly(single):
    sfx = "f32" if single else "f64"
    cnt = 30 if single else 120
    fix = to_f32 if single else (lambda v: v)
    top = 1000

    rows = []
    for i in range(cnt):
        n = n_of(i, top)
        x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 3))
        s = 1 if abs(x) <= 1 else 0
        rows.append(((float(n), x), [mpmath.legendre(n, M(x)), mpmath.mpf(s)]))
    write(f"elliptic_legendre_{sfx}.txt", f"P_n(x) and its scale, {sfx}", rows)

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
    write(f"elliptic_hermite_h_{sfx}.txt", f"H_n(x) and its scale, {sfx}", herm(True))
    write(f"elliptic_hermite_he_{sfx}.txt", f"He_n(x) and its scale, {sfx}", herm(False))

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
    write(f"elliptic_laguerre_{sfx}.txt", f"L_n(x) and its scale, {sfx}", lag(False))
    write(f"elliptic_assoc_laguerre_{sfx}.txt", f"L_n^(alpha)(x) and its scale, {sfx}", lag(True))

    for name, f, sc in [("chebyshev_t", mpmath.chebyt, lambda n: 1), ("chebyshev_u", mpmath.chebyu, lambda n: n + 1)]:
        rows = []
        for i in range(cnt):
            n = n_of(i, 5000)
            x = fix(rng.uniform(-1, 1) if i % 5 else rng.choice([-1, 1]) * rng.uniform(1, 1.5))
            s = sc(n) if abs(x) <= 1 else 0
            rows.append(((float(n), x), [f(n, M(x)), mpmath.mpf(s)]))
        write(f"elliptic_{name}_{sfx}.txt", f"{name}(n, x) and its scale, {sfx}", rows)

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
    write(f"elliptic_jacobi_p_{sfx}.txt", f"P_n^(alpha, beta)(x) and its scale, {sfx}", rows)

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
    write(f"elliptic_gegenbauer_{sfx}.txt", f"C_n^(alpha)(x) and its scale, {sfx}", rows)


if __name__ == "__main__":
    import sys

    parts = sys.argv[1:] or ["complete", "incomplete", "carlson", "jacobi", "orthopoly"]
    for p in parts:
        rng.seed("elliptic-" + p)
        for single in (False, True):
            globals()[p](single)
