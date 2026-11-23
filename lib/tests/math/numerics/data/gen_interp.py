"""Reference data of the interpolation tests (`../interp.rs`), written to `interp_ref.rs`.

Run from this directory with `python3 gen_interp.py`; the data are deterministic (seeded).

* Splines: for random non-uniform knots (2, 3, 4, 5, 8 and 13 of them) and random values, in f64
  and in f32, the slopes of the five splines solved in exact rational arithmetic from the data
  as stored (the natural, clamped and not-a-knot systems by Gaussian elimination on fractions;
  Akima's and PCHIP's explicit formulas on fractions), and the value and first derivative of each
  spline at a few points, from the exact Hermite cubics. SciPy 1.18's `CubicSpline` (bc_type
  'natural', ((1, left), (1, right)), 'not-a-knot'), `Akima1DInterpolator` (method 'akima') and
  `PchipInterpolator` are compared with the exact values here, and the largest discrepancy is
  printed: that is what shows that the rules followed are SciPy's.
* The not-a-knot end slope for an end interval R times longer than its neighbour (the data of
  audit finding n_interp-10), exact.
* Not-a-knot splines with a short second, middle or penultimate interval, and over knots whose
  intervals range over six orders of magnitude (review round 1, B1): exact slopes, and the change
  one unit in the last place of the values makes in them.
* Chebyshev interpolants: coefficients from the exact interpolation at the first-kind Chebyshev
  points (mpmath, 60 digits), checked against numpy.polynomial.chebyshev.Chebyshev.interpolate;
  values, derivatives and integrals at points.
"""

import random
from fractions import Fraction as Fr

import mpmath
import numpy as np
from numpy.polynomial import chebyshev as npc
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

mpmath.mp.dps = 60

KINDS = ("natural", "clamped", "not_a_knot", "akima", "pchip")


def solve(a, b):
    """Exact Gaussian elimination on fractions."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [u - f * v for u, v in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def c2_slopes(x, y, kind, left=None, right=None):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [left, right] if kind == "clamped" else [d[0], d[0]]
    if n == 3 and kind == "not_a_knot":
        a = [[Fr(1), Fr(1), Fr(0)], [h[1], 2 * (h[0] + h[1]), h[0]], [Fr(0), Fr(1), Fr(1)]]
        b = [2 * d[0], 3 * (h[1] * d[0] + h[0] * d[1]), 2 * d[1]]
        return solve(a, b)
    a = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for i in range(1, n - 1):
        a[i][i - 1] = h[i]
        a[i][i] = 2 * (h[i - 1] + h[i])
        a[i][i + 1] = h[i - 1]
        b[i] = 3 * (h[i] * d[i - 1] + h[i - 1] * d[i])
    if kind == "natural":
        a[0][0], a[0][1], b[0] = Fr(2), Fr(1), 3 * d[0]
        a[n - 1][n - 2], a[n - 1][n - 1], b[n - 1] = Fr(1), Fr(2), 3 * d[n - 2]
    elif kind == "clamped":
        a[0][0], b[0] = Fr(1), left
        a[n - 1][n - 1], b[n - 1] = Fr(1), right
    else:
        # the third derivative continued across x[1] and x[n-2]
        a[0][0], a[0][1], a[0][2] = h[1] ** 2, h[1] ** 2 - h[0] ** 2, -h[0] ** 2
        b[0] = 2 * h[1] ** 2 * d[0] - 2 * h[0] ** 2 * d[1]
        k = n - 2
        a[n - 1][n - 1], a[n - 1][n - 2], a[n - 1][n - 3] = h[k - 1] ** 2, h[k - 1] ** 2 - h[k] ** 2, -h[k] ** 2
        b[n - 1] = 2 * h[k - 1] ** 2 * d[k] - 2 * h[k] ** 2 * d[k - 1]
    return solve(a, b)


def akima_slopes(x, y):
    n = len(x)
    if n == 2:
        s = (y[1] - y[0]) / (x[1] - x[0])
        return [s, s]
    m = [Fr(0)] * (n + 3)
    for k in range(n - 1):
        m[k + 2] = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    m[1] = 2 * m[2] - m[3]
    m[0] = 2 * m[1] - m[2]
    m[n + 1] = 2 * m[n] - m[n - 1]
    m[n + 2] = 2 * m[n + 1] - m[n]
    out = []
    for i in range(n):
        d0, d1, d2, d3 = m[i : i + 4]
        w1, w2 = abs(d3 - d2), abs(d1 - d0)
        out.append((d1 + d2) / 2 if w1 + w2 == 0 else (w1 * d1 + w2 * d2) / (w1 + w2))
    return out


def sgn(v):
    return (v > 0) - (v < 0)


def pchip_slopes(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    m = [Fr(0)] * n
    for i in range(1, n - 1):
        if sgn(d[i - 1]) * sgn(d[i]) > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def end(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if sgn(s) != sgn(d0):
            return Fr(0)
        if sgn(d0) != sgn(d1) and abs(s) > 3 * abs(d0):
            return 3 * d0
        return s

    m[0] = end(h[0], h[1], d[0], d[1])
    m[n - 1] = end(h[n - 2], h[n - 3], d[n - 2], d[n - 3])
    return m


def hermite(x, y, m, t):
    """The value and derivative of the Hermite cubic at t, exactly."""
    n = len(x)
    i = max(k for k in range(n - 1) if x[k] <= t)
    h = x[i + 1] - x[i]
    u = (t - x[i]) / h
    v = 1 - u
    val = (1 + 2 * u) * v * v * y[i] + u * v * v * h * m[i] + u * u * (3 - 2 * u) * y[i + 1] + u * u * (u - 1) * h * m[i + 1]
    der = 6 * u * v * (y[i + 1] - y[i]) / h + v * (v - 2 * u) * m[i] + u * (3 * u - 2) * m[i + 1]
    return val, der


def scipy_spline(kind, x, y, left, right):
    if kind == "natural":
        return CubicSpline(x, y, bc_type="natural")
    if kind == "clamped":
        return CubicSpline(x, y, bc_type=((1, left), (1, right)))
    if kind == "not_a_knot":
        return CubicSpline(x, y, bc_type="not-a-knot")
    if kind == "akima":
        return Akima1DInterpolator(x, y, method="akima")
    return PchipInterpolator(x, y)


def lit(v, f32):
    """A Rust literal that reads back as the value: the shortest round trip of the f32, or of the
    f64."""
    v = float(v)
    if f32:
        s = repr(np.float32(v))
        s = s[len("np.float32(") : -1] if s.startswith("np.float32(") else s
    else:
        s = repr(v)
    if "e" not in s and "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def arr(vals, f32=False):
    return "&[" + ", ".join(lit(v, f32) for v in vals) + "]"


def spline_cases(out, f32):
    rng = random.Random(21 if f32 else 20)
    t = np.float32 if f32 else np.float64
    eps = float(np.finfo(t).eps)
    name = "SPLINES_F32" if f32 else "SPLINES_F64"
    worst = {k: 0.0 for k in KINDS}
    out.append(f"/// Exact slopes, values and derivatives of the five splines ({'f32' if f32 else 'f64'} data).")
    out.append(f"pub const {name}: &[Case] = &[")
    for n in (2, 3, 4, 5, 8, 13):
        while True:
            gaps = [rng.uniform(0.05, 1.0) * 10 ** rng.uniform(-0.7, 0.7) for _ in range(n - 1)]
            xs = [rng.uniform(-2.0, 2.0)]
            for g in gaps:
                xs.append(xs[-1] + g)
            xs = [float(t(v)) for v in xs]
            if all(xs[i] < xs[i + 1] for i in range(n - 1)):
                break
        ys = [float(t(rng.uniform(-1.0, 1.0) + 0.3 * v)) for v in xs]
        left, right = float(t(rng.uniform(-2, 2))), float(t(rng.uniform(-2, 2)))
        fx, fy = [Fr(v) for v in xs], [Fr(v) for v in ys]
        slopes = {
            "natural": c2_slopes(fx, fy, "natural"),
            "clamped": c2_slopes(fx, fy, "clamped", Fr(left), Fr(right)),
            "not_a_knot": c2_slopes(fx, fy, "not_a_knot"),
            "akima": akima_slopes(fx, fy),
            "pchip": pchip_slopes(fx, fy),
        }
        # points: the knots' midpoints, the ends, and two random points
        pts = [(xs[i] + xs[i + 1]) / 2 for i in range(n - 1)] + [xs[0], xs[-1]]
        pts += [rng.uniform(xs[0], xs[-1]) for _ in range(2)]
        pts = [float(t(p)) for p in pts]
        pts = [min(max(p, xs[0]), xs[-1]) for p in pts]
        values, derivs = {}, {}
        for k in KINDS:
            vd = [hermite(fx, fy, slopes[k], Fr(p)) for p in pts]
            values[k] = [float(v) for v, _ in vd]
            derivs[k] = [float(d) for _, d in vd]
            sp = scipy_spline(k, np.array(xs), np.array(ys), left, right)
            sm = sp(np.array(xs), 1)
            scale = max(abs(float(s)) for s in slopes[k]) or 1.0
            worst[k] = max(worst[k], max(abs(float(a) - float(b)) for a, b in zip(sm, slopes[k])) / scale / eps)
            sv, sd = sp(np.array(pts)), sp(np.array(pts), 1)
            vs = max(abs(v) for v in ys) or 1.0
            worst[k] = max(worst[k], max(abs(a - b) for a, b in zip(sv, values[k])) / vs / eps)
            worst[k] = max(worst[k], max(abs(a - b) for a, b in zip(sd, derivs[k])) / scale / eps)
        out.append("    Case {")
        out.append(f"        x: {arr(xs, f32)},")
        out.append(f"        y: {arr(ys, f32)},")
        out.append(f"        left: {lit(left, f32)},")
        out.append(f"        right: {lit(right, f32)},")
        out.append("        slopes: [" + ", ".join(arr([float(s) for s in slopes[k]]) for k in KINDS) + "],")
        out.append(f"        at: {arr(pts, f32)},")
        out.append("        value: [" + ", ".join(arr(values[k]) for k in KINDS) + "],")
        out.append("        derivative: [" + ", ".join(arr(derivs[k]) for k in KINDS) + "],")
        out.append("    },")
    out.append("];")
    print(f"{name}: largest difference of SciPy from the exact splines, in eps of the data's type:")
    for k in KINDS:
        print(f"    {k:>10}: {worst[k]:.2f}")


def nak_ratio(out):
    ys = [0.3, -1.2, 0.7, 1.9, -0.4, 0.8]
    out.append("/// The first slope of the not-a-knot spline through knots 0, R, R + 1, R + 2.1, R + 3.3,")
    out.append("/// R + 4.2 (rounded to the type) and the values `NAK_VALUES`, exactly: (R, f64 slope, f32 slope).")
    out.append("pub const NAK_VALUES: [f64; 6] = " + arr(ys)[1:] + ";")
    out.append("pub const NAK_RATIO: &[(f64, f64, f64)] = &[")
    for r in (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
        row = []
        for t in (np.float64, np.float32):
            x = [Fr(float(t(v))) for v in (0.0, r, r + 1.0, r + 2.1, r + 3.3, r + 4.2)]
            y = [Fr(float(t(v))) for v in ys]
            row.append(float(c2_slopes(x, y, "not_a_knot")[0]))
            sp = CubicSpline(np.array([float(v) for v in x]), np.array([float(v) for v in y]))(0.0, 1)
            if t is np.float64:
                print(f"not-a-knot R = {r:.0e}: SciPy's first slope is {abs(sp - row[0]) / abs(row[0]) / 2.2e-16:.0f} eps off")
        out.append(f"    ({lit(r, False)}, {lit(row[0], False)}, {lit(row[1], False)}),")
    out.append("];")


def nak_short(out):
    """Not-a-knot splines whose knots have a short interval among long ones, or intervals that
    range over six orders of magnitude: the exact slopes of the data as stored, and how far one
    unit in the last place of the values moves them (the largest change over 40 perturbations of
    every value by one ulp up or down, relative to the largest slope, in eps of the type), which
    is the accuracy the data allow. SciPy's error on the f64 data is printed beside it."""
    rng = random.Random(23)
    geometries = []
    for d in (1e-3, 1e-4):
        geometries += [
            ("sine", [0.0, 1.0, 1.0 + d, 2.0]),
            ("random", [0.0, 1.0, 1.0 + d, 2.0]),
            ("sine", [0.0, 1.0, 1.0 + d, 2.0, 3.0]),
            ("random", [0.0, 1.0, 1.0 + d, 2.0, 3.0]),
            ("random", [0.0, 1.0, 2.0, 2.0 + d, 3.0]),
            ("sine", [0.0, 1.0, 1.0 + d, 2.0, 3.0, 4.0]),
            ("random", [0.0, d, 1.0, 2.0]),
        ]
    wide = [0.0, 0.05, 0.054, 0.06, 500.0, 760.0, 6700.0, 9300.0, 9330.0]
    geometries += [("smooth", wide), ("random", wide)]
    out.append("/// Not-a-knot splines with a short interval among long ones, and over knots whose intervals")
    out.append("/// range from 4e-3 to 6e3: (f32 data?, knots, values, the exact slopes, the change one ulp in")
    out.append("/// the values makes in them, relative to the largest, in eps of the type).")
    out.append("pub const NAK_SHORT: &[NakCase] = &[")
    for kind, knots in geometries:
        values = [rng.uniform(-1.0, 1.0) for _ in knots]
        for t in (np.float64, np.float32):
            f32 = t is np.float32
            eps = float(np.finfo(t).eps)
            xs = [float(t(v)) for v in knots]
            if kind == "sine":
                ys = [float(np.sin(t(v))) for v in xs]
            elif kind == "smooth":
                ys = [float(np.sin(t(3.0 * v / 9330.0))) for v in xs]
            else:
                ys = [float(t(v)) for v in values]
            fx = [Fr(v) for v in xs]
            exact = [float(s) for s in c2_slopes(fx, [Fr(v) for v in ys], "not_a_knot")]
            big = max(abs(s) for s in exact)
            moved = 0.0
            for _ in range(40):
                yp = [float(t(v) + rng.choice([-1, 1]) * np.spacing(t(v))) for v in ys]
                e = [float(s) for s in c2_slopes(fx, [Fr(v) for v in yp], "not_a_knot")]
                moved = max(moved, max(abs(a - b) for a, b in zip(e, exact)) / big / eps)
            if not f32:
                sp = CubicSpline(np.array(xs), np.array(ys), bc_type="not-a-knot")(np.array(xs), 1)
                sci = max(abs(float(a) - b) for a, b in zip(sp, exact)) / big / eps
                print(f"not-a-knot {kind} {len(xs)} knots, shortest interval {min(b - a for a, b in zip(xs, xs[1:])):.0e}: one ulp moves the slopes {moved:.1f} eps, SciPy is {sci:.3g} eps off")
            out.append(f"    NakCase {{ f32: {'true' if f32 else 'false'}, x: {arr(xs, f32)}, y: {arr(ys, f32)},")
            out.append(f"        slopes: {arr(exact)}, sensitivity: {lit(moved, False)} }},")
    out.append("];")


def cheb_cases(out):
    funcs = {
        "exp": (lambda v: mpmath.exp(v), np.exp, 0.0, 1.0, (0, 1, 2, 12)),
        "runge": (lambda v: 1 / (1 + 25 * v * v), lambda v: 1 / (1 + 25 * v * v), -1.0, 1.0, (40,)),
        "sin3exp": (lambda v: mpmath.sin(3 * v) * mpmath.exp(v), lambda v: np.sin(3 * v) * np.exp(v), -1.0, 1.0, (20,)),
        "cos20": (lambda v: mpmath.cos(20 * v), lambda v: np.cos(20 * v), -3.0, 7.0, (100,)),
    }
    out.append("/// Chebyshev interpolants: (function, a, b, coefficients, points, values, derivatives,")
    out.append("/// integrals from a), the exact interpolant of the function at the first-kind points.")
    out.append("pub const CHEBYSHEV: &[ChebCase] = &[")
    for name, (fm, fn, a, b, degrees) in funcs.items():
        for deg in degrees:
            n = deg + 1
            # nodes and values in high precision; the crate's nodes are within an ulp of these
            nodes = [mpmath.mpf(a) + (mpmath.mpf(b) - a) / 2 * (1 - mpmath.cos(mpmath.pi * (2 * j + 1) / (2 * n))) for j in range(n)]
            vals = [fm(v) for v in nodes]
            coef = []
            for k in range(n):
                s = mpmath.fsum(v * mpmath.cos(k * mpmath.pi * (2 * j + 1) / (2 * n)) * (-1) ** k for j, v in enumerate(vals))
                coef.append(s / n if k == 0 else 2 * s / n)
            ref = npc.Chebyshev.interpolate(fn, deg, domain=[a, b])
            diff = max(abs(float(c) - float(r)) for c, r in zip(coef, ref.coef))
            pts = [a, b, a + 0.3 * (b - a), a + 0.71 * (b - a), a + 0.5 * (b - a), b + 0.05 * (b - a)]
            u = [(2 * mpmath.mpf(p) - a - b) / (b - a) for p in pts]

            def T(k, z):
                return mpmath.chebyt(k, z)

            val = [mpmath.fsum(c * T(k, z) for k, c in enumerate(coef)) for z in u]
            der = [mpmath.fsum(c * k * mpmath.chebyu(k - 1, z) for k, c in enumerate(coef) if k > 0) * 2 / (b - a) for z in u]
            # the integral from a: the primitive of T_k is known in closed form
            def prim(z):
                s = coef[0] * z
                if n > 1:
                    s += coef[1] * z * z / 2
                for k in range(2, n):
                    s += coef[k] * (T(k + 1, z) / (k + 1) - T(k - 1, z) / (k - 1)) / 2
                return s

            itg = [(prim(z) - prim(mpmath.mpf(-1))) * (b - a) / 2 for z in u]
            print(f"chebyshev {name} degree {deg}: numpy's coefficients {diff:.1e} from the exact ones")
            out.append(f"    ChebCase {{ name: {name!r:s}, a: {lit(a, False)}, b: {lit(b, False)}, coefficients: {arr([float(c) for c in coef])},".replace("'", '"'))
            out.append(f"        at: {arr(pts)}, value: {arr([float(v) for v in val])}, derivative: {arr([float(v) for v in der])}, integral: {arr([float(v) for v in itg])} }},")
    out.append("];")


def main():
    out = ["// Generated by gen_interp.py; do not edit.", ""]
    spline_cases(out, False)
    out.append("")
    spline_cases(out, True)
    out.append("")
    nak_ratio(out)
    out.append("")
    nak_short(out)
    out.append("")
    cheb_cases(out)
    with open("interp_ref.rs", "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
