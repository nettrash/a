#!/usr/bin/env python3
"""Reference values for the dual-number tests: each function and its derivatives, by SymPy's
symbolic differentiation, evaluated by mpmath, at points on grids that include the edges of each
function (near singularities, branch points, overflow, cancellation).

SymPy's forms cancel where the functions do not: 1 − tanh²x for the derivative of tanh, e^x − 1
for exp_m1 at 1e−300. So each value is computed at 600 significant digits and again at 700, and
the two must round to the same f64: at least 50 correct digits after any cancellation (the
largest here, 1 − tanh²(−350) ≈ 10⁻³⁰⁴, leaves about 290).

Run from this directory: python3 gen_reference.py > reference.txt

Each line is one point. A point marked `b` is a float of both `f32` and `f64` (so the same line
tests both); `d` is an `f64` only. Values are the 50-digit results rounded to the nearest `f64`
(Python's repr, which reads back exactly); `inf` and `nan` where the value is out of range or not
real, which the tests skip.

  u <name> <param> <flag> x f f' f'' f'''           a function of one variable
  b <name> <flag> x y f fx fy fxx fxy fyy           a function of two: f(x, y)
  g <name> <flag> x y z f gx gy gz hxx hxy hxz hyy hyz hzz   one of three, with its gradient
                                                          and Hessian
"""

import math

import mpmath
import numpy as np
import sympy as sp

mpmath.mp.dps = 600
X, Y, Z = sp.symbols("x y z", real=True)


def f32(v):
    with np.errstate(over="ignore"):
        return float(np.float32(v))


def flag(v):
    return "b" if f32(v) == v else "d"


def num(v):
    """The 50-digit value as the nearest f64, written to read back exactly."""
    if v is None:
        return "nan"
    try:
        c = mpmath.mpmathify(v)
    except Exception:
        return "nan"
    if isinstance(c, mpmath.mpc):
        if abs(c.imag) > mpmath.mpf(10) ** -400 * max(1, abs(c.real)):
            return "nan"
        c = c.real
    if mpmath.isnan(c):
        return "nan"
    if mpmath.isinf(c):
        return "inf" if c > 0 else "-inf"
    f = float(c)
    return repr(f)


def at_two_precisions(f, *args):
    """f(*args) at 600 and at 700 digits, which must agree to the f64 they round to."""
    out = []
    for dps in (600, 700):
        with mpmath.workdps(dps):
            try:
                out.append(num(f(*[mpmath.mpf(a) for a in args])))
            except (ZeroDivisionError, ValueError):
                out.append("nan")
    assert out[0] == out[1], (f, args, out)
    return out[0]


def unary(name, expr, points, param="-", odd=None, value=None):
    """Writes f, f', f'', f''' of `expr` at the points. `odd` is the symmetry used where SymPy's
    principal branch is not the real function (the real cube root of a negative number). `value`
    is the float of the parameter: a point tests f32 too only where the parameter is an f32."""
    ds = [expr]
    for _ in range(3):
        ds.append(sp.diff(ds[-1], X))
    fs = [sp.lambdify(X, d, modules="mpmath") for d in ds]
    for x in points:
        if odd is not None and x < 0:
            # f(−x) = −f(x): the k-th derivative has the sign (−1)^(k+1).
            vals = [at_two_precisions(lambda t, k=k: -(fs[k](-t)) * (-1) ** k, x) for k in range(4)]
        else:
            vals = [at_two_precisions(fs[k], x) for k in range(4)]
        fl = flag(x) if value is None or flag(value) == "b" else "d"
        print("u", name, param, fl, repr(x), *vals)


def binary(name, expr, points):
    fx, fy = sp.diff(expr, X), sp.diff(expr, Y)
    parts = [expr, fx, fy, sp.diff(fx, X), sp.diff(fx, Y), sp.diff(fy, Y)]
    fs = [sp.lambdify((X, Y), d, modules="mpmath") for d in parts]
    for x, y in points:
        fl = "b" if flag(x) == "b" and flag(y) == "b" else "d"
        vals = [at_two_precisions(f, x, y) for f in fs]
        print("b", name, fl, repr(x), repr(y), *vals)


def both(*vs):
    return [f32(v) for v in vs]


one_m = [1 - 2.0**-24, 1 - 2.0**-53]  # just below 1, in f32 and in f64

unary("recip", 1 / X, both(-3, -0.5, -1e-3, 1e-20, 0.1, 0.75, 1, 2.5, 1e10, 1e30) + [1e-150, 1e300])
unary("sqrt", sp.sqrt(X), both(1e-30, 1e-6, 0.1, 0.5, 1, 2, 3, 100, 1e20, 1e30) + [1e-300, 1e300])
unary("cbrt", sp.root(X, 3), both(-8, -0.3, -1e-9, 1e-9, 0.3, 1, 2, 27, 1e20) + [1e-300, -1e300, 5.102799809411062e-186], odd=True)
unary("exp", sp.exp(X), both(-87, -20, -1, -1e-8, 0, 1e-8, 0.5, 1, 10, 88) + [-700, 700])
unary("exp2", 2**X, both(-126, -10.5, -1, 0, 0.25, 1, 3.5, 20, 127) + [-1000, 1000])
unary("exp_m1", sp.exp(X) - 1, both(-30, -1, -1e-3, -1e-10, 1e-20, 1e-10, 1e-3, 0.5, 1, 10, 80) + [1e-300, 700])
unary("ln", sp.log(X), both(1e-30, 1e-5, 0.1, 0.5, 1 - 2.0**-24, 1, 1 + 2.0**-23, 2, 10, 1e30) + [1e-300, 1e300])
unary("log2", sp.log(X, 2), both(1e-20, 0.3, 1, 1 + 2.0**-23, 2, 3, 1024, 1e30) + [1e-300, 1e300])
unary("log10", sp.log(X, 10), both(1e-20, 0.3, 1, 1 + 2.0**-23, 10, 3, 1e5, 1e30, 4.4082306305806286e-20) + [1e-300, 1e300])
unary("ln_1p", sp.log(1 + X), both(-1 + 2.0**-24, -0.5, -1e-3, -1e-10, 1e-20, 1e-10, 1e-3, 0.5, 1, 1e10) + [-1 + 2.0**-53, 1e300])
unary("sin", sp.sin(X), both(-10, -1, -1e-8, 0, 1e-8, 0.5, 1, 1.5707964, 3.1415927, 100, 1e5) + [math.pi, 1e15])
unary("cos", sp.cos(X), both(-10, -1, -1e-8, 0, 1e-8, 0.5, 1, 1.5707964, 3.1415927, 100, 1e5) + [math.pi / 2, 1e15])
unary("tan", sp.tan(X), both(-1.5, -1, -1e-8, 0, 1e-8, 0.5, 1, 1.5, 1.5707963, 10, 1e4) + [math.pi / 2, 1.5707963267948])
unary("asin", sp.asin(X), both(-1 + 2.0**-24, -0.9, -0.5, -1e-8, 0, 1e-8, 0.3, 0.7, 0.99, 1 - 2.0**-24) + one_m[1:] + [-1 + 2.0**-53])
unary("acos", sp.acos(X), both(-1 + 2.0**-24, -0.9, -0.5, -1e-8, 0, 1e-8, 0.3, 0.7, 0.99, 1 - 2.0**-24) + one_m[1:] + [-1 + 2.0**-53])
unary("atan", sp.atan(X), both(-1e20, -10, -1, -1e-8, 0, 1e-8, 0.5, 1, 3, 1e5, 1e20) + [1e150, -1e300])
unary("sinh", sp.sinh(X), both(-89, -10, -1, -1e-8, 0, 1e-8, 0.5, 1, 5, 20, 89) + [-710, 710])
unary("cosh", sp.cosh(X), both(-89, -10, -1, -1e-8, 0, 1e-8, 0.5, 1, 5, 20, 89) + [-710, 710])
unary("tanh", sp.tanh(X), both(-40, -19, -5, -1, -1e-8, 0, 1e-8, 0.5, 1, 5, 10, 19.5, 40) + [100, -350])
unary("asinh", sp.asinh(X), both(-1e20, -100, -1, -1e-8, 0, 1e-8, 0.5, 1, 3, 1e5, 1e20, 1e30) + [1e200, -1e300])
unary("acosh", sp.acosh(X), both(1 + 2.0**-23, 1.001, 1.1, 1.5, 2, 10, 1e5, 1e20, 1e30) + [1 + 2.0**-52, 1e200, 1e300])
unary("atanh", sp.atanh(X), both(-1 + 2.0**-24, -0.9, -0.5, -1e-8, 0, 1e-8, 0.3, 0.7, 0.99, 1 - 2.0**-24) + [1 - 2.0**-53, -1 + 2.0**-53])
for n in [-3, 0, 1, 2, 5, 31]:
    unary("powi", X**n, both(-1.5, -0.7, 0.3, 1, 1.25, 2.5) + [1e-10, 1e10], param=str(n))
# The exponent is the float itself, as an exact rational: 0.3 is 0.29999999999999998890 in f64
# (and 0.30000001192 in f32, which is another exponent and another line), not 3/10.
for e in [0.5, -1.5, 2.5, 0.3]:
    unary("powf", X**sp.Rational(e), both(1e-6, 0.3, 1, 1.5, 7, 1e6), param=repr(e), value=e)
# Exponents whose e − 1 is not a float, of f64 and of f32 precision, over a wide range of ln x.
for e in [0.1, 1 / 3, -7.5 + 2.0**-50, 1 + 2.0**-30]:
    unary("powf", X**sp.Rational(e), both(1e-6, 0.3, 1.5, 7, 1e3, 1e6) + [1e-300, 1e300], param=repr(e), value=e)
for e in [f32(0.3), f32(math.e)]:
    unary("powf", X**sp.Rational(e), both(1e-6, 0.3, 1.5, 7, 1e3, 1e6), param=repr(e), value=e)
unary("powf", X**3, both(-2.5, -0.5, 0.5, 2.5), param="3.0")
unary("powf", X**-2, both(-2.5, -0.5, 0.5, 2.5), param="-2.0")
for b in [3.0, 0.5]:
    unary("log", sp.log(X) / sp.log(sp.Rational(b)), both(1e-20, 0.3, 1, 2, 7, 1e20), param=repr(b), value=b)
# A small base at the bottom of the range, where 1/x² alone overflows before f″ does.
b = 0.026980563021900154
unary("log", sp.log(X) / sp.log(sp.Rational(b)), [4.437317020667586e-155, 1e-20, 0.3, 7.0], param=repr(b), value=b)
# Negative powers whose derivatives std's powi, as 1/x^|m|, would give as 0.
unary("powi", X**-12, both(978.71948, 0.7, 2.5), param="-12")
unary("powi", X**-41, [-1.5176770336069122e7, 1.5], param="-41")

binary("atan2", sp.atan2(X, Y), [
    (1, 1), (1, -1), (-1, -1), (-1, 1), (0.5, 2), (2, 0.5), (1, 0), (-1, 0), (0, 1), (0, -1),
    (1e-20, 1), (1, 1e-20), (3, -4), (f32(1e30), f32(-2e30)), (1 + 2.0**-23, 1)]
    + [(1e-300, 2e-300), (1e300, -1e300)]
    # Next to the diagonals, where the mixed partial (y² − x²)/r⁴ is small and cancels.
    + [(1 + 2.0**-52, 1), (3 * (1 + 2.0**-50), 3), (1 + 2.0**-52, -1), (1e-300 * (1 + 2.0**-52), 1e-300)])
binary("hypot", sp.sqrt(X**2 + Y**2), [
    (3, 4), (-3, 4), (1, 1e-10), (1e-10, 1), (0.5, -0.25), (1, 0), (0, -2), (f32(1e30), f32(1e30)),
    (f32(1e-10), f32(1e-30))]
    + [(1e300, 1e300), (1e-300, 3e-300)]
    # Extreme ratios, where the square of the small ratio underflows though y²/h³ does not.
    + [(-3.0752355837317934e-81, -1.6847603082516674e-255), (1e-100, 1e-260)])
binary("pow", X**Y, [
    (2, 3), (2, 0.5), (0.5, -1.5), (3, 0), (1, 2.5), (10, -2), (0.1, 0.1), (7, 1.25), (f32(1e10), 0.25)]
    + [(1e100, 2.5), (1e-100, 0.5)]
    # x^y underflowing at x ≠ 0; an inexact y − 1 with a large ln x; and powers x^(y−1) that
    # leave the range while their products with y or with 1 + y·ln x do not.
    + [(1e-200, 2), (1e-100, 3.5), (1e300, 0.1), (1000, -7.5 + 2.0**-50),
       (8.470260661517206e-194, -0.5976377187252175), (1.58535383168637e33, -8.350589541959247)])

# Compositions, written the same way in the tests (compositions.rs).
compositions = {
    "g1": sp.sin(sp.exp(X)) / (1 + X**2),
    "g2": sp.log(1 + X**2) * sp.atan(X) ** 3,
    "g3": sp.sqrt(sp.cosh(X)) * sp.tanh(X / 2) - X ** sp.Rational(5, 2),
    "g4": sp.exp(-X**2) * sp.cos(3 * X),
    "g5": sp.asinh(X**3 - X) / (2 + sp.sin(X)),
    "g6": sp.atan2(sp.sin(X), sp.cos(2 * X) + 2) * sp.sqrt(1 + X**2),
}
for name, expr in compositions.items():
    pts = both(0.1, 0.5, 1, 1.75, 2.5, 4)
    if name != "g3":
        pts = both(-2.5, -0.5) + pts
    unary(name, expr, pts)

# A function of three variables for the gradient and the Hessian, written the same way in the
# tests (derivatives.rs).
F = sp.exp(X * Y) * sp.sin(Z) + sp.log(1 + X**2 + Y**2) * Z + sp.atan(X - Z**2)
V = (X, Y, Z)
parts = [F] + [sp.diff(F, v) for v in V]
parts += [sp.diff(F, V[i], V[j]) for i in range(3) for j in range(i, 3)]
fs = [sp.lambdify(V, d, modules="mpmath") for d in parts]
for pt in [(0.5, -0.25, 1.0), (1.5, 0.75, -0.5), (-1.0, 2.0, 0.25), (0.0, 0.0, 0.0)]:
    fl = "b" if all(flag(v) == "b" for v in pt) else "d"
    vals = [at_two_precisions(f, *pt) for f in fs]
    print("g", "F", fl, *[repr(v) for v in pt], *vals)
