"""Reference data for lib/tests/math/numerics/root.rs: SciPy's brentq on the bracket battery, and a
model of the crate's bisect and brent in Python, step for step, so that the tests can hold the Rust
code to the algorithm it documents.

Run with python3 (NumPy and SciPy installed): python3 gen_roots.py > roots_table.txt
The table printed is pasted into root.rs as TABLE. Each function is written so that Python and
Rust evaluate it in the same operations (powers by repeated products, the libm functions alike),
which makes the iterates, not only the roots, comparable bit for bit.

That holds as it stands only for the functions made of IEEE arithmetic (+, -, *, / and the square
root, correctly rounded everywhere). The others call the platform's libm (cos, sin, tan, exp, pow,
cbrt), whose last bits differ between platforms (Apple's against glibc's), and a last bit decides
the last steps. So the script also writes roots_libm.txt beside itself: for each of those
brackets (LIBM below) and each of the model's three runs (brent at the defaults, brent at xtol
2e-12, bisect at the defaults), every point the model evaluated, in order, with the value it got
there. The tests replay those values, and so compare the steps on the values the model saw on any
platform. The script was run on aarch64 macOS (Apple's libm), as were SciPy's runs in the table.

Columns: name, a, b, brentq evaluations and root (xtol 2e-12, rtol 4 eps), brentq evaluations and
root (xtol 5e-324, the least it accepts), the model's evaluations and root with the crate's
defaults (xtol 0, rtol 4 eps), and the model's bisect evaluations and root with the defaults. A
method that did not converge within 10,000 iterations has 0 evaluations and a NaN root.
"""
import math
import struct
import sys

import numpy as np
from scipy.optimize import brentq

np.seterr(all="ignore")
EPS = 2.0 ** -52
TINY = 5e-324
MIN_POS = 2.0 ** -1022
MAXF = sys.float_info.max
D = np.float64


def pw(x, n):
    r = 1.0
    for _ in range(n):
        r *= x
    return r


def cases():
    pi = math.pi
    v = []

    def c(name, a, b, f):
        v.append((name, a, b, f))

    c("sqrt2", 1.0, 2.0, lambda x: x * x - 2.0)
    c("dottie", 0.0, 1.0, lambda x: math.cos(x) - x)
    c("cubic", 1.0, 2.0, lambda x: x * x * x - x - 2.0)
    c("wallis", 2.0, 3.0, lambda x: x * x * x - 2.0 * x - 5.0)
    c("exp2", 0.0, 1.0, lambda x: math.exp(x) - 2.0)
    c("aps01", pi / 2.0, pi, lambda x: math.sin(x) - x / 2.0)
    for n in [1, 2, 5]:
        a = float(n * n) + 1e-9
        b = float((n + 1) * (n + 1)) - 1e-9

        def f(x):
            s = 0.0
            for i in range(1, 21):
                i = float(i)
                d = x - i * i
                s += (2.0 * i - 5.0) * (2.0 * i - 5.0) / (d * d * d)
            return -2.0 * s

        c(f"aps02_{n}", a, b, f)
    c("aps03_a", -9.0, 31.0, lambda x: -40.0 * x * math.exp(-x))
    c("aps03_c", -9.0, 31.0, lambda x: -200.0 * x * math.exp(-3.0 * x))
    for n, a in [(4, 0.2), (12, 1.0)]:
        c(f"aps04_{n}_{a}", 0.0, 5.0, lambda x, n=n, a=a: pw(x, n) - a)
    c("aps05", 0.0, 1.5, lambda x: math.sin(x) - 0.5)
    for n in [1.0, 20.0, 100.0]:
        c(f"aps06_{int(n)}", 0.0, 1.0, lambda x, n=n: 2.0 * x * math.exp(-n) - 2.0 * math.exp(-n * x) + 1.0)
    for n in [5.0, 20.0]:
        c(f"aps07_{int(n)}", 0.0, 1.0, lambda x, n=n: (1.0 + (1.0 - n) * (1.0 - n)) * x - (1.0 - n * x) * (1.0 - n * x))
    for n in [2, 10, 20]:
        c(f"aps08_{n}", 0.0, 1.0, lambda x, n=n: x * x - pw(1.0 - x, n))
    for n in [1, 5, 20]:
        c(f"aps09_{n}", 0.0, 1.0, lambda x, n=float(n): (1.0 + pw(1.0 - n, 4)) * x - pw(1.0 - n * x, 4))
    for n in [1, 15]:
        c(f"aps10_{n}", 0.0, 1.0, lambda x, n=n: math.exp(-float(n) * x) * (x - 1.0) + pw(x, n))
    for n in [2.0, 20.0]:
        c(f"aps11_{int(n)}", 0.01, 1.0, lambda x, n=n: (n * x - 1.0) / ((n - 1.0) * x))
    for n in [2.0, 9.0, 33.0]:
        c(f"aps12_{int(n)}", 1.0, 100.0, lambda x, n=n: math.pow(x, 1.0 / n) - math.pow(n, 1.0 / n))
    def aps13(x):
        # x e^(-1/x^2), with the IEEE answers where x^2 underflows: -1/0 = -inf, e^-inf = 0.
        t = x * x
        return 0.0 if x == 0.0 else x * (math.exp(-1.0 / t) if t != 0.0 else 0.0)

    c("aps13", -1.0, 4.0, aps13)
    for n in [1.0, 40.0]:
        c(f"aps14_{int(n)}", -1e4, pi / 2.0, lambda x, n=n: (n / 20.0) * (x / 1.5 + math.sin(x) - 1.0) if x >= 0.0 else -n / 20.0)
    for n in [20.0, 1000.0]:
        def f(x, n=n):
            if x > 2e-3 / (1.0 + n):
                return math.e - 1.859
            if x >= 0.0:
                return math.exp((n + 1.0) * x / 2e-3) - 1.859
            return -0.859
        c(f"aps15_{int(n)}", -1e4, 1e-4, f)
    c("mult3", 0.5, 3.0, lambda x: pw(x - 1.0, 3))
    c("mult5", 0.0, 3.0, lambda x: pw(x - 1.0, 5))
    c("mult9", 0.0, 3.5, lambda x: pw(x - 1.0, 9))
    c("zero_x", -1.0, 2.0, lambda x: x)
    c("zero_x3", -1.0, 2.0, lambda x: x * x * x)
    c("zero_x9", -1.0, 2.0, lambda x: pw(x, 9))
    c("zero_sin", -1.0, 2.0, lambda x: math.sin(x))
    c("zero_cbrt", -1.0, 2.0, lambda x: math.cbrt(x))
    c("zero_xex", -9.0, 31.0, lambda x: -40.0 * x * math.exp(-x))
    c("flat_1e-6", 0.0, 1.0, lambda x: (x - 0.3) * 1e-6)
    c("flat_1e-300", 0.0, 1.0, lambda x: (x - 0.3) * 1e-300)
    c("dyn_exp", -700.0, 700.0, lambda x: math.exp(x) - math.exp(-600.0))
    c("dyn_exp50", -10.0, 14.0, lambda x: math.exp(50.0 * x) - 5.0)
    c("big_lin", 0.0, MAXF, lambda x: x - 1e308)
    c("big_sym", -MAXF, MAXF, lambda x: x / 1e308 - 1.0)
    c("small_sq", 0.0, 1.0, lambda x: x * x - 1e-300)
    c("small_sub", 0.0, 1e-300, lambda x: math.sqrt(x) - 1e-155)
    c("wide_sq", 0.0, 1e20, lambda x: x * x - 2.0)
    c("wide_lin", 1e-6, 1e9, lambda x: x - 1234.5)
    c("pole_tan", 1.0, 2.0, lambda x: math.tan(x))
    c("pole_inv", 0.0, 1.0, lambda x: float(D(1.0) / D(x - 0.3)))
    c("end_a", 1.0, 2.0, lambda x: x - 1.0)
    c("near_end", 1.0, 2.0, lambda x: x - (1.0 + EPS))
    c("step", 0.0, 1.0, lambda x: -1.0 if x < 0.3 else 1.0)
    c("kink", 0.0, 1.0, lambda x: abs(x - 0.3) + x - 0.5)
    c("rev_sqrt2", 2.0, 1.0, lambda x: x * x - 2.0)
    return v


# ---- the model of the crate's code ----------------------------------------------------------

def ordinal(x):
    b = struct.unpack("<q", struct.pack("<d", x))[0]
    return -(b & 0x7FFFFFFFFFFFFFFF) if b < 0 else b


def from_ordinal(k):
    x = struct.unpack("<d", struct.pack("<q", abs(k)))[0]
    return -x if k < 0 else x


def spacing(x):
    return max(abs(x) * EPS, MIN_POS * EPS)


def midpoint(a, b):
    if abs(a) <= 1.0 and abs(b) <= 1.0:
        return (a + b) * 0.5
    return a * 0.5 + b * 0.5


def middle_float(lo, hi):
    # Halved with truncation towards zero, as Rust's division of an i128.
    k = ordinal(lo) + ordinal(hi)
    q = abs(k) // 2
    return from_ordinal(q if k >= 0 else -q)


def spans(lo, hi):
    if lo <= 0.0 and hi >= 0.0:
        return True
    a, b = abs(lo), abs(hi)
    return max(a, b) > 2.0 * min(a, b)


def reachable(lo, hi, xtol):
    return xtol > 0.0 and hi * 0.5 - lo * 0.5 <= xtol * 2.0 ** 63


class Halver:
    def __init__(self):
        self.took_midpoint = False

    def point(self, lo, hi, xtol):
        m = midpoint(lo, hi)
        if not spans(lo, hi) or reachable(lo, hi, xtol):
            return m
        self.took_midpoint = not self.took_midpoint
        return m if self.took_midpoint else middle_float(lo, hi)


def count(x, y):
    return float(abs(ordinal(y) - ordinal(x)))


def model_bisect(f, a, b, xtol=0.0, rtol=4 * EPS, ftol=0.0, max_iter=512):
    n = 0
    lo, hi = (a, b) if a <= b else (b, a)
    flo = f(lo); n += 1
    if flo == 0.0:
        return lo, n
    fhi = f(hi); n += 1
    if fhi == 0.0:
        return hi, n
    if (flo < 0) == (fhi < 0):
        return None, n
    hv = Halver()
    for _ in range(max_iter):
        m = hv.point(lo, hi, xtol)
        if m <= lo or m >= hi:
            return (lo if abs(flo) <= abs(fhi) else hi), n
        fm = f(m); n += 1
        if fm == 0.0:
            return m, n
        if max(m - lo, hi - m) <= xtol + rtol * abs(m):
            return m, n
        if (fm < 0) == (flo < 0):
            lo, flo = m, fm
        else:
            hi, fhi = m, fm
    return None, n


def model_brent(f, a, b, xtol=0.0, rtol=4 * EPS, max_iter=512):
    n = 0
    fa = f(a); n += 1
    if fa == 0.0:
        return a, n
    fb = f(b); n += 1
    if fb == 0.0:
        return b, n
    if (fa < 0) == (fb < 0):
        return None, n
    c, fc = a, fa
    d = b - a
    e = d
    hv = Halver()
    watch = []
    it = 0
    while True:
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = (xtol + rtol * abs(b)) * 0.5
        m = (c - b) * 0.5 if abs(b) <= 1.0 and abs(c) <= 1.0 else c * 0.5 - b * 0.5
        if abs(m) <= tol:
            return b, n
        lo, hi = (b, c) if b < c else (c, b)
        mid = midpoint(lo, hi)
        if mid <= lo or mid >= hi:
            return b, n
        if it == max_iter:
            return None, n
        it += 1
        step = max(tol, spacing(b))
        extended = not reachable(lo, hi, xtol)
        stalled = len(watch) == 3 and watch[2] > 0.5 * watch[0]
        nxt = None
        if not (extended and stalled) and abs(e) >= step and abs(fa) > abs(fb) and math.isfinite(fa) and math.isfinite(fc):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                qa = fa / fc
                r = fb / fc
                p = s * (2.0 * m * qa * (qa - r) - (b - a) * (r - 1.0))
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            before = e
            e = d
            if 2.0 * p < 3.0 * m * q - abs(step * q) and p < abs(0.5 * before * q):
                d = p / q
                x = b + (d if abs(d) > step else math.copysign(step, m))
                if lo < x < hi:
                    nxt = x
        if nxt is None:
            nxt = hv.point(lo, hi, xtol)
            d = nxt - b
            e = d
            watch = []
        a, fa = b, fb
        b = nxt
        fb = f(b); n += 1
        if fb == 0.0:
            return b, n
        if (fb < 0) == (fc < 0):
            c, fc = a, fa
            d = b - a
            e = d
        watch.append(count(b, c))
        if len(watch) > 3:
            watch.pop(0)


def scipy(f, a, b, xtol):
    try:
        r, info = brentq(f, a, b, xtol=xtol, rtol=4 * EPS, maxiter=10000, full_output=True, disp=False)
        return (info.function_calls, r) if info.converged else (0, math.nan)
    except ValueError:
        return 0, math.nan


def hexf(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "f64::NAN"
    return f"f64::from_bits(0x{struct.unpack('<Q', struct.pack('<d', x))[0]:016x})"


# The brackets whose functions call the platform's libm, in the order of cases(). small_sub calls
# only the square root, which IEEE 754 rounds correctly on every platform.
LIBM = [
    "dottie", "exp2", "aps01", "aps03_a", "aps03_c", "aps05", "aps06_1", "aps06_20", "aps06_100",
    "aps10_1", "aps10_15", "aps12_2", "aps12_9", "aps12_33", "aps13", "aps14_1", "aps14_40",
    "aps15_20", "aps15_1000", "zero_sin", "zero_cbrt", "zero_xex", "dyn_exp", "dyn_exp50",
    "pole_tan",
]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def traced(f, trace):
    """f, recording each point it is evaluated at and the value there."""
    def g(x):
        v = f(x)
        trace.append((x, v))
        return v
    return g


if __name__ == "__main__":
    import os
    print("// name, brentq(2e-12): evaluations, root; brentq(5e-324); model brent(defaults); model bisect(defaults)")
    traces = []
    for name, a, b, f in cases():
        n1, r1 = scipy(f, a, b, 2e-12)
        n2, r2 = scipy(f, a, b, TINY)
        t3, t4, t5 = [], [], []
        r3, n3 = model_brent(traced(f, t3), a, b)
        r4, n4 = model_bisect(traced(f, t4), a, b)
        r5, n5 = model_brent(traced(f, t5), a, b, xtol=2e-12)
        assert (len(t3), len(t4), len(t5)) == (n3, n4, n5)
        print(f'    ("{name}", {n1}, {hexf(r1)}, {n2}, {hexf(r2)}, {n3 if r3 is not None else 0}, {hexf(r3)}, '
              f'{n4 if r4 is not None else 0}, {hexf(r4)}, {n5 if r5 is not None else 0}, {hexf(r5)}),')
        if name in LIBM:
            for run, t in [("brent", t3), ("xtol", t5), ("bisect", t4)]:
                traces.extend((name, run, x, v) for x, v in t)
    assert sorted({t[0] for t in traces}) == sorted(LIBM)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "roots_libm.txt"), "w") as out:
        out.write("# From gen_roots.py on aarch64 macOS (Apple's libm): for each bracket whose function calls\n")
        out.write("# the libm, and each run of the model (brent: brent at the defaults; xtol: brent at xtol\n")
        out.write("# 2e-12; bisect: bisect at the defaults), the points evaluated in order and the values there,\n")
        out.write("# as the bits of the f64s in hexadecimal: name run x f(x).\n")
        for name, run, x, v in traces:
            out.write(f"{name} {run} {bits(x):016x} {bits(v):016x}\n")
