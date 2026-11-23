"""Reference nodes and weights of the Gaussian rules, for lib/tests/math/numerics/gauss/.

Writes gauss_ref.txt next to this script. Every zero is found by Newton's method on the
orthonormal three-term recurrence in mpmath at 60 digits, started from the eigenvalues of the
Jacobi matrix (numpy) or, for the largest orders, from the asymptotic formulas written out below,
and its index is then proved by a Sturm count: the number of sign changes of p_0(x), ..., p_n(x)
is the number of zeros of p_n above x. The weights are the Christoffel numbers
mu0 / sum_{k<n} p_k(x)^2 with p_0 = 1, evaluated at the 60-digit zero; the scaled weights of
Hermite and Laguerre are those times e^(x^2) and e^x.

Lines: `legendre|hermite|laguerre n k node weight [scaled]` and
`jacobi alpha beta n k node weight`, k the 0-based index in ascending order of the nodes,
numbers to 30 significant digits (a weight below the range of an f64 is written as it is).

Run: python3 gen_gauss.py   (about two minutes; numpy, scipy, mpmath)
"""
import os
import numpy as np
import mpmath as mp
from scipy.special import jn_zeros, ai_zeros

mp.mp.dps = 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauss_ref.txt")


def coefficients(family, n, alpha=None, beta=None):
    """Diagonal a[0..n-1], off-diagonal b[1..n] (b[0] = 0), total weight mu0, all mpf."""
    a = [mp.mpf(0)] * n
    b = [mp.mpf(0)] * (n + 1)
    if family == "legendre":
        for k in range(1, n + 1):
            b[k] = k / mp.sqrt(4 * mp.mpf(k) ** 2 - 1)
        mu0 = mp.mpf(2)
    elif family == "hermite":
        for k in range(1, n + 1):
            b[k] = mp.sqrt(mp.mpf(k) / 2)
        mu0 = mp.sqrt(mp.pi)
    elif family == "laguerre":
        for k in range(n):
            a[k] = mp.mpf(2 * k + 1)
        for k in range(1, n + 1):
            b[k] = mp.mpf(k)
        mu0 = mp.mpf(1)
    elif family == "jacobi":
        al, be = mp.mpf(alpha), mp.mpf(beta)
        ab = al + be
        a[0] = (be - al) / (ab + 2)
        for k in range(1, n):
            s = 2 * k + ab
            a[k] = (be * be - al * al) / (s * (s + 2))
        for k in range(1, n + 1):
            s = 2 * k + ab
            if k == 1:
                b2 = 4 * (1 + al) * (1 + be) / (s * s * (s + 1))
            else:
                b2 = 4 * k * (k + al) * (k + be) * (k + ab) / (s * s * (s + 1) * (s - 1))
            b[k] = mp.sqrt(b2)
        mu0 = mp.power(2, ab + 1) * mp.beta(al + 1, be + 1)
    else:
        raise ValueError(family)
    return a, b, mu0


def run(a, b, n, x):
    """p_n(x), p_n'(x), sum_{k<n} p_k(x)^2, and the sign changes of p_0..p_n (p_0 = 1)."""
    p0, p1 = mp.mpf(0), mp.mpf(1)
    d0, d1 = mp.mpf(0), mp.mpf(0)
    s = mp.mpf(0)
    changes = 0
    for k in range(n):
        s += p1 * p1
        xa = x - a[k]
        p2 = (xa * p1 - b[k] * p0) / b[k + 1]
        d2 = (p1 + xa * d1 - b[k] * d0) / b[k + 1]
        if p2 * p1 < 0:
            changes += 1
        p0, p1, d0, d1 = p1, p2, d1, d2
    return p1, d1, s, changes


def zero(a, b, n, guess, k):
    """The k-th zero (0-based, ascending) by Newton from guess, checked by the Sturm count."""
    x = mp.mpf(guess)
    # The recurrence's last step cancels near the largest zeros, which leaves the step good to
    # about n * 10^-60 of the zero: 10^-45 is far inside both that and 30 digits.
    for _ in range(100):
        p, dp, _, _ = run(a, b, n, x)
        step = p / dp
        x -= step
        if abs(step) <= mp.mpf(10) ** -45 * max(abs(x), mp.mpf(10) ** -30):
            break
    else:
        raise RuntimeError("no convergence")
    delta = mp.mpf(10) ** -40 * max(abs(x), mp.mpf(1))
    above_lo = run(a, b, n, x - delta)[3]
    above_hi = run(a, b, n, x + delta)[3]
    want = n - 1 - k
    if above_lo != want + 1 or above_hi != want:
        raise RuntimeError(f"zero {k} of degree {n}: Sturm counts {above_lo}, {above_hi}")
    _, _, s, _ = run(a, b, n, x)
    return x, s


def eigen_guesses(a, b, n):
    j = np.zeros((n, n))
    for k in range(n):
        j[k, k] = float(a[k])
        if k + 1 < n:
            j[k, k + 1] = j[k + 1, k] = float(b[k + 1])
    return np.linalg.eigvalsh(j)


def airy_abs(k):
    t = 3 * np.pi * (4 * np.asarray(k, dtype=float) - 1) / 8
    return t ** (2 / 3) * (1 + 5 / 48 * t**-2 - 5 / 36 * t**-4 + 77125 / 82944 * t**-6
                           - 108056875 / 6967296 * t**-8)


def solve_increasing(f, df, r, lo, hi):
    x = 0.5 * (lo + hi)
    for _ in range(200):
        v = f(x) - r
        if v < 0:
            lo = x
        else:
            hi = x
        nx = x - v / df(x) if df(x) != 0 else 0.5 * (lo + hi)
        if not (lo < nx < hi):
            nx = 0.5 * (lo + hi)
        if abs(nx - x) < 1e-16 * max(1.0, abs(x)):
            return nx
        x = nx
    return x


def asymptotic_guesses(family, n):
    """Ascending first guesses for the largest orders (independent of the library's code)."""
    if family == "legendre":
        rho = n + 0.5
        j = jn_zeros(0, n // 2)
        th = j / rho
        th = th + (th / np.tan(th) - 1) / (8 * th * rho**2)
        pos = np.cos(th)  # descending
        mid = [0.0] if n % 2 else []
        return np.concatenate([-pos, mid, pos[::-1]])
    if family == "hermite":
        nu = 2 * n + 1
        k = np.arange(1, n // 2 + 1)
        r = (8 / 3) * airy_abs(k) ** 1.5 / nu
        phi = np.array([solve_increasing(lambda p: 2 * p - np.sin(2 * p),
                                         lambda p: 2 - 2 * np.cos(2 * p), ri, 0.0, np.pi / 2)
                        for ri in r])
        pos = np.sqrt(nu) * np.cos(phi)
        mid = [0.0] if n % 2 else []
        return np.concatenate([-pos, mid, pos[::-1]])
    if family == "laguerre":
        nu = 4 * n + 2
        i = np.arange(1, n + 1)
        r = (8 / 3) * airy_abs(i) ** 1.5 / nu
        phi = np.array([solve_increasing(lambda p: p - np.sin(p), lambda p: 1 - np.cos(p), ri,
                                         0.0, np.pi) for ri in r])
        x = (nu * np.cos(phi / 2) ** 2)[::-1]
        m = int(np.sqrt(n)) + 4
        j = jn_zeros(0, m)
        x[:m] = j**2 / nu * (1 + (j**2 - 2) / (3 * nu**2))
        return x
    raise ValueError(family)


def indices(n, full):
    if n <= full:
        return list(range(n))
    ks = {0, 1, 2, 3, n // 4, n // 2 - 1, n // 2, n // 2 + 1, n - n // 4, n - 4, n - 3, n - 2,
          n - 1}
    return sorted(k for k in ks if 0 <= k < n)


def fmt(x):
    return mp.nstr(x, 30, min_fixed=0, max_fixed=0) if x != 0 else "0"


def main():
    lines = []
    orders = list(range(1, 21)) + [25, 30, 40, 50, 64, 99, 100, 101, 128, 200, 500, 1000, 2000,
                                    5000, 10000]
    for family in ("legendre", "hermite", "laguerre"):
        for n in orders:
            a, b, mu0 = coefficients(family, n)
            g = eigen_guesses(a, b, n) if n <= 2000 else asymptotic_guesses(family, n)
            for k in indices(n, 64):
                x, s = zero(a, b, n, g[k], k)
                w = mu0 / s
                row = [family, str(n), str(k), fmt(x), fmt(w)]
                if family == "hermite":
                    row.append(fmt(w * mp.exp(x * x)))
                elif family == "laguerre":
                    row.append(fmt(w * mp.exp(x)))
                lines.append(" ".join(row))
        print(family, "done", flush=True)
    pairs = [(-0.999, -0.999), (-0.9, 0.5), (-0.5, -0.5), (-0.5, 0.5), (0.0, 0.0), (0.5, 0.5),
             (1.0, 0.0), (0.0, 1.0), (1.5, 0.25), (2.0, 3.0), (5.0, 5.0), (10.0, 20.0),
             (20.0, 10.0), (0.3, 150.0), (100.0, 0.0), (600.0, 600.0)]
    for alpha, beta in pairs:
        for n in [1, 2, 3, 5, 10, 20, 50, 113, 200]:
            a, b, mu0 = coefficients("jacobi", n, alpha, beta)
            g = eigen_guesses(a, b, n)
            for k in indices(n, 20):
                x, s = zero(a, b, n, g[k], k)
                lines.append(" ".join(["jacobi", repr(alpha), repr(beta), str(n), str(k), fmt(x),
                                       fmt(mu0 / s)]))
    print("jacobi done", flush=True)
    with open(OUT, "w") as f:
        f.write("# Reference Gaussian rules: see gen_gauss.py.\n")
        f.write("\n".join(lines) + "\n")
    print(len(lines), "lines written to", OUT)


if __name__ == "__main__":
    main()
