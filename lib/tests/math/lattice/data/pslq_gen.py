#!/usr/bin/env python3
"""Reference outputs for the PSLQ tests (tests/math/lattice/pslq.rs), written to pslq_expected.txt.

Usage: python3 pslq_gen.py [out]   (needs mpmath 1.3; PARI/GP 2.19 as `gp` for the lindep column)

For every case the inputs are computed here as the Rust test computes them: each number at the
precision p + 64 and rounded once to p bits (mpmath's functions are accurate to a few ulps there,
so the rounding to p bits is the correctly rounded value of the crate's BigFloat functions but
with a chance of about 2^-60). The low 64 bits of each mantissa go into the file, and the test
checks its own inputs against them.

Then three independent answers:

* `ref`: Ferguson, Bailey and Arno's PSLQ written out here from the paper, with the crate's
  conventions (gamma, the order of the reductions, the exact y = X.B and the exact test of a
  relation |sum m_i X_i| <= sum |m_i| delta_i, the shortest such column of B, first nonzero
  entry positive) but H in mpmath floats of twice the precision rather than fixed point. For a
  search that stops at max_norm it gives the bound the crate proves, 1/max_j(|r_j| + |c_0| |delta|)
  from the Gram-Schmidt of (X, a_0, ..., a_{n-2}), computed here in floating point of three times
  the precision: the test checks the crate's bound to 1e-12 relative.
* `mpmath`: mpmath.pslq at the same precision with tol 2^(-p+20) (and, for the powers of an
  algebraic number of degree up to 16, mpmath.findpoly), or `none`; for the cases whose relation
  is unique up to sign (UNIQUE), as for `pari`.
* `pari`: PARI/GP's lindep (an LLL method) at the same precision.

Output lines: `input <case> <i> <hex>`, `ref <case> relation <m...>` or
`ref <case> nonebelow <bound>`, `mpmath <case> <m...>|none`, `findpoly <case> <m...>|none`,
`pari <case> <m...>`.
"""
import subprocess
import sys
from mpmath import mp, mpf, sqrt, root, atan, pi, e, log, zeta, euler, catalan, findpoly
import mpmath

GAMMA = 1.16
ERROR_BITS = 8
GUARD_BITS = 64


def rounded(f, p):
    """f() computed at p + 64 bits, rounded once to p."""
    with mp.workprec(p + 64):
        v = f()
    with mp.workprec(p):
        return +v


def alg(r1, k1, sign, r2, k2, deg, p):
    def a():
        return root(mpf(r1), k1) + sign * root(mpf(r2), k2)
    xs = []
    for k in range(deg + 1):
        xs.append(rounded(lambda: a() ** k, p))
    return xs


def bbp_s(j, p):
    def f():
        s = mpf(0)
        k = 0
        while True:
            t = mpf(1) / (mpf(16) ** k * (8 * k + j))
            if t < mpf(2) ** (-(p + 80)):
                break
            s += t
            k += 1
        return s
    return rounded(f, p)


def cases():
    c = {}
    c["sqrt2_sqrt3"] = (alg(2, 2, 1, 3, 2, 4, 200), None)
    c["deg12"] = (alg(3, 3, 1, 2, 4, 12, 600), None)
    c["deg16"] = (alg(3, 4, -1, 2, 4, 16, 400), None)
    c["deg35"] = (alg(2, 5, 1, 3, 7, 35, 1000), None)
    p = 200
    c["machin"] = ([rounded(lambda: pi / 4, p), rounded(lambda: atan(mpf(1) / 5), p),
                    rounded(lambda: atan(mpf(1) / 239), p)], None)
    c["machin5"] = ([rounded(lambda: pi / 4, p)] +
                    [rounded(lambda q=q: atan(mpf(1) / q), p) for q in (49, 57, 239, 110443)], None)
    p = 300
    c["bbp"] = ([rounded(lambda: +pi, p)] + [bbp_s(j, p) for j in (1, 4, 5, 6)], None)
    c["logs"] = ([rounded(lambda q=q: log(q), p) for q in (2, 3, 5, 7)] +
                 [rounded(lambda: 31 * log(2) - 17 * log(3) + 11 * log(5) + 4 * log(7), p)], None)
    c["zeta3_pi"] = ([rounded(lambda: mpf(1), p), rounded(lambda: +pi, p),
                      rounded(lambda: zeta(3), p)], 10**12)
    c["zeta3_pi3"] = ([rounded(lambda: zeta(3), p), rounded(lambda: pi ** 3, p)], 10**20)
    p = 700
    c["euler_poly6"] = ([rounded(lambda k=k: euler ** k, p) for k in range(7)], 10**6)
    p = 400
    c["constants"] = ([rounded(f, p) for f in (lambda: mpf(1), lambda: +pi, lambda: +e,
                                                lambda: log(2), lambda: +euler,
                                                lambda: +catalan)], 10**8)
    p = 200
    c["acot"] = ([rounded(lambda: +pi, p)] +
                 [rounded(lambda q=q: atan(mpf(1) / q), p) for q in range(2, 11)], None)
    with mp.workprec(53):
        c["scales"] = ([mpf(1), mpf(2) ** -200, 3 * mpf(2) ** -200], None)
    c["pi_e_1e6"] = ([rounded(lambda: +pi, p), rounded(lambda: +e, p)], 10**6)
    c["pi_e"] = ([rounded(lambda: +pi, p), rounded(lambda: +e, p)], None)
    c["pi_e_1"] = ([rounded(lambda: +pi, p), rounded(lambda: +e, p), rounded(lambda: mpf(1), p)],
                   None)
    return c


def integers(xs, p):
    """X_i = x_i / 2^L and the exponents of their errors, as the crate makes them."""
    ps = []
    for x in xs:
        man, exp = x.man_exp
        man = int(man)
        # mpmath strips trailing zeros; restore the mantissa of p bits.
        shift = p - abs(man).bit_length()
        ps.append((man << shift, int(exp) - shift))
    low = min(q for _, q in ps)
    top = max(q + abs(m).bit_length() for m, q in ps)
    X = [m << (q - low) for m, q in ps]
    err = [(q - low) + ERROR_BITS for _, q in ps]
    return X, err, top - low, ps


def nint(v):
    """The nearest integer of an mpf, halves away from zero."""
    f = mpmath.floor(abs(v) + mpf(0.5))
    return int(f) if v >= 0 else -int(f)


class Ref:
    def __init__(self, X, err, span, max_norm):
        n = len(X)
        self.n, self.X, self.err = n, X, err
        self.frac = span + GUARD_BITS + 2 * n.bit_length()
        self.wp = 2 * self.frac + 128
        self.max_norm = max_norm
        mp.prec = self.wp
        self.y = list(X)
        self.A = [[int(i == j) for j in range(n)] for i in range(n)]
        self.B = [[int(i == j) for j in range(n)] for i in range(n)]
        tail = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            tail[k] = tail[k + 1] + X[k] * X[k]
        s = [sqrt(mpf(t)) for t in tail]
        H = [[mpf(0)] * (n - 1) for _ in range(n)]
        for j in range(n - 1):
            H[j][j] = s[j + 1] / s[j]
            for i in range(j + 1, n):
                H[i][j] = -mpf(X[i]) * X[j] / (s[j] * s[j + 1])
        self.H = H
        self.best = mpf(1)
        self.proved_at = None

    def reduce(self, i, j):
        H, n = self.H, self.n
        if H[j][j] == 0:
            raise StopIteration
        t = nint(H[i][j] / H[j][j])
        if t == 0:
            return
        self.y[j] += t * self.y[i]
        for k in range(j + 1):
            H[i][k] -= t * H[j][k]
        for k in range(n):
            self.A[i][k] -= t * self.A[j][k]
            self.B[k][j] += t * self.B[k][i]

    def tolerance(self, m):
        return sum(abs(v) << e for v, e in zip(m, self.err))

    def relation(self):
        n = self.n
        found = None
        for j in range(n):
            col = [self.B[i][j] for i in range(n)]
            r = sum(a * b for a, b in zip(col, self.X))
            assert r == self.y[j]
            if abs(r) <= self.tolerance(col):
                nn = sum(v * v for v in col)
                if found is None or nn < found[0]:
                    found = (nn, col)
        if found is None:
            return None
        m = found[1]
        first = next(v for v in m if v)
        return [-v for v in m] if first < 0 else m

    def iterate(self):
        n, H = self.n, self.H
        w = n - 1
        best, r = None, 0
        for i in range(w):
            v = abs(H[i][i]) * mpf(GAMMA) ** (i + 1)
            if best is None or v > best:
                best, r = v, i
        self.y[r], self.y[r + 1] = self.y[r + 1], self.y[r]
        H[r], H[r + 1] = H[r + 1], H[r]
        self.A[r], self.A[r + 1] = self.A[r + 1], self.A[r]
        for k in range(n):
            self.B[k][r], self.B[k][r + 1] = self.B[k][r + 1], self.B[k][r]
        if r + 2 < n:
            p, q = H[r][r], H[r][r + 1]
            t0 = sqrt(p * p + q * q)
            if t0 == 0:
                raise StopIteration
            t1, t2 = p / t0, q / t0
            for i in range(r, n):
                t3, t4 = H[i][r], H[i][r + 1]
                H[i][r] = t1 * t3 + t2 * t4
                H[i][r + 1] = -t2 * t3 + t1 * t4
            H[r][r + 1] = mpf(0)
        for i in range(r + 1, n):
            for j in range(min(i - 1, r + 1), -1, -1):
                self.reduce(i, j)

    def proved(self):
        """1/max_j(|r_j| + |c_0| |delta|) from the Gram-Schmidt of (X, a_0, ..., a_{n-2})."""
        n = self.n
        with mp.workprec(3 * self.frac + 200):
            rows = [[mpf(v) for v in self.X]] + [[mpf(v) for v in self.A[j]] for j in range(n - 1)]
            delta = sqrt(sum(mpf(2) ** (2 * e) for e in self.err))
            # Orthonormal q and L by modified Gram-Schmidt, twice.
            qs, L = [], []
            for j, row in enumerate(rows):
                w = list(row)
                lj = [mpf(0)] * (j + 1)
                for _ in range(2):
                    for i, qi in enumerate(qs):
                        c = mpmath.fsum(a * b for a, b in zip(w, qi))
                        w = [a - c * b for a, b in zip(w, qi)]
                        lj[i] += c
                nrm = sqrt(mpmath.fsum(a * a for a in w))
                lj[j] = nrm
                qs.append([a / nrm for a in w])
                L.append(lj)
            worst = mpf(0)
            for j in range(1, n):
                c = [mpf(0)] * j
                for k in range(j - 1, -1, -1):
                    s = L[j][k] - mpmath.fsum(c[i] * L[i][k] for i in range(k + 1, j))
                    c[k] = s / L[k][k]
                r = [rows[j][col] - mpmath.fsum(c[i] * rows[i][col] for i in range(j))
                     for col in range(n)]
                u = sqrt(mpmath.fsum(v * v for v in r)) + abs(c[0]) * delta
                worst = max(worst, u)
            return 1 / worst

    def run(self):
        n = self.n
        try:
            for i in range(1, n):
                for j in range(i - 1, -1, -1):
                    self.reduce(i, j)
            it = 0
            while True:
                m = self.relation()
                if m is not None:
                    return ("relation", m, it)
                if self.max_norm is not None:
                    largest = max(abs(self.H[j][j]) for j in range(n - 1))
                    floating = -mpmath.log(largest, 2)
                    if self.best >= self.max_norm:
                        return ("nonebelow", self.best, it)
                    if floating >= mpmath.log(self.max_norm, 2) and (
                            self.proved_at is None or floating > self.proved_at + 1):
                        self.proved_at = floating
                        b = self.proved()
                        self.best = max(self.best, b)
                        if self.best >= self.max_norm:
                            return ("nonebelow", self.best, it)
                it += 1
                if it > 10**6:
                    return ("iterations", None, it)
                self.iterate()
        except StopIteration:
            return ("exhausted", None, 0)


# The cases whose relations are unique up to sign, which every method must find alike; the
# others have many relations (to the precision, for pi_e and pi_e_1), and only the reference,
# which follows the same path, is compared.
UNIQUE = ("sqrt2_sqrt3", "deg12", "deg16", "deg35", "machin", "machin5", "bbp", "logs")


def mpmath_answers(name, xs, p):
    mp.prec = p
    xs = [+x for x in xs]
    r = mpmath.pslq(xs, tol=mpf(2) ** (-p + 20), maxcoeff=10**30, maxsteps=10**6)
    fp = None
    # findpoly tries every degree up to the one asked, a PSLQ each: degree 35 at 1000 bits would
    # take most of an hour, and its pslq above is the same check.
    if name in ("sqrt2_sqrt3", "deg12", "deg16"):
        # findpoly tries the degrees from 1 up: a bound on the coefficients keeps the relations
        # to the precision of the low degrees out.
        f = findpoly(xs[1], len(xs) - 1, tol=mpf(2) ** (-p + 20), maxcoeff=10**7,
                     maxsteps=10**6)
        fp = None if f is None else f[::-1]
    return r, fp


def pari_lindep(xs, p):
    digits = int(p * 0.30103)
    vals = ",".join(mpmath.nstr(x, digits + 5, strip_zeros=False) for x in xs)
    prog = f"default(realprecision,{digits}); print(lindep([{vals}]))"
    out = subprocess.run(["gp", "-q"], input=prog, capture_output=True, text=True,
                         timeout=600).stdout.strip()
    return [int(v) for v in out.strip("[]~").split(",")]


def norm_sign(m):
    if m is None:
        return None
    m = [int(v) for v in m]
    first = next((v for v in m if v), 0)
    return [-v for v in m] if first < 0 else m


def main():
    out = open(sys.argv[1] if len(sys.argv) > 1 else "pslq_expected.txt", "w")
    precs = {"sqrt2_sqrt3": 200, "deg12": 600, "deg16": 400, "deg35": 1000, "machin": 200,
             "machin5": 200, "bbp": 300, "logs": 300, "zeta3_pi": 300, "zeta3_pi3": 300,
             "euler_poly6": 700, "constants": 400, "acot": 200, "scales": 53, "pi_e_1e6": 200,
             "pi_e": 200, "pi_e_1": 200}
    for name, (xs, max_norm) in cases().items():
        p = precs[name]
        X, err, span, ps = integers(xs, p)
        for i, (m, q) in enumerate(ps):
            out.write(f"input {name} {i} {abs(m) & (2**64 - 1):x}\n")
        kind, val, it = Ref(X, err, span, max_norm).run()
        if kind == "relation":
            out.write(f"ref {name} relation {' '.join(map(str, val))}\n")
        elif kind == "nonebelow":
            out.write(f"ref {name} nonebelow {mpmath.nstr(val, 30)}\n")
        else:
            out.write(f"ref {name} {kind}\n")
        print(name, kind, it, file=sys.stderr)
        if name in UNIQUE:
            r, fp = mpmath_answers(name, xs, p)
            r = norm_sign(r)
            out.write(f"mpmath {name} {' '.join(map(str, r)) if r else 'none'}\n")
            if name in ("sqrt2_sqrt3", "deg12", "deg16"):
                fp = norm_sign(fp)
                out.write(f"findpoly {name} {' '.join(map(str, fp)) if fp else 'none'}\n")
            mp.prec = p
            out.write(f"pari {name} {' '.join(map(str, norm_sign(pari_lindep(xs, p))))}\n")
        out.flush()


if __name__ == "__main__":
    main()
