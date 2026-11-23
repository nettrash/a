#!/usr/bin/env python3
"""Regenerates enum_svp.txt, enum_cvp.txt and enum_short.txt, the references of
lib/tests/math/lattice/enumerate.rs: `python3 gen_enum.py` in this directory.

The bases are made from seeds exactly as the test makes them (SplitMix64, the `Rng` of
common.rs, and `make` of enumerate.rs, draw for draw). The references:

* shortest vectors: the norm of `fplll -a svp` (fplll 5.5), the same as the minimum of PARI's
  `qfminim` (flag 2, on the Gram matrix of a basis by `mathnf`), and, up to dimension 16, as an
  exact rational Fincke-Pohst enumeration written here; dependent rows are first reduced by
  `fplll -a lll`, whose zero rows are dropped;
* closest vectors: the squared distance to the target by the exact enumeration here (the target
  first brought near the origin by Babai's nearest plane method, exactly), compared with the
  vector of `fplll -a cvp` (whose CVP is not proved: the file's header counts where it agrees);
* short vectors: PARI's `qfminim(G, B, , 2)`: the number of pairs, the number of each norm, and
  the sign-free checksum sum over the pairs of (sum_j (j + 1)*v_j)^2 of the vectors v (columns of
  `mathnf` times the coordinates PARI gives).

Every reference is computed twice where two tools can compute it, and the script stops on a
disagreement, but for fplll's CVP, which is not proved: the header of enum_cvp.txt counts the
cases where it finds a farther vector than the exact enumeration (never a closer one).
"""
import math
import subprocess
from fractions import Fraction

MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def number(self, bits):
        x = 0
        left = bits
        while left > 0:
            take = min(left, 64)
            w = self.next() if take == 64 else self.next() >> (64 - take)
            x = (x << take) + w
            left -= take
        return -x if self.next() & 1 == 1 else x


def from_fn(n, m, f):
    """Matrix::from_fn: column after column."""
    a = [[None] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            a[i][j] = f(i, j)
    return a


def random_basis(rng, n, m, bits):
    return from_fn(n, m, lambda i, j: rng.number(bits))


def knapsack(rng, n, bits):
    return from_fn(n, n + 1, lambda i, j: abs(rng.number(bits)) if j == 0 else int(i + 1 == j))


def make(kind, n, m, bits, seed):
    rng = Rng(seed)
    if kind == "random":
        return random_basis(rng, n, m, bits)
    if kind == "knapsack":
        return knapsack(rng, n, bits)
    if kind == "hilbert":
        return [[(1 << bits) // (i + j + 1) for j in range(n)] for i in range(n)]
    if kind == "gram":
        a = random_basis(rng, n, n, bits)
        return [[sum(x * y for x, y in zip(a[i], a[j])) for j in range(n)] for i in range(n)]
    if kind == "qary":
        k = n // 2
        q = abs(rng.number(bits)) + 2
        b = [[0] * n for _ in range(n)]
        for i in range(n):
            b[i][i] = 1 if i < k else q
        for i in range(k):
            for j in range(k, n):
                b[i][j] = abs(rng.number(bits)) % q
        return b
    if kind == "combo":
        base = random_basis(rng, n - 1, m, bits)
        last = [sum((i + 1) * base[i][j] for i in range(n - 1)) for j in range(m)]
        return base + [last]
    if kind == "ill":
        b = [[0] * n for _ in range(n)]
        b[0][0] = 2
        for i in range(1, n):
            b[i][i] = 1 << (bits * i)
        for i in range(1, n):
            b[i][0] = abs(rng.number(2)) + 1
        return b
    raise ValueError(kind)


def target(kind, b, m, tbits, seed):
    rng = Rng(seed + 1000003)
    if kind == "far":
        return [rng.number(tbits) for _ in range(m)]
    if kind == "near":
        c = [rng.number(tbits) for _ in range(len(b))]
        e = [rng.number(3) for _ in range(m)]
        return [sum(c[i] * b[i][j] for i in range(len(b))) + e[j] for j in range(m)]
    if kind == "half":
        return [rng.number(3) if j == 0 else 1 << (tbits * j - 1) for j in range(m)]
    raise ValueError(kind)


def fmt(b):
    return "[" + "".join("[" + " ".join(str(x) for x in r) + "]" for r in b) + "]"


def fplll(args, text):
    out = subprocess.run(["fplll"] + args, input=text, capture_output=True, text=True, timeout=3600)
    if out.returncode != 0 or "fplll:" in out.stdout + out.stderr:
        raise RuntimeError(out.stdout + out.stderr)
    return out.stdout


def parse_matrix(s):
    rows = []
    for line in s.replace("]", "]\n").split("\n"):
        line = line.strip().strip("[").strip("]").strip()
        if line:
            rows.append([int(x) for x in line.split()])
    return rows


def gp(script):
    out = subprocess.run(["gp", "-q", "-D", "parisizemax=4000000000"], input=script, capture_output=True,
                         text=True, timeout=3600)
    if out.returncode != 0 or "***" in out.stdout + out.stderr:
        raise RuntimeError(out.stdout + out.stderr)
    return out.stdout


def gp_matrix(a):
    return "[" + ";".join(",".join(str(x) for x in r) for r in a) + "]"


def independent(b):
    """The nonzero rows of fplll's LLL reduction: a basis of the lattice of the rows."""
    r = parse_matrix(fplll(["-a", "lll"], fmt(b)))
    return [row for row in r if any(row)]


def norm2(v):
    return sum(x * x for x in v)


# An exact rational Fincke-Pohst enumeration, plain recursion, for small dimensions: a check of
# the tools that shares nothing with them or with the crate.
def gso(b):
    n = len(b)
    bs = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    r = []
    for i in range(n):
        v = [Fraction(x) for x in b[i]]
        for j in range(i):
            mu[i][j] = sum(Fraction(x) * y for x, y in zip(b[i], bs[j])) / r[j]
            v = [x - mu[i][j] * y for x, y in zip(v, bs[j])]
        bs.append(v)
        r.append(sum(x * x for x in v))
    return mu, r, bs


def fp_closest(b, t):
    """min |v - t|^2 over the lattice (t = None: min |v|^2 over v != 0), exactly."""
    n = len(b)
    mu, r, bs = gso(b)
    if t is None:
        tc = [Fraction(0)] * n
        perp = Fraction(0)
        best = Fraction(norm2(b[0]))
    else:
        tc = [sum(Fraction(x) * y for x, y in zip(t, bs[j])) / r[j] for j in range(n)]
        perp = norm2(t) - sum(tc[j] ** 2 * r[j] for j in range(n))
        # Babai's nearest plane point: the first bound.
        x = [0] * n
        part = Fraction(0)
        for l in range(n - 1, -1, -1):
            c = tc[l] - sum(x[j] * mu[j][l] for j in range(l + 1, n))
            x[l] = math.floor(c + Fraction(1, 2))
            part += (x[l] - c) ** 2 * r[l]
        best = part + perp
    best = [best]
    x = [0] * n

    def rec(l, part):
        if l < 0:
            if t is None and not any(x):
                return
            best[0] = min(best[0], part + perp)
            return
        c = tc[l] - sum(x[j] * mu[j][l] for j in range(l + 1, n))
        w = (best[0] - perp - part) / r[l]
        if w < 0:
            return
        s = math.isqrt(math.ceil(w)) + 1
        for v in range(math.floor(c) - s, math.ceil(c) + s + 1):
            d = (v - c) ** 2 * r[l]
            if part + d + perp <= best[0]:
                x[l] = v
                rec(l - 1, part + d)
        x[l] = 0

    rec(n - 1, Fraction(0))
    return best[0]


def svp_norm(b):
    basis = independent(b)
    v = parse_matrix(fplll(["-a", "svp"], fmt(basis)))[0]
    f = norm2(v)
    p = int(gp("B=Mat(%s);M=mathnf(B~);G=M~*M;x=qfminim(G,,,2)[3][,1];print(x~*G*x)" % gp_matrix(b)).strip())
    if f != p:
        raise RuntimeError("fplll %d, PARI %d" % (f, p))
    if len(basis) <= 16:
        e = fp_closest(basis, None)
        if e != f:
            raise RuntimeError("fplll %d, exact %s" % (f, e))
    return f


def babai(b, t):
    """t minus the lattice vector of Babai's nearest plane method, exactly: the same closest
    vector problem, with the target near the origin (fplll's CVP can take very long on a far
    target)."""
    mu, r, bs = gso(b)
    t = list(t)
    for l in range(len(b) - 1, -1, -1):
        c = sum(Fraction(x) * y for x, y in zip(t, bs[l])) / r[l]
        q = math.floor(c + Fraction(1, 2))
        t = [x - q * y for x, y in zip(t, b[l])]
    return t


def fplll_closest(basis, t):
    """fplll's closest vector to t (already near the origin), "error" if fplll fails, or
    "timeout" if it takes more than 15 seconds. A lattice of a lower rank than its dimension gets the projection of the
    target on its span, both scaled to integers (fplll's CVP takes very long on the component
    off the span)."""
    if len(basis) < len(t):
        mu, r, bs = gso(basis)
        ts = [Fraction(0)] * len(t)
        for j in range(len(basis)):
            c = sum(Fraction(x) * y for x, y in zip(t, bs[j])) / r[j]
            ts = [x + c * y for x, y in zip(ts, bs[j])]
        d = 1
        for x in ts:
            d = d * x.denominator // math.gcd(d, x.denominator)
        basis = [[d * x for x in row] for row in basis]
        goal = [int(x * d) for x in ts]
    else:
        d = 1
        goal = t
    try:
        out = subprocess.run(["fplll", "-a", "cvp"], input=fmt(basis) + "\n[" + " ".join(map(str, goal)) + "]\n",
                             capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return "timeout"
    if out.returncode != 0 or not out.stdout.strip() or "fplll:" in out.stdout + out.stderr:
        return "error"
    vd = parse_matrix(out.stdout)[0]
    if any(x % d for x in vd):
        return "error"
    return [x // d for x in vd]


FPLLL_CVP = {"agree": 0, "error": [], "timeout": [], "worse": []}


def cvp_dist(b, t, case):
    """The exact enumeration's, compared with fplll's (whose CVP is not proved: it can return a
    farther vector, never a closer one)."""
    basis = independent(b)
    t = babai(basis, t)
    e = fp_closest(basis, t)
    v = fplll_closest(basis, t) if len(basis) > 1 else "error"
    if isinstance(v, str):
        FPLLL_CVP[v].append(case)
    else:
        f = norm2([x - y for x, y in zip(v, t)])
        if f < e:
            raise RuntimeError("fplll %d below the exact %s" % (f, e))
        if f == e:
            FPLLL_CVP["agree"] += 1
        else:
            FPLLL_CVP["worse"].append(case)
    return e


def short(b, bound):
    out = gp(
        "B=Mat(%s);M=mathnf(B~);G=M~*M;r=qfminim(G,%d,,2);V=M*r[3];"
        "print(#V);for(i=1,#V,print(norml2(V[,i])\" \"sum(j=1,#V[,i],j*V[j,i])^2))"
        % (gp_matrix(b), bound))
    lines = out.split()
    count = int(lines[0])
    norms = {}
    check = 0
    for i in range(count):
        nm, h = int(lines[1 + 2 * i]), int(lines[2 + 2 * i])
        norms[nm] = norms.get(nm, 0) + 1
        check += h
    return count, ",".join("%d:%d" % (k, norms[k]) for k in sorted(norms)), check


SVP = (
    [("random", n, n, bits, seed) for n in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) for bits in (10, 30)
     for seed in (1, 2)]
    + [("knapsack", n, n + 1, 10 * n, seed) for n in (2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 24) for seed in (1, 2)]
    + [("knapsack", 8, 9, 1000, 3), ("knapsack", 12, 13, 2000, 3), ("random", 4, 4, 500, 3),
       ("random", 6, 6, 1000, 3), ("random", 3, 3, 3000, 3)]
    + [("hilbert", n, n, bits, 0) for n in (2, 4, 6, 8, 10, 12) for bits in (30, 100)]
    + [("gram", n, n, 8, seed) for n in (3, 5, 8, 10, 12) for seed in (1,)]
    + [("qary", n, n, bits, 1) for n, bits in ((4, 10), (8, 12), (12, 14), (16, 16), (20, 16), (24, 18))]
    + [("random", n, m, 20, 4) for n, m in ((1, 3), (3, 5), (5, 8), (10, 15), (15, 20))]
    + [("random", n, m, 10, 5) for n, m in ((3, 1), (5, 3), (8, 5), (12, 8), (20, 10))]
    + [("combo", n, m, 12, 6) for n, m in ((3, 3), (5, 6), (8, 8), (12, 15))]
)

# The heavier sweep of the ignored test: knapsack lattices up to dimension 30 and q-ary ones.
SVP_HEAVY = (
    [("knapsack", n, n + 1, 10 * n, seed) for n in (25, 26, 27, 28, 29, 30) for seed in (1, 2, 3)]
    + [("qary", n, n, 20, seed) for n in (26, 28, 30) for seed in (1, 2)]
    + [("random", 30, 30, 60, seed) for seed in (1, 2)]
)

CVP = (
    [("random", n, n, 10, seed, tk, 30 if tk == "far" else 10) for n in (1, 2, 3, 4, 6, 8, 10, 12, 15, 20)
     for seed in (1, 2) for tk in ("far", "near")]
    + [("knapsack", n, n + 1, 10 * n, 1, "far", 10 * n + 10) for n in (4, 8, 12, 16, 20)]
    + [("knapsack", n, n + 1, 10 * n, 1, "near", 10) for n in (4, 8, 12, 16, 20)]
    + [("random", n, m, 15, 2, "far", 40) for n, m in ((2, 4), (3, 5), (6, 9), (10, 14))]
    + [("knapsack", 6, 7, 500, 3, "far", 520), ("random", 3, 3, 800, 3, "far", 900),
       ("random", 5, 5, 200, 3, "near", 50)]
    + [("hilbert", n, n, 40, 0, "far", 50) for n in (3, 5, 7, 9)]
    + [("qary", n, n, 12, 1, "far", 14) for n in (6, 10, 14)]
    + [("random", n, m, 10, 5, "far", 20) for n, m in ((4, 2), (6, 4), (9, 6))]
    # Gram-Schmidt norms 4, 2^(2e), 2^(4e), ... and the target half-way at every upper level:
    # the crate's search has to run in integers.
    + [("ill", n, n, e, seed, "half", e) for n in (2, 3, 4, 5, 6) for e in (40, 100, 300, 500) for seed in (1,)]
    + [("ill", n, n, 100, 2, "half", 100) for n in (3, 4, 5, 6)]
)

SHORT = (
    [("random", n, n, 4, seed, f) for n in (1, 2, 3, 4, 5, 6, 8) for seed in (1, 2) for f in (1, 2, 4)]
    + [("knapsack", n, n + 1, 3 * n, 1, f) for n in (4, 6, 8, 10) for f in (1, 2, 3)]
    + [("gram", n, n, 3, 1, f) for n in (3, 4, 6) for f in (1, 2)]
    + [("qary", n, n, 6, 1, f) for n in (6, 8, 10) for f in (1, 2)]
    + [("random", n, m, 4, 4, f) for n, m in ((2, 4), (3, 6), (6, 4)) for f in (1, 3)]
    + [("combo", 5, 5, 4, 6, 2), ("hilbert", 5, 5, 12, 0, 3), ("random", 3, 3, 100, 3, 5)]
)


def main():
    with open("enum_svp.txt", "w") as f:
        f.write("# kind n m bits seed norm2: the squared norm of a shortest vector (fplll svp = PARI qfminim)\n")
        for case in SVP:
            f.write("%s %d %d %d %d %d\n" % (case + (svp_norm(make(*case)),)))
        f.write("# heavy\n")
        for case in SVP_HEAVY:
            f.write("%s %d %d %d %d %d\n" % (case + (svp_norm(make(*case)),)))
    lines = []
    for case in CVP:
        kind, n, m, bits, seed, tk, tbits = case
        b = make(kind, n, m, bits, seed)
        t = target(tk, b, m, tbits, seed)
        lines.append("%s %d %d %d %d %s %d %d\n" % (case + (cvp_dist(b, t, case),)))
    with open("enum_cvp.txt", "w") as f:
        f.write("# kind n m bits seed target tbits dist2: the squared distance of a closest vector, by the\n")
        f.write("# exact enumeration of gen_enum.py; fplll cvp agrees on %d, finds a farther vector on %d (%s)\n"
                % (FPLLL_CVP["agree"], len(FPLLL_CVP["worse"]), "; ".join(" ".join(map(str, c)) for c in FPLLL_CVP["worse"])))
        f.write("# fails on %d (%s) and takes more than 15 s on %d (%s)\n"
                % (len(FPLLL_CVP["error"]), "; ".join(" ".join(map(str, c)) for c in FPLLL_CVP["error"]),
                   len(FPLLL_CVP["timeout"]), "; ".join(" ".join(map(str, c)) for c in FPLLL_CVP["timeout"])))
        f.writelines(lines)
    with open("enum_short.txt", "w") as f:
        f.write("# kind n m bits seed bound pairs norms checksum (PARI qfminim, bound = f * minimum)\n")
        for kind, n, m, bits, seed, fct in SHORT:
            b = make(kind, n, m, bits, seed)
            bound = svp_norm(b) * fct
            count, norms, check = short(b, bound)
            f.write("%s %d %d %d %d %d %d %s %d\n" % (kind, n, m, bits, seed, bound, count, norms, check))


if __name__ == "__main__":
    main()
