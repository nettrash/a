"""A third reference for gf2_reference.txt: sympy's DomainMatrix over GF(2) recomputes, for every
line of at most 194 rows and columns, the reduced row echelon form, the null space (the rows of
`nullspace()` as columns), the inverse and the solution from the reduced form of [A | B], and
reports disagreements.

    python3 gen_gf2_sympy.py < gf2_reference.txt

The matrices are those of gen_gf2.h, rebuilt here from the same seeds."""

import sys

from sympy import GF
from sympy.polys.matrices import DomainMatrix

M64 = (1 << 64) - 1
F = GF(2)


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class Seq:
    def __init__(self, s):
        self.s = s

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        return mix(self.s)


def naive_mul(a, b, n):
    return [[sum(a[i][k] & b[k][j] for k in range(len(b))) & 1 for j in range(n)] for i in range(len(a))]


def gen(kind, m, n, seed):
    """The rows as lists of 0s and 1s."""
    if kind == "low":
        r = min(m, n) // 3
        return naive_mul(gen("dense", m, r, seed), gen("dense", r, n, seed ^ 0xA5A5A5A5A5A5A5A5), n)
    s = Seq(seed)
    stride = (n + 63) // 64
    words = []
    for _ in range(m):
        row = []
        for w in range(stride):
            v = s.next()
            if kind in ("sparse", "permuted"):
                v &= s.next()
                v &= s.next()
            row.append(v)
        words.append(row)
    a = [[(words[i][j // 64] >> (j % 64)) & 1 for j in range(n)] for i in range(m)]
    if kind == "permuted":
        for i in range(m):
            for j in range(min(n, i + 1)):
                a[i][j] = int(j == i)
        for i in range(m - 1, 0, -1):
            j = s.next() % (i + 1)
            a[i], a[j] = a[j], a[i]
    if kind == "deficient":
        for i in range(3, m):
            if i % 4 == 3:
                a[i] = [x ^ y for x, y in zip(a[i - 1], a[i - 3])]
        for j in range(2, n):
            if j % 3 == 2:
                for i in range(m):
                    a[i][j] = a[i][j - 1] ^ a[i][j - 2]
    return a


def hash_rows(a, m, n):
    h = mix(mix(m) ^ n)
    for i in range(m):
        for w in range((n + 63) // 64):
            word = 0
            for j in range(64 * w, min(n, 64 * w + 64)):
                word |= a[i][j] << (j - 64 * w)
            h = mix(h ^ word)
    return h


def dm(a, m, n):
    return DomainMatrix([[F(x) for x in row] for row in a], (m, n), F)


def rows(d):
    return [[int(x) % 2 for x in row] for row in d.to_Matrix().tolist()]


def rref(a, m, n):
    if m == 0 or n == 0:
        return [[0] * n for _ in range(m)], ()
    r, pivots = dm(a, m, n).rref()
    return rows(r), tuple(pivots)


def main():
    checks = bad = 0
    for line in sys.stdin:
        f = line.split()
        op = f[0]

        def check(ok, what):
            nonlocal checks, bad
            checks += 1
            if not ok:
                bad += 1
                print("disagree (%s): %s" % (what, line.strip()), file=sys.stderr)

        if op == "ech":
            m, n = int(f[2]), int(f[3])
            if max(m, n) > 194:
                continue
            a = gen(f[1], m, n, int(f[4]))
            r, pivots = rref(a, m, n)
            check(len(pivots) == int(f[5]), "rank")
            check(hash_rows(r, m, n) == int(f[6], 16), "rref")
            if m > 0 and n > 0 and len(pivots) < n:
                k = rows(dm(a, m, n).nullspace())
                x = [[k[c][i] for c in range(len(k))] for i in range(n)]
            else:
                x = [[1 if i == c else 0 for c in range(n - len(pivots))] for i in range(n)]
            check(hash_rows(x, n, n - len(pivots)) == int(f[7], 16), "null space")
        elif op == "inv":
            n = int(f[2])
            if n > 194 or n == 0:
                continue
            a = gen(f[1], n, n, int(f[3]))
            d = dm(a, n, n)
            if int(d.det()) % 2 == 0:
                check(f[4] == "singular", "determinant")
            else:
                check(f[4] != "singular", "determinant")
                check(hash_rows(rows(d.inv()), n, n) == int(f[4], 16), "inverse")
        elif op == "solve":
            m, n, k = int(f[2]), int(f[3]), int(f[6])
            if max(m, n) > 194:
                continue
            a = gen(f[1], m, n, int(f[4]))
            b = naive_mul(a, gen("dense", n, k, int(f[7])), k) if f[5] == "image" else gen(f[5], m, k, int(f[7]))
            ab = [a[i] + b[i] for i in range(m)]
            r, pivots = rref(ab, m, n + k)
            ra = sum(1 for p in pivots if p < n)
            check(ra == int(f[8]) and len(pivots) == int(f[9]), "ranks")
            if len(pivots) == ra:
                x = [[0] * k for _ in range(n)]
                for i, p in enumerate(pivots):
                    x[p] = r[i][n:]
                check(hash_rows(x, n, k) == int(f[10], 16), "solution")
            else:
                check(f[10] == "none", "no solution")
    print("%d checks, %d disagreements" % (checks, bad))
    return bad != 0


if __name__ == "__main__":
    sys.exit(main())
