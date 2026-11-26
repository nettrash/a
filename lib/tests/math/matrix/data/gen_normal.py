#!/usr/bin/env python3
"""The expected Hermite and Smith normal forms for tests/math/matrix/normal.rs.

The matrices come from a seeded generator that the test has too: xorshift64 as `Rng` in
common.rs, the elements row after row (see `generate` here and in the test). FLINT 3.6 computes
the forms (gen_normal.c: `fmpz_mat_hnf` of A and of [A | I], and `fmpz_mat_snf`), PARI/GP 2.19
computes them again (gen_normal.gp: `mathnf` and `matsnf` in FLINT's conventions), the two must
agree, and the Hermite form of [A | I] must begin with that of A. Writes normal_expected.txt
beside this script: for each matrix its recipe, the Hermite form H, the transformation U of the
form of [A | I] (as FNV-1a digests of their text, and the text itself when it is short), and
the elementary divisors in full.

Usage: python3 gen_normal.py    (needs cc, FLINT, GMP and MPFR in /opt/homebrew, and gp)

       python3 gen_normal.py sweep <count> <seed> <file> [big]
           writes the same for <count> recipes drawn at random from <seed> to <file>, for the
           ignored test `sweep_against_flint_and_pari` (A_NORMAL_SWEEP=<file>): of up to 12 rows
           and columns, or with `big` of up to 32 and elements of up to 1000 bits.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MASK = (1 << 64) - 1


class Rng:
    """xorshift64 with the shifts 13, 7, 17, as the tests' `Rng`."""

    def __init__(self, seed):
        self.x = seed

    def next(self):
        x = self.x
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        self.x = x
        return x


def number(r, bits):
    """A random integer below 2^bits in magnitude, of either sign."""
    words = (bits + 63) // 64
    x = 0
    for _ in range(words):
        x = (x << 64) | r.next()
    x >>= words * 64 - bits
    return -x if r.next() & 1 else x


def random_rows(r, m, n, bits):
    return [[number(r, bits) for _ in range(n)] for _ in range(m)]


def generate(kind, m, n, bits, k, seed):
    """The matrix of a recipe, as a list of rows.

    random: elements of `bits` bits; rank: the product of random m × k and k × n factors of `bits`
    bits; small: elements in [-bits, bits]; sparse: an element of `bits` bits one time in four, 0
    otherwise; smith: diag(2^a·3^b·5^c, zero one time in five) taken through 3·(m + n) random
    row and column operations, each adding -3 to 3 times one row (column) to another.
    """
    r = Rng(seed)
    if kind == "random":
        return random_rows(r, m, n, bits)
    if kind == "rank":
        b = random_rows(r, m, k, bits)
        c = random_rows(r, k, n, bits)
        return [[sum(b[i][t] * c[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
    if kind == "small":
        return [[r.next() % (2 * bits + 1) - bits for _ in range(n)] for _ in range(m)]
    if kind == "sparse":
        return [[number(r, bits) if r.next() % 4 == 0 else 0 for _ in range(n)] for _ in range(m)]
    if kind == "smith":
        a = [[0] * n for _ in range(m)]
        for i in range(min(m, n)):
            if r.next() % 5 == 0:
                continue
            e2, e3, e5 = r.next() % 4, r.next() % 3, r.next() % 2
            a[i][i] = 2**e2 * 3**e3 * 5**e5
        for _ in range(3 * (m + n)):
            rows = r.next() & 1 == 0
            size = m if rows else n
            i, j = r.next() % size, r.next() % size
            c = r.next() % 7 - 3
            if i == j:
                continue
            if rows:
                a[i] = [x + c * y for x, y in zip(a[i], a[j])]
            else:
                for row in a:
                    row[i] += c * row[j]
        return a
    raise ValueError(kind)


# The recipes: (kind, m, n, bits, k), each with the seed 0x4E4F_524D_0000 + its index.
CASES = [
    ("random", 1, 1, 64, 0),
    ("random", 1, 5, 40, 0),
    ("random", 5, 1, 40, 0),
    ("random", 2, 2, 8, 0),
    ("random", 3, 3, 4, 0),
    ("random", 3, 5, 64, 0),
    ("random", 5, 3, 64, 0),
    ("random", 4, 4, 200, 0),
    ("random", 6, 6, 2, 0),
    ("random", 10, 10, 100, 0),
    ("random", 12, 7, 50, 0),
    ("random", 7, 12, 50, 0),
    ("random", 16, 16, 300, 0),
    ("random", 30, 30, 100, 0),
    ("random", 8, 8, 2000, 0),
    ("random", 3, 8, 3000, 0),
    ("random", 8, 3, 3000, 0),
    ("rank", 4, 4, 10, 2),
    ("rank", 6, 9, 20, 3),
    ("rank", 9, 6, 20, 3),
    ("rank", 15, 15, 30, 10),
    ("rank", 30, 30, 60, 20),
    ("rank", 8, 8, 1000, 5),
    ("small", 4, 6, 1, 0),
    ("small", 6, 4, 1, 0),
    ("small", 8, 8, 2, 0),
    ("small", 12, 12, 2, 0),
    ("small", 20, 25, 1, 0),
    ("sparse", 6, 6, 8, 0),
    ("sparse", 10, 14, 30, 0),
    ("sparse", 20, 20, 20, 0),
    ("smith", 3, 3, 0, 0),
    ("smith", 5, 5, 0, 0),
    ("smith", 6, 8, 0, 0),
    ("smith", 8, 6, 0, 0),
    ("smith", 10, 10, 0, 0),
    ("smith", 16, 16, 0, 0),
    ("smith", 25, 30, 0, 0),
    # Tall and wide, for the transformations whose work is linear in the long side.
    ("random", 160, 3, 11, 0),
    ("rank", 120, 6, 6, 2),
    ("small", 100, 4, 2, 0),
    ("sparse", 90, 5, 30, 0),
    ("smith", 60, 6, 0, 0),
    ("random", 3, 160, 11, 0),
    ("rank", 6, 120, 6, 2),
    ("smith", 6, 60, 0, 0),
]


def text(rows):
    return ";".join(",".join(str(x) for x in row) for row in rows)


def fnv(s):
    """FNV-1a, 64 bits, of the bytes of `s`."""
    h = 0xCBF29CE484222325
    for b in s.encode():
        h = ((h ^ b) * 0x100000001B3) & MASK
    return h


def lines_by_tag(out, tags):
    """The output of a program as one dictionary of tagged lines for each matrix."""
    res, cur = [], {}
    for line in out.splitlines():
        tag, _, rest = line.partition(" ")
        if tag not in tags:
            continue
        cur[tag] = rest
        if tag == tags[-1]:
            res.append(cur)
            cur = {}
    return res


def sweep_cases(count, seed, big):
    """`count` recipes drawn from `seed`: every kind, 1 to 12 rows and columns, mostly small, or
    with `big` 1 to 32 and elements of up to 1000 bits."""
    import random

    g = random.Random(seed)
    out = []
    for _ in range(count):
        kind = g.choice(["random", "rank", "small", "sparse", "smith"])
        top = 32 if big else 12
        m, n = g.randint(1, top), g.randint(1, top)
        if big:
            bits = {"random": g.choice([2, 64, 300, 1000]), "rank": g.choice([2, 50, 400]),
                    "small": g.choice([1, 2]), "sparse": g.choice([10, 500]), "smith": 0}[kind]
        else:
            bits = {"random": g.choice([1, 2, 3, 8, 64, 65, 200]), "rank": g.choice([1, 2, 5, 30]),
                    "small": g.choice([1, 2, 3]), "sparse": g.choice([2, 10, 100]), "smith": 0}[kind]
        k = g.randint(0, min(m, n)) if kind == "rank" else 0
        out.append((kind, m, n, bits, k, g.getrandbits(63) | 1))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        cases = sweep_cases(int(sys.argv[2]), int(sys.argv[3]), sys.argv[5:] == ["big"])
        path = sys.argv[4]
    else:
        cases = [(kind, m, n, bits, k, 0x4E4F_524D_0000 + i) for i, (kind, m, n, bits, k) in enumerate(CASES)]
        path = os.path.join(HERE, "normal_expected.txt")
    mats = [generate(*c) for c in cases]
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "gen_normal")
        subprocess.run(
            ["cc", "-O2", "-I/opt/homebrew/include", os.path.join(HERE, "gen_normal.c"), "-L/opt/homebrew/lib",
             "-lflint", "-lgmp", "-lmpfr", "-o", exe],
            check=True,
        )
        data = [str(len(mats))]
        for a in mats:
            data.append(f"{len(a)} {len(a[0])}")
            data.extend(" ".join(str(x) for x in row) for row in a)
        flint = subprocess.run([exe], input="\n".join(data) + "\n", capture_output=True, text=True, check=True,
                               timeout=3600).stdout
        flint = lines_by_tag(flint, ["H", "L", "U", "S"])
        script = os.path.join(tmp, "run.gp")
        with open(script, "w") as f:
            f.write(f'read("{os.path.join(HERE, "gen_normal.gp")}");\n')
            for a in mats:
                f.write("normal_print(Mat([" + ";".join(",".join(str(x) for x in row) for row in a) + "]));\n")
            f.write("quit;\n")
        pari = subprocess.run(["gp", "-q", "-s", "1G", script], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, check=True, timeout=3600).stdout
        pari = lines_by_tag(pari, ["H", "S"])
    assert len(flint) == len(mats) == len(pari), (len(flint), len(mats), len(pari))
    out = ["# The Hermite and Smith normal forms of the matrices of gen_normal.py: written by it, from",
           "# FLINT 3.6 and PARI/GP 2.19, which agree on every one. For each matrix: its recipe",
           "# (kind m n bits k seed); H and U = the digest (FNV-1a, 64 bits) of the text of the Hermite",
           "# form of A and of the transformation of the form of [A | I], then the text or '-'; S = the",
           "# elementary divisors."]
    for i, ((kind, m, n, bits, k, seed), a, f, p) in enumerate(zip(cases, mats, flint, pari)):
        if f["H"] != p["H"]:
            sys.exit(f"case {i}: FLINT and PARI differ on H")
        if f["S"] != p["S"]:
            sys.exit(f"case {i}: FLINT and PARI differ on S: {f['S'][:80]} / {p['S'][:80]}")
        if f["L"] != f["H"]:
            sys.exit(f"case {i}: the form of [A | I] does not begin with H")
        out.append(f"case {kind} {m} {n} {bits} {k} {seed:#x}")
        for tag in ("H", "U"):
            t = f[tag]
            out.append(f"{tag} {fnv(t):016x} {t if len(t) <= 300 else '-'}")
        out.append(f"S {f['S']}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{len(mats)} matrices; FLINT and PARI agree")


if __name__ == "__main__":
    main()
