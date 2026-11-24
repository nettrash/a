#!/usr/bin/env python3
"""Writes ext_reference.txt, the reference values of the tests of a::math::field's extension
fields (lib/tests/math/field/ext.rs), with PARI/GP and FLINT.

The inputs are not stored: the tests make them again with the same seeded generator (SplitMix64,
as `Rng` in ext.rs), so the file holds the fields, the seeds and the expected outputs. For each
field, PARI/GP (`ffgen` elements: products, `1/a`, `a^e`, `a^(p^i)`, `norm`, `trace`, `issquare`,
`sqrt`, `fforder`) and FLINT's `fq` (`fq_mul`, `fq_inv`, `fq_pow`, `fq_frobenius`, `fq_norm`,
`fq_trace`, `fq_is_square`, `fq_sqrt`, `fq_multiplicative_order`) compute every output, the
script stops unless they agree (the square roots up to the sign), and the file gets PARI's.

The first irreducible polynomials (`first` lines) are PARI's `polisirreducible` tried on every
monic polynomial in the order of the integers `sum c_i p^i` from `x^k` up: all of them for
`p < 20000`; for larger `p`, the binomials `x^k + c` for `c <= 2000`, then, if none is
irreducible, the candidates from `x^k + x` up (the flag after `k` says whether the binomials
were all tried).

Run: python3 ext_gen.py (gp 2.19 and FLINT 3 installed; the C file is compiled into a temporary
directory). Takes about a minute.
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = (1 << 64) - 1


class Rng:
    """SplitMix64, as the tests' Rng."""

    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def number(self, bits):
        x, left = 0, bits
        while left > 0:
            take = min(left, 64)
            w = self.next() if take == 64 else self.next() >> (64 - take)
            x = (x << take) + w
            left -= take
        return x


BN254 = 21888242871839275222246405745257275088696311157297823662689037894645226208583
BLS = int(
    "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f624"
    "1eabfffeb153ffffb9feffffffffaaab",
    16,
)
M31 = (1 << 31) - 1
M61 = (1 << 61) - 1
P64 = (1 << 64) + 13

# The point at which the digests evaluate the coefficients of an element, modulo p.
DIGEST_T = 0x9E3779B97F4A7C15

GP_PRELUDE = r"""
pv(v) = my(s = ""); for (i = 1, #v, s = Str(s, if (i > 1, ",", ""), v[i])); s;
firstirr(p, k) = {
  my(B, P);
  if (k == 1, return([x, 1]));
  B = if (p < 20000, p - 1, 2000);
  for (c = 0, B, if (polisirreducible(Mod(1, p) * (x^k + c)), return([x^k + c, 1])));
  for (m = p, p^k - 1, P = x^k + Pol(digits(m, p));
    if (polisirreducible(Mod(1, p) * P), return([P, B == p - 1])));
}
"""


def gp(script):
    r = subprocess.run(
        ["gp", "-q", "-f", "--default", "parisizemax=4G"],
        input=GP_PRELUDE + script,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    bad = [l for l in (r.stderr + r.stdout).split("\n") if "***" in l and "Warning" not in l]
    if r.returncode != 0 or bad:
        sys.exit("gp failed:\n" + r.stderr + r.stdout[-2000:])
    return r.stdout


def first_irreducible(cases):
    """PARI's first irreducible polynomials: [(p, k, full, coeffs)]."""
    script = "".join(
        f'my(r = firstirr({p}, {k})); print("{p} {k} ", r[2], " ", pv(Vecrev(r[1])));\n'
        for p, k in cases
    )
    out = []
    for line in gp(script).split("\n"):
        if line.strip():
            p, k, full, c = line.split()
            out.append((int(p), int(k), int(full), [int(v) for v in c.split(",")]))
    return out


def pari_ffinit(p, k):
    return [int(v) for v in gp(f"print(pv(Vecrev(lift(ffinit({p}, {k})))));").split(",")]


def samples_of(p, k, seed, n, ebits):
    """The inputs the tests make: a, b, e, i for each sample."""
    rng = Rng(seed)
    pb = p.bit_length()
    out = []
    for _ in range(n):
        a = [rng.number(pb + 16) % p for _ in range(k)]
        b = [rng.number(pb + 16) % p for _ in range(k)]
        e = rng.number(ebits)
        i = rng.next() % (2 * k)
        out.append((a, b, e, i))
    return out


def parse_results(text):
    rows = []
    for line in text.split("\n"):
        t = line.split()
        if not t:
            continue
        rows.append({t[j]: t[j + 1] for j in range(0, len(t), 2)})
    return rows


def pari_run(p, g, samples, want_ord):
    k = len(g) - 1
    s = [f"p = {p}; k = {k}; g = Mod(1, p) * Pol({list(reversed(g))});"]
    s.append("w = ffgen(g, 'x); el(v) = sum(i = 1, #v, v[i] * w^(i - 1));")
    s.append("co(z) = my(P = z.pol); vector(k, i, lift(polcoef(P, i - 1)));")
    for a, b, e, i in samples:
        s.append(f"a = el({a}); b = el({b});")
        s.append(
            'print("mul ", pv(co(a * b)), " inv ", if (a == 0, "-", pv(co(1 / a))), " pow ", pv(co(a^'
            + str(e)
            + f')), " frob ", pv(co(a^(p^{i % k}))), " norm ", lift(norm(a)), " tr ", '
            + 'lift(trace(a)), " sq ", issquare(a), " sqrt ", if (issquare(a), '
            + 'pv(co(sqrt(a))), "-"), " ord ", '
            + ('if (a == 0, "-", fforder(a))' if want_ord else '"-"')
            + ");"
        )
    return parse_results(gp("\n".join(s) + "\n"))


def flint_run(exe, p, g, samples, want_ord):
    k = len(g) - 1
    s = [f"C {p} {k} " + " ".join(map(str, g))]
    for a, b, e, i in samples:
        s.append("S " + " ".join(map(str, a + b)) + f" {e} {i} {int(want_ord)}")
    s.append("E")
    r = subprocess.run([exe], input="\n".join(s) + "\n", capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit("flint failed:\n" + r.stderr)
    return parse_results(r.stdout)


def canonical(c, p):
    """Of r and -r, the one whose highest nonzero coefficient is below p/2."""
    top = next((v for v in reversed(c) if v), 0)
    return [(-v) % p for v in c] if 2 * top > p else c


def vec(s):
    return [int(v) for v in s.split(",")]


def encode(c, p, digest):
    if digest:
        d, t = 0, 1
        for v in c:
            d = (d + v * t) % p
            t = t * DIGEST_T % p
        return format(d, "x")
    return ",".join(format(v, "x") for v in c)


def main():
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "ext_gen_flint")
    subprocess.run(
        [
            "cc", "-O2", os.path.join(HERE, "ext_gen_flint.c"), "-o", exe,
            "-I/opt/homebrew/include", "-L/opt/homebrew/lib", "-lflint", "-lgmp", "-lmpfr",
        ],
        check=True,
    )

    p600 = int(gp("print(nextprime(2^599 + 2^300));").strip())

    # name, p, g (c_0..c_k) or ("first", k) or ("pari", k), samples, order?, square roots?,
    # exponent bits (None: k·bits(p) + 20), digest?
    cases = [
        ("gf2_1", 2, [1, 1], 12, True, True, None, False),
        ("gf2_8_aes", 2, [1, 1, 0, 1, 1, 0, 0, 0, 1], 6, True, True, None, False),
        ("gf2_8_pari", 2, ("pari", 8), 4, True, True, None, False),
        ("gf2_13", 2, ("first", 13), 4, True, True, None, False),
        ("gf2_64", 2, ("first", 64), 4, True, True, None, False),
        ("gf3_1", 3, [1, 1], 12, True, True, None, False),
        ("gf3_2", 3, ("first", 2), 6, True, True, None, False),
        ("gf3_5_pari", 3, ("pari", 5), 4, True, True, None, False),
        ("gf3_6", 3, ("first", 6), 4, True, True, None, False),
        ("gf3_64", 3, ("first", 64), 3, True, True, None, False),
        ("gf5_4", 5, ("first", 4), 4, True, True, None, False),
        ("gf7_3", 7, ("first", 3), 4, True, True, None, False),
        ("gf7_12", 7, ("first", 12), 4, True, True, None, False),
        ("gf101_4_pari", 101, ("pari", 4), 4, True, True, None, False),
        ("gf101_7", 101, ("first", 7), 4, True, True, None, False),
        ("gf65537_16", 65537, ("first", 16), 3, True, True, None, False),
        ("gfm31_10", M31, ("first", 10), 3, False, True, None, False),
        ("gfm31_24_pari", M31, ("pari", 24), 3, False, True, None, False),
        ("gfm61_5", M61, ("first", 5), 3, False, True, None, False),
        ("gfp64_3", P64, ("first", 3), 3, False, True, None, False),
        ("bn254_2", BN254, ("first", 2), 4, False, True, None, False),
        ("bn254_6", BN254, ("first", 6), 3, False, True, None, True),
        ("bn254_12", BN254, ("first", 12), 2, False, True, None, True),
        ("bls_1", BLS, [5, 1], 12, False, True, None, False),
        ("bls_2", BLS, ("first", 2), 4, False, True, None, False),
        ("bls_3", BLS, ("first", 3), 3, False, True, None, True),
        ("bls_6", BLS, ("first", 6), 3, False, True, None, True),
        ("bls_12", BLS, ("first", 12), 2, False, True, None, True),
        ("bls_64", BLS, ("first", 64), 1, False, True, 400, True),
        ("p600_3", p600, ("first", 3), 3, False, True, None, True),
        # Appended, so that the seeds of the cases above stay as they were.
        ("gf101_1", 101, ("first", 1), 12, True, True, None, False),
        # The products through ModPolynomial for a sparse g over a word prime (from degree 20), two
        # limbs (from 48) and the Big form beyond nine (from 4).
        ("gfm61_24", M61, ("first", 24), 2, False, True, None, True),
        ("gfp64_48", P64, ("first", 48), 1, False, True, None, True),
        ("p600_5", p600, ("first", 5), 2, False, True, None, True),
    ]

    firsts = [(p, k) for p in [2, 3, 5, 7, 11, 13] for k in range(1, 17)]
    firsts += [(2, 24), (2, 32), (2, 64), (3, 24), (3, 32), (3, 64), (5, 24), (17, 8)]
    firsts += [(31, 6), (101, 1), (101, 4), (101, 7), (65537, 16), (65537, 12)]
    firsts += [(M31, k) for k in (2, 3, 4, 5, 6, 10)] + [(M61, k) for k in (2, 3, 4, 5, 6, 24)]
    firsts += [(P64, 3), (P64, 48)] + [(BN254, k) for k in (2, 3, 4, 6, 12, 24, 64)]
    firsts += [(BLS, k) for k in (2, 3, 4, 6, 12, 24, 64)] + [(p600, 3), (p600, 5)]
    first = first_irreducible(firsts)
    first_of = {(p, k): c for p, k, _, c in first}

    lines = [
        "# Reference values for lib/tests/math/field/ext.rs, written by ext_gen.py with PARI/GP 2.19",
        "# and FLINT 3 (fq), which agree on every output. Numbers in hexadecimal.",
        "# first <p> <k> <all binomials tried> <g_0,...,g_k>: the first monic irreducible polynomial.",
        "# field <name> <p> <g_0,...,g_k> <seed> <samples> <ebits> <order> <sqrt> <digest>",
        "# s mul <> inv <> pow <> frob <> norm <> tr <> sq <0|1> sqrt <|-> ord <|->: one sample;",
        "# an element is its coefficients c_0,...,c_(k-1), or with digest 1 the sum of c_i T^i mod p,",
        f"# T = {DIGEST_T:#x}. The square root is the one whose top nonzero coefficient is below p/2.",
    ]
    for p, k, full, c in first:
        lines.append(f"first {p:x} {k} {full} " + ",".join(format(v, "x") for v in c))

    for seed, (name, p, g, n, want_ord, want_sqrt, ebits, digest) in enumerate(cases, 1):
        if isinstance(g, tuple):
            kind, k = g
            g = pari_ffinit(p, k) if kind == "pari" else first_of[(p, k)]
            if kind == "first":
                assert (p, k) in first_of
        k = len(g) - 1
        if ebits is None:
            ebits = k * p.bit_length() + 20
        samples = samples_of(p, k, 1000 + seed, n, ebits)
        pr = pari_run(p, g, samples, want_ord)
        fr = flint_run(exe, p, g, samples, want_ord)
        assert len(pr) == len(fr) == n, name
        if k == 1:
            # Degree 1 must see more than 0: both elements of GF(2); for odd p a nonzero square
            # (a root other than 0's) and a non-square.
            sq = {(a[0] != 0, x["sq"]) for (a, _, _, _), x in zip(samples, pr)}
            want = {(True, "1"), (False, "1")} if p == 2 else {(True, "1"), (True, "0")}
            assert want <= sq, f"{name}: the degree-1 samples cover only {sq}"
        lines.append(
            f"field {name} {p:x} " + ",".join(format(v, "x") for v in g)
            + f" {1000 + seed} {n} {ebits} {int(want_ord)} {int(want_sqrt)} {int(digest)}"
        )
        for (a, b, e, i), x, y in zip(samples, pr, fr):
            for key in ("mul", "inv", "pow", "frob", "norm", "tr", "sq", "ord"):
                if x[key] != y[key]:
                    sys.exit(f"{name}: PARI and FLINT differ on {key}: {x[key]} {y[key]}")
            if (x["sqrt"] == "-") != (y["sqrt"] == "-"):
                sys.exit(f"{name}: PARI and FLINT differ on the square root")
            root = "-"
            if x["sqrt"] != "-":
                r1, r2 = canonical(vec(x["sqrt"]), p), canonical(vec(y["sqrt"]), p)
                if r1 != r2:
                    sys.exit(f"{name}: PARI and FLINT differ on the square root")
                root = encode(r1, p, digest) if want_sqrt else "-"
            enc = lambda key: "-" if x[key] == "-" else encode(vec(x[key]), p, digest)
            ordv = x["ord"] if x["ord"] == "-" else format(int(x["ord"]), "x")
            lines.append(
                f"s mul {enc('mul')} inv {enc('inv')} pow {enc('pow')} frob {enc('frob')} "
                f"norm {int(x['norm']):x} tr {int(x['tr']):x} sq {x['sq']} sqrt {root} ord {ordv}"
            )
        print(name, "ok", file=sys.stderr)

    with open(os.path.join(HERE, "ext_reference.txt"), "w") as out:
        out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
