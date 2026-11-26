"""Reference outputs for the tests of a::math::pairing and the towers of a::math::field from
py_ecc 8 (the plain exponent (p^12 - 1)/r), PARI/GP 2.19 (norms, traces and squares in F_p^12
as F_p[w]/(w^12 - 2 xi0 w^6 + xi0^2 + 1)), the IETF draft on pairing-friendly curves (its vector
of e(P, Q) on BLS12-381) and the test vectors of the bls12_381 crate (their FNV-1a digests).

Run with the py_ecc of a virtual environment and gp on the path:
    python3 gen_py.py <bls12_381 crate's src/tests directory> > py.txt
Prints `key value` lines, merged into pairing_ref.txt and ../../field/data/tower_ref.txt.
"""
import subprocess
import sys

from py_ecc import optimized_bls12_381 as B, optimized_bn128 as N

M64 = (1 << 64) - 1


class Rng:
    """SplitMix64, as the tests have it."""

    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def number(self, words):
        """`words` words, the first the least significant."""
        return sum(self.next() << (64 * i) for i in range(words))


class Fnv:
    def __init__(self):
        self.h = 0xCBF29CE484222325

    def eat(self, b):
        for x in b:
            self.h ^= x
            self.h = (self.h * 0x100000001B3) & M64


CURVES = {
    # name: (module, xi0, words of a random element of F_p, bytes of an element)
    "bls": (B, 1, 7, 48),
    "bn": (N, 9, 5, 32),
}


def to_tower(flat, xi0, p):
    """F_p^12 = F_p[w]/(w^6 = xi0 + u) as polynomial in w to the tower order c0.c0.c0, c0.c0.c1,
    c0.c1.c0, ..., c1.c2.c1: the coefficient of u^e v^j w^i is that of w^(2j+i) (and w^(2j+i+6))."""
    out = []
    for i in range(2):
        for j in range(3):
            d = 2 * j + i
            b1 = flat[d + 6] % p
            b0 = (flat[d] + xi0 * b1) % p
            out += [b0, b1]
    return out


def to_flat(tower, xi0, p):
    flat = [0] * 12
    for i in range(2):
        for j in range(3):
            d = 2 * j + i
            b0, b1 = tower[i * 6 + j * 2], tower[i * 6 + j * 2 + 1]
            flat[d] = (flat[d] + b0 - xi0 * b1) % p
            flat[d + 6] = (flat[d + 6] + b1) % p
    return flat


def eat_all(h, values, n):
    for v in values:
        h.eat((v % (1 << (8 * n))).to_bytes(n, "big"))


def pairings_py_ecc(name):
    """e(aG1, bG2) for the first two seeded pairs of the tests, as py_ecc computes it, raised to
    the factor c of arkworks: -3 for BLS12-381 (py_ecc loops over |x| and does not conjugate, so
    its value is the inverse of f_x's), 2x(6x^2 + 3x + 1) for BN254; and as py_ecc gives it."""
    M, xi0, _, n = CURVES[name]
    p = M.field_modulus
    if name == "bls":
        c = -3
    else:
        x = 4965661367192848881
        c = 2 * x * (6 * x * x + 3 * x + 1)
    c %= M.curve_order
    r = Rng(2)
    h = Fnv()
    raw = Fnv()
    for _ in range(2):
        a = r.number(4)
        b = r.number(4)
        e = M.pairing(M.multiply(M.G2, b), M.multiply(M.G1, a))
        eat_all(raw, to_tower([int(x) for x in e.coeffs], xi0, p), n)
        eat_all(h, to_tower([int(x) for x in (e ** c).coeffs], xi0, p), n)
    print(f"{name}_pairing_py_ecc {h.h:016x}")
    # py_ecc's own values, which the documented recipe reproduces from ours.
    print(f"{name}_pairing_py_ecc_raw {raw.h:016x}")


def ietf():
    """The IETF draft's e(BP, BP') on BLS12-381 (appendix B), naive final exponent: its twelve
    e_i are the tower coefficients c0.c0.c0, c0.c0.c1, ..., c1.c2.c1 of e (checked here against
    py_ecc's value, the inverse of f_x's)."""
    e = [
        0x11619B45F61EDFE3B47A15FAC19442526FF489DCDA25E59121D9931438907DFD448299A87DDE3A649BDBA96E84D54558,
        0x153CE14A76A53E205BA8F275EF1137C56A566F638B52D34BA3BF3BF22F277D70F76316218C0DFD583A394B8448D2BE7F,
        0x095668FB4A02FE930ED44767834C915B283B1C6CA98C047BD4C272E9AC3F3BA6FF0B05A93E59C71FBA77BCE995F04692,
        0x16DEEDAA683124FE7260085184D88F7D036B86F53BB5B7F1FC5E248814782065413E7D958D17960109EA006B2AFDEB5F,
        0x09C92CF02F3CD3D2F9D34BC44EEE0DD50314ED44CA5D30CE6A9EC0539BE7A86B121EDC61839CCC908C4BDDE256CD6048,
        0x111061F398EFC2A97FF825B04D21089E24FD8B93A47E41E60EAE7E9B2A38D54FA4DEDCED0811C34CE528781AB9E929C7,
        0x01ECFCF31C86257AB00B4709C33F1C9C4E007659DD5FFC4A735192167CE197058CFB4C94225E7F1B6C26AD9BA68F63BC,
        0x08890726743A1F94A8193A166800B7787744A8AD8E2F9365DB76863E894B7A11D83F90D873567E9D645CCF725B32D26F,
        0x0E61C752414CA5DFD258E9606BAC08DAEC29B3E2C57062669556954FB227D3F1260EEDF25446A086B0844BCD43646C10,
        0x0FE63F185F56DD29150FC498BBEEA78969E7E783043620DB33F75A05A0A2CE5C442BEAFF9DA195FF15164C00AB66BDDE,
        0x10900338A92ED0B47AF211636F7CFDEC717B7EE43900EEE9B5FC24F0000C5874D4801372DB478987691C566A8C474978,
        0x1454814F3085F0E6602247671BC408BBCE2007201536818C901DBD4D2095DD86C1EC8B888E59611F60A301AF7776BE3D,
    ]
    p = B.field_modulus
    py = B.pairing(B.G2, B.G1)
    inv = to_tower([int(x) for x in (py ** (B.curve_order - 1)).coeffs], 1, p)
    # Which order of the e_i is the tower's.
    orders = {
        "tower": list(range(12)),
        "by w": [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11],
    }
    for k, o in orders.items():
        if [e[i] for i in o] == inv:
            print(f"# the IETF e_i are in the {k} order and equal f_x(P)^((p^12 - 1)/r)", file=sys.stderr)
            return
    raise SystemExit("no order of the IETF e_i matched")


def pari():
    """Norms, traces and squares in the towers, by PARI on F_p[w]/(w^12 - 2 xi0 w^6 + xi0^2 + 1)
    for 8 seeded elements of F_p^12, F_p^6 and F_p^2 per curve. An element c of F_p^2 is a square
    of F_p^2 when c^((p^2 - 1)/2) = 1 (in F_p^12 every element of F_p^2 is a square)."""
    lines = []
    for name, (M, xi0, words, n) in CURVES.items():
        p = M.field_modulus
        r = Rng(5)
        cases = []
        for _ in range(8):
            a = [r.number(words) % p for _ in range(12)]
            b = [r.number(words) % p for _ in range(6)] + [0] * 6
            c = [r.number(words) % p for _ in range(2)] + [0] * 10
            cases.append((a, b, c))
        gp = [f"p={p}; w=ffgen(Mod(1,p)*('w^12-{2*xi0}*'w^6+{xi0*xi0+1}),'w);",
              "f(c)=sum(i=0,11,c[i+1]*w^i);",
              "o(x)=my(v=Vec(x.pol));v=concat(vector(12-#v),v);print(Vecrev(v));"]
        for a, b, c in cases:
            A, Bf, C = (to_flat(x, xi0, p) for x in (a, b, c))
            gp.append(f"a=f({A}); b=f({Bf}); c=f({C});")
            gp.append("o(a*a^(p^6)); o(a+a^(p^6)); o(b*b^(p^2)*b^(p^4)); o(b+b^(p^2)+b^(p^4));"
                      " o(c*c^p); o(c+c^p); print([lift(norm(a)), lift(trace(a))]);"
                      " print([c^((p^2-1)/2)==1, (c^2)^((p^2-1)/2)==1, issquare(a), issquare(a^2)]);")
        out = subprocess.run(["gp", "-q", "-f"], input="\n".join(gp) + "\n", capture_output=True,
                             text=True, timeout=600, check=True).stdout.split("\n")
        out = [eval(l.replace("[", "[").replace(";", ",")) for l in out if l.strip()]
        keys = ["fp12_norm", "fp12_trace", "fp6_norm", "fp6_trace", "fp2_norm", "fp2_trace",
                "absolute", "squares"]
        hs = {k: Fnv() for k in keys}
        for i in range(8):
            rows = out[i * 8:(i + 1) * 8]
            for k, row in zip(keys[:6], rows[:6]):
                t = to_tower(row, xi0, p)
                if k.startswith("fp12"):
                    t = t[:6]  # in F_p^6: c1 = 0
                elif k.startswith("fp6"):
                    t = t[:2]  # in F_p^2
                else:
                    t = t[:1]  # in F_p
                eat_all(hs[k], t, n)
            eat_all(hs["absolute"], rows[6], n)
            hs["squares"].eat(bytes(rows[7]))
        for k in keys:
            lines.append(f"{name}_pari_{k} {hs[k].h:016x}")
    print("\n".join(lines))


def zcash(directory):
    """The FNV-1a digests of the bls12_381 crate's encodings of [i]G for i in 0..1000."""
    for f in ["g1_compressed", "g1_uncompressed", "g2_compressed", "g2_uncompressed"]:
        h = Fnv()
        data = open(f"{directory}/{f}_valid_test_vectors.dat", "rb").read()
        h.eat(data)
        print(f"zcash_{f} {h.h:016x} {len(data)}")


if __name__ == "__main__":
    zcash(sys.argv[1])
    ietf()
    pari()
    for name in CURVES:
        pairings_py_ecc(name)
