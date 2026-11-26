#!/usr/bin/env python3
"""Extracts gnark-crypto's published test vectors for hashing to BN254's G1 (RFC 9380's method with
the Shallue-van de Woestijne map; RFC 9380 itself has no BN254 suite) into h2c_bn254_vectors.txt,
which lib/tests/math/pairing/hash.rs reads.

Usage: python3 gen_h2c_bn254.py hash_vectors_test.go > h2c_bn254_vectors.txt

hash_vectors_test.go is
https://raw.githubusercontent.com/Consensys/gnark-crypto/master/ecc/bn254/hash_vectors_test.go
(fetched 2026-11-23). The format is that of h2c_bls12381_vectors.txt: a block per suite, `suite`
and `dst`, then for each vector `msg` and its values in hexadecimal without the 0x.
"""

import re
import sys

BLOCKS = [
    ("encodeToG1Vector", "BN254G1_XMD:SHA-256_SVDW_NU_"),
    ("hashToG1Vector", "BN254G1_XMD:SHA-256_SVDW_RO_"),
]


def main():
    text = open(sys.argv[1]).read()
    out = sys.stdout
    out.write("# gnark-crypto's vectors for hashing to BN254's G1 (ecc/bn254/hash_vectors_test.go),\n")
    out.write("# extracted by gen_h2c_bn254.py\n")
    for var, suite in BLOCKS:
        start = text.index(var + " = ")
        end = text.index("}}", start)
        block = text[start:end]
        dst = re.search(r'dst: \[\]byte\("([^"]*)"\)', block).group(1)
        assert dst.endswith(suite), (dst, suite)
        out.write(f"suite {suite}\ndst {dst}\n")
        cases = re.findall(r"\{\s*(msg: .*?)\n\s*\}", block, re.S)
        assert len(cases) == 5, (var, len(cases))
        for case in cases:
            msg = re.search(r'msg: "([^"]*)"', case).group(1)
            out.write(f"msg {msg}\n")
            for name, x, y in re.findall(r'(\w+): point\{"0x([0-9a-f]+)", "0x([0-9a-f]+)"\}', case):
                out.write(f"{name}.x {x}\n{name}.y {y}\n")
            for name, v in re.findall(r'(u\d?): "0x([0-9a-f]+)"', case):
                out.write(f"{'u0' if name == 'u' else name} {v}\n")
        out.write("\n")


if __name__ == "__main__":
    main()
