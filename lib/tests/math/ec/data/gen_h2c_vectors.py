#!/usr/bin/env python3
"""Extracts the test vectors of RFC 9380, appendix J, for the suites in SUITES from the RFC's text
into h2c_vectors.txt, which lib/tests/math/ec/hash.rs reads.

Usage: python3 gen_h2c_vectors.py rfc9380.txt > h2c_vectors.txt

rfc9380.txt is https://www.rfc-editor.org/rfc/rfc9380.txt. Each suite is a block of `name value`
lines ended by a blank line: `suite` and `dst`, then for each vector a `msg` line (the message as
it is, ASCII) and its values (P.x, P.y, u0, u1, Q0.x, ... in hexadecimal without the 0x).
"""

import re
import sys

# The suites of the curves a::math::ec has; BLS12-381's come with the pairing.
SUITES = [
    "P256_XMD:SHA-256_SSWU_RO_",
    "P256_XMD:SHA-256_SSWU_NU_",
    "P384_XMD:SHA-384_SSWU_RO_",
    "P384_XMD:SHA-384_SSWU_NU_",
    "P521_XMD:SHA-512_SSWU_RO_",
    "P521_XMD:SHA-512_SSWU_NU_",
    "curve25519_XMD:SHA-512_ELL2_RO_",
    "curve25519_XMD:SHA-512_ELL2_NU_",
    "edwards25519_XMD:SHA-512_ELL2_RO_",
    "edwards25519_XMD:SHA-512_ELL2_NU_",
    "secp256k1_XMD:SHA-256_SSWU_RO_",
    "secp256k1_XMD:SHA-256_SSWU_NU_",
]

PAGE = re.compile(r"^(Faz-Hernandez, et al\.|RFC 9380 ).*$")
HEADING = re.compile(r"^(Appendix [A-Z]\.|[A-Z]\.[0-9.]+  |[0-9]+\.[0-9.]*  )")
CONTINUATION = "             "


def section(lines, title):
    """The lines of the section whose heading is `title`, up to the next heading."""
    start = lines.index(title)
    end = start + 1
    while end < len(lines) and not HEADING.match(lines[end]):
        end += 1
    return lines[start + 1 : end]


def fields(lines):
    """The `name = value` pairs of a section, in order, continuation lines joined."""
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        m = re.match(r"^   (\S[^=]*?)\s*= ?(.*)$", line)
        if m and not line.startswith(CONTINUATION):
            pairs.append([m.group(1).strip(), m.group(2).strip()])
        elif pairs and line.startswith(CONTINUATION):
            pairs[-1][1] += line.strip()
    return pairs


def main():
    text = open(sys.argv[1], encoding="utf-8-sig").read()
    lines = [l.replace("\f", "") for l in text.split("\n") if not PAGE.match(l.replace("\f", ""))]
    out = sys.stdout
    out.write("# RFC 9380, appendix J: the suites' test vectors, extracted by gen_h2c_vectors.py\n")
    out.write("# from https://www.rfc-editor.org/rfc/rfc9380.txt\n")
    for suite in SUITES:
        title = next(l for l in lines if re.fullmatch(r"J\.[0-9.]+  " + re.escape(suite), l))
        pairs = fields(section(lines, title))
        assert pairs[0] == ["suite", suite] and pairs[1][0] == "dst", pairs[:2]
        out.write(f"suite {suite}\ndst {pairs[1][1]}\n")
        for name, value in pairs[2:]:
            if name == "msg":
                out.write(f"msg {value}\n")
            else:
                assert re.fullmatch(r"[0-9a-f]+", value), (suite, name, value)
                out.write(f"{name.replace('[', '').replace(']', '')} {value}\n")
        out.write("\n")


if __name__ == "__main__":
    main()
