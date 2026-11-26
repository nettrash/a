#!/usr/bin/env python3
"""Extracts the test vectors of RFC 9380, appendix J.9 and J.10 (the suites of BLS12-381's G1 and
G2), from the RFC's text into h2c_bls12381_vectors.txt, which lib/tests/math/pairing/hash.rs reads.

Usage: python3 gen_h2c_bls12381.py rfc9380.txt > h2c_bls12381_vectors.txt

rfc9380.txt is https://www.rfc-editor.org/rfc/rfc9380.txt. Each suite is a block of `name value`
lines ended by a blank line: `suite` and `dst`, then for each vector a `msg` line (the message as
it is, ASCII) and its values (P.x, P.y, u0, u1, Q0.x, ... in hexadecimal without the 0x). An
element of F_p² = F_p[I]/(I² + 1) is written as its two coordinates, `c0 c1` for c0 + c1·I.
"""

import re
import sys

SUITES = [
    "BLS12381G1_XMD:SHA-256_SSWU_RO_",
    "BLS12381G1_XMD:SHA-256_SSWU_NU_",
    "BLS12381G2_XMD:SHA-256_SSWU_RO_",
    "BLS12381G2_XMD:SHA-256_SSWU_NU_",
]

PAGE = re.compile(r"^(Faz-Hernandez, et al\.|RFC 9380 ).*$")
HEADING = re.compile(r"^(Appendix [A-Z]\.|[A-Z]\.[0-9.]+  |[0-9]+\.[0-9.]*  )")
CONTINUATION = "             "
IMAGINARY = re.compile(r"^\s+\+ I \* ([0-9a-f]+)$")


def section(lines, title):
    """The lines of the section whose heading is `title`, up to the next heading."""
    start = lines.index(title)
    end = start + 1
    while end < len(lines) and not HEADING.match(lines[end]):
        end += 1
    return lines[start + 1 : end]


def fields(lines):
    """The `name = value` pairs of a section, in order: continuation lines joined, and the
    coordinate after `+ I *` kept apart, after a space."""
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        m = re.match(r"^   (\S[^=]*?)\s*= ?(.*)$", line)
        i = IMAGINARY.match(line)
        if m and not line.startswith(CONTINUATION):
            pairs.append([m.group(1).strip(), m.group(2).strip()])
        elif i:
            pairs[-1][1] += " " + i.group(1)
        elif pairs and line.startswith(CONTINUATION):
            pairs[-1][1] += line.strip()
    return pairs


def main():
    text = open(sys.argv[1], encoding="utf-8-sig").read()
    lines = [l.replace("\f", "") for l in text.split("\n") if not PAGE.match(l.replace("\f", ""))]
    out = sys.stdout
    out.write("# RFC 9380, appendix J.9 and J.10: BLS12-381's suites, extracted by gen_h2c_bls12381.py\n")
    out.write("# from https://www.rfc-editor.org/rfc/rfc9380.txt; an element of F_p^2 is `c0 c1`\n")
    for suite in SUITES:
        title = next(l for l in lines if re.fullmatch(r"J\.[0-9.]+  " + re.escape(suite), l))
        pairs = fields(section(lines, title))
        assert pairs[0] == ["suite", suite] and pairs[1][0] == "dst", pairs[:2]
        out.write(f"suite {suite}\ndst {pairs[1][1]}\n")
        g2 = "G2" in suite
        count = 0
        for name, value in pairs[2:]:
            if name == "msg":
                out.write(f"msg {value}\n")
                count += 1
            else:
                parts = value.split(" ")
                assert len(parts) == (2 if g2 else 1), (suite, name, value)
                assert all(re.fullmatch(r"[0-9a-f]+", p) for p in parts), (suite, name, value)
                out.write(f"{name.replace('[', '').replace(']', '')} {value}\n")
        assert count == 5, (suite, count)
        out.write("\n")


if __name__ == "__main__":
    main()
