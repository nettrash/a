#!/usr/bin/env python3
"""Extracts the test vectors of RFC 9380, appendix K.1 to K.3 (expand_message_xmd with SHA-256
and SHA-512), from the RFC's text into expand_xmd.txt, which lib/tests/hash/xmd.rs reads.

Usage: python3 gen_expand_xmd.py rfc9380.txt > expand_xmd.txt

rfc9380.txt is https://www.rfc-editor.org/rfc/rfc9380.txt. Each section is a block of
`name value` lines ended by a blank line: `hash`, `dst` and `dst_prime`, then for each vector
`msg` (the message as it is, ASCII), `len` (in decimal) and `uniform` (hexadecimal); msg_prime is
left out.
"""

import re
import sys

SECTIONS = [
    "K.1.  expand_message_xmd(SHA-256)",
    "K.2.  expand_message_xmd(SHA-256) (Long DST)",
    "K.3.  expand_message_xmd(SHA-512)",
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
    out.write("# RFC 9380, appendix K.1 to K.3: expand_message_xmd, extracted by gen_expand_xmd.py\n")
    out.write("# from https://www.rfc-editor.org/rfc/rfc9380.txt (msg_prime left out)\n")
    for title in SECTIONS:
        pairs = fields(section(lines, title))
        head = dict(pairs[:4])
        assert head["name"] == "expand_message_xmd"
        out.write(f"hash {head['hash']}\ndst {head['DST']}\n")
        rest = pairs[4:]
        dst_prime = None
        for i in range(0, len(rest), 5):
            v = dict(rest[i : i + 5])
            assert set(v) == {"msg", "len_in_bytes", "DST_prime", "msg_prime", "uniform_bytes"}
            length = int(v["len_in_bytes"], 16)
            assert len(v["uniform_bytes"]) == 2 * length
            if dst_prime is None:
                dst_prime = v["DST_prime"]
                out.write(f"dst_prime {dst_prime}\n")
            assert v["DST_prime"] == dst_prime
            out.write(f"msg {v['msg']}\nlen {length}\nuniform {v['uniform_bytes']}\n")
        out.write("\n")


if __name__ == "__main__":
    main()
