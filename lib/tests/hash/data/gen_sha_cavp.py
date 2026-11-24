#!/usr/bin/env python3
"""Selects test vectors from NIST's CAVP files for SHA-2 (byte-oriented messages) into the compact
file the tests read.

Usage: python3 gen_sha_cavp.py <dir of shabytetestvectors> <out dir>

The files are those of
https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/shs/shabytetestvectors.zip
(CAVS 11.1). It writes sha_cavp.txt, with lines
  short <alg> <msg hex or -> <digest hex>    from <alg>ShortMsg.rsp, at the lengths in SHORT
  long <alg> <msg hex> <digest hex>          the first LONG vectors of <alg>LongMsg.rsp
  monte <alg> <seed hex>                     from <alg>Monte.rsp, then
  check <alg> <count> <digest hex>           its first MONTE checkpoints
"""

import sys

ALGS = ["SHA224", "SHA256", "SHA384", "SHA512"]
# Byte lengths around the padding boundaries of each block size (64 and 128 bytes).
SHORT = {
    "SHA224": [0, 1, 2, 3, 31, 32, 55, 56, 57, 63, 64],
    "SHA256": [0, 1, 2, 3, 31, 32, 55, 56, 57, 63, 64],
    "SHA384": [0, 1, 2, 3, 63, 64, 111, 112, 113, 127, 128],
    "SHA512": [0, 1, 2, 3, 63, 64, 111, 112, 113, 127, 128],
}
LONG = 2
MONTE = 10


def records(path):
    """The `name = value` records of an .rsp file, one dict per blank-line separated group."""
    out, cur = [], {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            if cur:
                out.append(cur)
                cur = {}
            continue
        name, _, value = line.partition(" = ")
        cur[name] = value
    if cur:
        out.append(cur)
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    with open(f"{out}/sha_cavp.txt", "w") as f:
        f.write("# A selection of NIST CAVP's SHA-2 vectors (CAVS 11.1, shabytetestvectors.zip),\n")
        f.write("# by gen_sha_cavp.py: ShortMsg at the lengths around the block boundaries, the first\n")
        f.write(f"# {LONG} LongMsg vectors and the seed and first {MONTE} checkpoints of the Monte Carlo test.\n")
        for alg in ALGS:
            short = {int(r["Len"]) // 8: r for r in records(f"{src}/{alg}ShortMsg.rsp") if "Len" in r}
            for n in SHORT[alg]:
                r = short[n]
                msg = r["Msg"] if n > 0 else "-"
                assert n == 0 or len(msg) == 2 * n
                f.write(f"short {alg} {msg} {r['MD']}\n")
            long_ = [r for r in records(f"{src}/{alg}LongMsg.rsp") if "Len" in r]
            for r in long_[:LONG]:
                assert len(r["Msg"]) == int(r["Len"]) // 4
                f.write(f"long {alg} {r['Msg']} {r['MD']}\n")
            monte = records(f"{src}/{alg}Monte.rsp")
            seed = next(r["Seed"] for r in monte if "Seed" in r)
            f.write(f"monte {alg} {seed}\n")
            checks = [r for r in monte if "COUNT" in r]
            for r in checks[:MONTE]:
                f.write(f"check {alg} {r['COUNT']} {r['MD']}\n")


if __name__ == "__main__":
    main()
