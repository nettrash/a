#!/usr/bin/env python3
"""Whole digests by Python's hashlib for lib/tests/hash/sha2.rs: the answers that the compression
functions of each run must give, the SHA instructions where the CPU has them and the portable
functions under `RUSTFLAGS="--cfg a_baseline"` and Miri.

Usage: python3 gen_sha_whole.py > sha_whole.txt

Each line: a set of messages, how many, a hash, and the SHA-256 of the concatenation of the
digests of the messages by that hash, in order, in hexadecimal. The sets:
- `lengths`: the messages of every length from 0 to count - 1 bytes, the first bytes of
  m[i] = (i * 167 + 13) mod 256 (those of gen_sha_lengths.c);
- `random`: count messages drawn by SplitMix64 from the seed 0x53484132: each a length,
  next() mod 2049, then its bytes, eight from each next() in little-endian order, the last
  call's surplus dropped.
"""

import hashlib

HASHES = [("SHA224", hashlib.sha224), ("SHA256", hashlib.sha256), ("SHA384", hashlib.sha384), ("SHA512", hashlib.sha512)]
MASK = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)


def lengths(count):
    m = bytes((i * 167 + 13) % 256 for i in range(count))
    return [m[:n] for n in range(count)]


def random_messages(count):
    rng = SplitMix(0x53484132)
    out = []
    for _ in range(count):
        n = rng.next() % 2049
        b = b""
        while len(b) < n:
            b += rng.next().to_bytes(8, "little")
        out.append(b[:n])
    return out


print("# The SHA-256 of the concatenated digests of each set of messages by each hash, by Python's")
print("# hashlib; from gen_sha_whole.py. The smaller sets are those of the tests under Miri.")
for name, count, msgs in [("lengths", 301, lengths), ("lengths", 71, lengths), ("random", 64, random_messages), ("random", 4, random_messages)]:
    ms = msgs(count)
    for hname, h in HASHES:
        whole = b"".join(h(m).digest() for m in ms)
        print(name, count, hname, hashlib.sha256(whole).hexdigest())
