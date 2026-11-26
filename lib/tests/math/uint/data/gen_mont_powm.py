#!/usr/bin/env python3
"""Writes mont_powm.txt: pow(b, e, m) by Python's built-in pow on seeded random operands.

Each line: N (limbs), then m, b, e and pow(b, e, m) in hexadecimal. The moduli are odd and even
(with few and many trailing zeros), of full and random length; the bases are below R = 2^(64 N),
often above m; the exponents have up to 1024 bits (they are read as Uint<16>).
Run: python3 gen_mont_powm.py > mont_powm.txt
"""
import random

rng = random.Random(25_2026)
print("# N m b e pow(b, e, m), hexadecimal; by gen_mont_powm.py")
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16]:
    w = 64 * n
    for kind in range(5):
        if kind == 0:  # odd, full length
            m = rng.getrandbits(w) | (1 << (w - 1)) | 1
        elif kind == 1:  # even, full length, few trailing zeros
            m = (rng.getrandbits(w - 3) | (1 << (w - 4)) | 1) << 3
        elif kind == 2:  # odd, random length
            bits = rng.randint(2, w)
            m = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        elif kind == 3:  # even, many trailing zeros
            z = rng.randint(1, w - 1)
            m = (rng.getrandbits(w - z) | 1) << z
            if m == 0 or m.bit_length() > w:
                m = 1 << (w - 1)
        else:  # the top limb all ones, odd
            m = (((1 << 64) - 1) << (w - 64)) | rng.getrandbits(w - 64) | 1
        b = rng.getrandbits(w)
        e = rng.getrandbits(rng.choice([1, 17, 64, 255, 256, 511, 700, 1024]))
        print(n, format(m, "x"), format(b, "x"), format(e, "x"), format(pow(b, e, m), "x"))
