#!/usr/bin/env python3
"""expand_message_xmd (RFC 9380, sections 5.3.1 and 5.3.3) on Python's hashlib, for the lengths,
tags and hashes appendix K leaves out: every SHA-2, the largest outputs, tags of 255 bytes and
longer. Written from the RFC's pseudocode, independently of the crate, for lib/tests/hash/xmd.rs.

Usage: python3 gen_expand_extra.py > expand_extra.txt

Each line: hash, the lengths of the tag and the message, len_in_bytes, the first 16 bytes of the
output and the SHA-256 of the whole output, in hexadecimal. The tag's byte i is (7 i + 3) mod 256,
the message's (13 i + 5) mod 256.
"""

import hashlib

HASHES = {"SHA224": hashlib.sha224, "SHA256": hashlib.sha256, "SHA384": hashlib.sha384, "SHA512": hashlib.sha512}
BLOCK = {"SHA224": 64, "SHA256": 64, "SHA384": 128, "SHA512": 128}


def xmd(name, msg, dst, n):
    h = HASHES[name]
    b = h().digest_size
    if len(dst) > 255:
        dst = h(b"H2C-OVERSIZE-DST-" + dst).digest()
    ell = -(-n // b)
    assert ell <= 255 and n <= 65535 and len(dst) <= 255
    dst_prime = dst + bytes([len(dst)])
    b0 = h(bytes(BLOCK[name]) + msg + n.to_bytes(2, "big") + b"\x00" + dst_prime).digest()
    bi = h(b0 + b"\x01" + dst_prime).digest()
    out = bi
    for i in range(2, ell + 1):
        bi = h(bytes(x ^ y for x, y in zip(b0, bi)) + bytes([i]) + dst_prime).digest()
        out += bi
    return out[:n]


CASES = [
    # (hash, tag length, message length, len_in_bytes)
    ("SHA256", 1, 0, 1),
    ("SHA256", 16, 3, 31),
    ("SHA256", 16, 64, 32),
    ("SHA256", 16, 65, 33),
    ("SHA256", 255, 1000, 64),
    ("SHA256", 256, 0, 65),
    ("SHA256", 1000, 55, 255),
    ("SHA256", 38, 300, 8160),
    ("SHA224", 20, 10, 28),
    ("SHA224", 256, 10, 29),
    ("SHA224", 20, 0, 7140),
    ("SHA384", 1, 0, 48),
    ("SHA384", 255, 111, 49),
    ("SHA384", 256, 112, 96),
    ("SHA384", 40, 128, 12240),
    ("SHA512", 16, 0, 1),
    ("SHA512", 255, 127, 64),
    ("SHA512", 256, 128, 65),
    ("SHA512", 5000, 129, 1000),
    ("SHA512", 38, 5, 16320),
]


def main():
    print("# expand_message_xmd by gen_expand_extra.py on Python's hashlib: hash, len(DST), len(msg),")
    print("# len_in_bytes, the first 16 bytes of the output, SHA-256 of the output")
    for name, ldst, lmsg, n in CASES:
        dst = bytes((7 * i + 3) % 256 for i in range(ldst))
        msg = bytes((13 * i + 5) % 256 for i in range(lmsg))
        out = xmd(name, msg, dst, n)
        print(name, ldst, lmsg, n, out[:16].hex(), hashlib.sha256(out).hexdigest())


if __name__ == "__main__":
    main()
