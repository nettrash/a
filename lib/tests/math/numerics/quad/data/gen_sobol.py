"""The Sobol' direction numbers of the library, and the reference points its tests compare with.

1. Writes lib/src/math/numerics/quad/sobol_table.rs: the primitive polynomials and initial direction
   numbers of Joe and Kuo's table new-joe-kuo-6.21201 (S. Joe and F. Y. Kuo, "Constructing Sobol
   sequences with better two-dimensional projections", SIAM J. Sci. Comput. 30 (2008) 2635-2654),
   as SciPy ships them in scipy/stats/_sobol_direction_numbers.npz, for dimensions 2 to 1024
   (dimension 1 is the van der Corput sequence and needs none).
2. Writes sobol_ref.txt next to this script: for every dimension d from 1 to 1024 the point of
   index 1000 of scipy.stats.qmc.Sobol(d, scramble=False, bits=32) in dimension d (as 32-bit
   integers), and a checksum of the first 4096 points, sum over i < 4096 of (i + 1) * x_i (as a
   32-bit integer) modulo 2^64; and the first 64 points of dimensions 1 to 8 in full.

Run: python3 gen_sobol.py   (numpy, scipy; about a minute)
"""
import os
import numpy as np
import scipy.stats as st
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "..", "..", "..", "..", "..", "src", "math", "numerics", "quad",
                     "sobol_table.rs")
REF = os.path.join(HERE, "sobol_ref.txt")
DIMENSIONS = 1024


def table():
    z = np.load(os.path.join(os.path.dirname(st.__file__), "_sobol_direction_numbers.npz"))
    poly, vinit = z["poly"], z["vinit"]
    words = []
    for j in range(1, DIMENSIONS):
        p = int(poly[j])
        s = p.bit_length() - 1
        words.append(p)
        words.extend(int(m) for m in vinit[j, :s])
    assert max(words) < 1 << 16
    lines = [
        "//! The direction numbers of the Sobol' sequence: Joe and Kuo's table new-joe-kuo-6.21201",
        "//! (S. Joe and F. Y. Kuo, \"Constructing Sobol sequences with better two-dimensional",
        "//! projections\", SIAM J. Sci. Comput. 30, 2008), as SciPy ships it, for dimensions 2 to",
        f"//! {DIMENSIONS}. Written by `tests/math/numerics/quad/data/gen_sobol.py`; do not edit.",
        "//!",
        "//! Copyright (c) 2008, Frances Y. Kuo and Stephen Joe. All rights reserved. The table is",
        "//! redistributed under its authors' BSD-style licence, whose conditions are that a",
        "//! redistribution of source keeps this notice, that a redistribution in binary form",
        "//! reproduces it in its documentation, and that the names of the authors and of their",
        "//! universities are not used to endorse what is made from it; the full text, and its",
        "//! disclaimer of warranty, accompany the table at <https://web.maths.unsw.edu.au/~fkuo/sobol/>.",
        "//!",
        "//! For each dimension from the second, in order: the primitive polynomial over GF(2) as the",
        "//! integer `2^s + 2a + 1` (its degree `s`, the coefficients `a` of its middle terms), then the",
        "//! `s` initial direction numbers `m_1 … m_s`.",
        "",
        f"/// The dimensions the table covers.",
        f"pub(super) const DIMENSIONS: usize = {DIMENSIONS};",
        "",
        "/// The polynomials and initial direction numbers, dimension after dimension.",
        "#[rustfmt::skip]",
        f"pub(super) static TABLE: [u16; {len(words)}] = [",
    ]
    row = []
    for w in words:
        row.append(f"{w},")
        if len(row) == 16:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("];")
    with open(TABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.normpath(TABLE), len(words), "words")


def reference():
    out = ["# Sobol' points of scipy.stats.qmc.Sobol(d, scramble=False, bits=32): see gen_sobol.py",
           "# point1000 d x_1000[d-1]  /  checksum d sum_{i<4096} (i+1) * x_i[d-1] mod 2^64",
           "# full d i x_i[0] ... x_i[d-1] for d <= 8, i < 64"]
    m = 1 << 64
    pts = qmc.Sobol(d=DIMENSIONS, scramble=False, bits=32).random(4096)
    ints = np.round(pts * 2.0**32).astype(np.uint64)
    for d in range(1, DIMENSIONS + 1):
        col = ints[:, d - 1]
        total = 0
        for i, v in enumerate(col):
            total = (total + (i + 1) * int(v)) % m
        out.append(f"point1000 {d} {int(col[1000])}")
        out.append(f"checksum {d} {total}")
    for d in range(1, 9):
        p = np.round(qmc.Sobol(d=d, scramble=False, bits=32).random(64) * 2.0**32).astype(np.uint64)
        for i in range(64):
            out.append(f"full {d} {i} " + " ".join(str(int(v)) for v in p[i]))
    with open(REF, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", REF, len(out), "lines")


if __name__ == "__main__":
    table()
    reference()
