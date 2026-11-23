#!/usr/bin/env python3
"""Regenerates bkz_fplll.txt, the reference of lib/tests/math/lattice/bkz.rs: `python3 gen_bkz.py`
in this directory.

Each line is a basis made from its seed as the test makes it (`make` of gen_enum.py, mirrored
draw for draw from enumerate.rs) and a block size, with the squared norm of the first vector of
`fplll -a bkz -b <block size>` (fplll 5.5, BKZ 1.0, delta = 0.99, eta = 0.51, no pruning) and the
number of its nonzero rows. The test compares the first vector of its own BKZ with fplll's on
average: the bases differ, the lattices do not.
"""
import sys

sys.dont_write_bytecode = True

from gen_enum import fmt, fplll, make, parse_matrix  # noqa: E402

CASES = (
    [("knapsack", n, n + 1, 10 * n, seed, beta) for n in (10, 14, 18) for seed in (1, 2, 3, 4)
     for beta in (4, 8)]
    + [("qary", n, n, 12, seed, beta) for n in (12, 16) for seed in (1, 2) for beta in (4, 8)]
    + [("random", n, n, 20, seed, beta) for n in (12, 16) for seed in (1, 2) for beta in (6,)]
)

# The ignored sweep: dimensions 30 to 60, block sizes 10 and 20.
HEAVY = (
    [("knapsack", n, n + 1, 10 * n, seed, beta) for n in (30, 40) for seed in (1, 2, 3) for beta in (10, 20)]
    + [("knapsack", 60, 61, 600, seed, 20) for seed in (1, 2)]
    + [("qary", 40, 40, 20, seed, 20) for seed in (1, 2)]
)


def first(case):
    kind, n, m, bits, seed, beta = case
    r = parse_matrix(fplll(["-a", "bkz", "-b", str(beta)], fmt(make(kind, n, m, bits, seed))))
    rows = [row for row in r if any(row)]
    return sum(x * x for x in rows[0]), len(rows)


def main():
    with open("bkz_fplll.txt", "w") as f:
        f.write("# kind n m bits seed beta first_norm2 rank: fplll -a bkz -b beta\n")
        for case in CASES:
            f.write("%s %d %d %d %d %d %d %d\n" % (case + first(case)))
        f.write("# heavy\n")
        for case in HEAVY:
            f.write("%s %d %d %d %d %d %d %d\n" % (case + first(case)))


if __name__ == "__main__":
    main()
