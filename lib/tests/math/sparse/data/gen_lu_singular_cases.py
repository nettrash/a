#!/usr/bin/env python3
"""Exactly singular matrices of signs for the tests of the sparse LU decomposition
(lib/tests/math/sparse/lu.rs): writes lu_singular_cases.rs, next to this script, which the test
file includes.

The matrices are the 35 that review round 2 of the decomposition found called regular at the
default pivot threshold (families ppmm_unit and dupcol_ppmm_unit of its adversarial campaign:
elements -1, 0 and 1, four consecutive rows with r_i + r_{i+1} = r_{i+2} + r_{i+3}, and in the
second family two equal columns as well), each with the column order it was found with. They
come as the review's case files: a line "type threshold n", a line with the column order
q[new] = old, then n rows of n pairs (real part, imaginary part).

Everything the tests rely on is checked here, in exact arithmetic with SymPy, before the file is
written:

* every element is -1, 0 or 1 and real, so the matrix is the same in every element type;
* the rank is n - 1: the matrix is singular, and only just;
* the left null vector y is orthogonal both to the vector of equal elements and to the vector of
  alternating signs whose sizes grow from 1 to 2, which the estimate of the condition used to
  start from: that is why the estimate saw nothing of the singular direction.

Run: python3 gen_lu_singular_cases.py [case directory]
     (SymPy 1.14 when the file was last written; the default directory is that of the review.)
"""

import os
import sys

import sympy

DEFAULT = os.path.expanduser("~/.cache/numadd/w56/sp_lu/review2/cases/dt")


def load(path):
    """The column order and the rows of integers of a case file."""
    with open(path) as f:
        _, _, n = f.readline().split()
        n = int(n)
        order = [int(x) for x in f.readline().split()]
        rows = []
        for _ in range(n):
            pairs = [int(x) for x in f.readline().split()]
            assert len(pairs) == 2 * n, path
            assert all(im == 0 for im in pairs[1::2]), f"{path}: an element is not real"
            rows.append(pairs[0::2])
    assert sorted(order) == list(range(n)), f"{path}: the order is not a permutation"
    assert all(x in (-1, 0, 1) for row in rows for x in row), f"{path}: an element is not a sign"
    return order, rows


def check(name, rows):
    """The exact rank and what the left null vector is orthogonal to."""
    n = len(rows)
    a = sympy.Matrix(rows)
    rank = a.rank()
    assert rank == n - 1, f"{name}: rank {rank}, not {n - 1}"
    (y,) = a.T.nullspace()
    ones = sympy.ones(n, 1)
    alternating = sympy.Matrix(
        [(-1) ** i * (1 + sympy.Rational(i, n - 1)) for i in range(n)]
    )
    assert (y.T * ones)[0] == 0, f"{name}: the left null vector is not orthogonal to ones"
    assert (y.T * alternating)[0] == 0, f"{name}: not orthogonal to the alternating vector"
    return rank


def row_text(row):
    return "".join({1: "+", -1: "-", 0: "."}[x] for x in row)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    names = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))
    out = [
        f"// Written by gen_lu_singular_cases.py with SymPy {sympy.__version__}: do not edit, run the",
        "// script. The 35 exactly singular matrices of signs that review round 2 found called",
        "// regular at the default pivot threshold, each with its column order. Every one has the",
        "// exact rank n - 1, and its left null vector is orthogonal to the vector of equal elements",
        "// and to the vector of alternating signs: `+` is 1, `-` is -1 and `.` is 0.",
        "",
        "const SIGN_CASES: &[SignCase] = &[",
    ]
    for name in names:
        order, rows = load(os.path.join(directory, name))
        rank = check(name, rows)
        print(f"{name}: n = {len(rows)}, exact rank {rank}")
        out.append("    SignCase {")
        out.append(f'        name: "{name[:-4]}",')
        if len(order) <= 16:
            out.append(f"        order: &[{', '.join(str(q) for q in order)}],")
        else:
            out.append("        order: &[")
            for start in range(0, len(order), 16):
                chunk = ", ".join(str(q) for q in order[start : start + 16])
                out.append(f"            {chunk},")
            out.append("        ],")
        out.append("        rows: &[")
        for row in rows:
            out.append(f'            "{row_text(row)}",')
        out.append("        ],")
        out.append("    },")
    out.append("];")
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lu_singular_cases.rs")
    with open(target, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {len(names)} cases to {target}")


if __name__ == "__main__":
    main()
