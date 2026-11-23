"""The Matrix Market samples of tests/math/sparse/io.rs, and the check of what the crate writes.

    python3 gen_matrix_market.py write    # scipy/*.mtx and expected/*.txt, from the matrices below
    python3 gen_matrix_market.py check    # reads ours/f64/*.mtx and ours/f32/*.mtx with scipy
    python3 gen_matrix_market.py lenient  # scipy's readings of the banners the format leaves out

`write` builds one matrix for every field and symmetry the format defines, writes it with
scipy.io.mmwrite in coordinate format and, but for a pattern, in array format, reads each file
back with scipy.io.mmread, and records in expected/<field>_<symmetry>.txt the matrix scipy reads:
for every nonzero cell, the bits of its parts as f64 and the bits of the nearest f32 to each
number of the file (summed in f32 where a cell is named twice), rounded exactly from the decimal
digits, which is what a correct reader of f32 must give. The f64 values are checked against
scipy's own reading before anything is written.

`check` reads the files the crate's writer produced (tests/math/sparse/io.rs writes them into a
directory when A_MATRIX_MARKET_OUT names one; the ones in ours/ are those) with scipy.io.mmread and
compares them, bit for bit, with expected/: the f64 files as f64, the f32 files through the exact
f32 rounding of their digits and through numpy's float32 of scipy's f64.

Requires numpy and scipy (the fast_matrix_market reader and writer of scipy >= 1.12).
"""

import io
import math
import os
import struct
import sys
import warnings
from fractions import Fraction

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
COMMENT = "written by gen_matrix_market.py\nwith scipy.io.mmwrite"

# The combinations the definition of the format allows: (field, symmetry, formats).
KINDS = [
    ("real", "general", ("coordinate", "array")),
    ("real", "symmetric", ("coordinate", "array")),
    ("real", "skew-symmetric", ("coordinate", "array")),
    ("integer", "general", ("coordinate", "array")),
    ("integer", "symmetric", ("coordinate", "array")),
    ("integer", "skew-symmetric", ("coordinate", "array")),
    ("complex", "general", ("coordinate", "array")),
    ("complex", "symmetric", ("coordinate", "array")),
    ("complex", "skew-symmetric", ("coordinate", "array")),
    ("complex", "hermitian", ("coordinate", "array")),
    ("pattern", "general", ("coordinate",)),
    ("pattern", "symmetric", ("coordinate",)),
]

# Numbers that are hard to read: digits that round, the edges of both precisions, and values
# whose shortest f64 digits round to another f32 than the f64 itself does (1 + 2^-24, and the
# f32 overflow threshold 2^128 - 2^103).
HARD = [
    1.0 / 3.0,
    0.1,
    -2.5e-7,
    1e-300,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -5e19,
    123456789.0,
    1.0 + 2.0**-24,
    16777217.0,
    2.0**128 - 2.0**103,
    1e-45,
    -1e-50,
    math.pi,
    -0.75,
    3.4028234663852886e38,
    -1.401298464324817e-45,
]
INTEGERS = [
    7,
    -3,
    2**53 + 1,
    -(2**62),
    16777217,
    123456789,
    -1,
    2,
    2**62 - 1,
    1,
    -16777219,
    99,
]


def kind_name(field, symmetry):
    return f"{field}_{symmetry}"


def file_name(fmt, field, symmetry):
    return f"{fmt}_{field}_{symmetry}.mtx"


def triplets(field, symmetry):
    """The matrix as triplets (i, j, value), 0-based, with a repeated cell and, but for a
    pattern, an explicit zero: what the coordinate file holds. The array file holds their sums.
    For a symmetry, the triplets are those on and below the diagonal (below it for
    skew-symmetric); the matrix is their expansion."""
    if field == "pattern":
        if symmetry == "general":
            cells = [(0, 0), (2, 0), (4, 1), (1, 2), (3, 3), (0, 3), (2, 0)]
            return 5, 4, [(i, j, 1.0) for i, j in cells]
        cells = [(0, 0), (2, 0), (4, 1), (2, 2), (3, 2), (4, 4), (4, 1)]
        return 5, 5, [(i, j, 1.0) for i, j in cells]
    if field == "integer":
        values = [float(v) if abs(v) < 2**53 else v for v in INTEGERS]
    elif field == "real":
        values = HARD
    else:
        values = [
            complex(1.0 / 3.0, -0.1),
            complex(0.0, 1.0),
            complex(-2.5e-7, 1e300),
            complex(5e-324, -0.0),
            complex(1.0 + 2.0**-24, 16777217.0),
            complex(-1.0, 0.0),
            complex(3.0, 4.0),
            complex(2.0**128 - 2.0**103, -1e-45),
            complex(-5e19, math.pi),
            complex(0.1, 0.2),
            complex(-0.75, -0.125),
            complex(1e-300, -1.0),
        ]
    square = symmetry != "general"
    rows, cols = (7, 7) if square else ((5, 7) if field == "complex" else (7, 5))
    if symmetry == "general":
        cells = [(i, j) for j in range(cols) for i in range(rows) if (3 * i + 5 * j) % 7 < 4]
    elif symmetry == "skew-symmetric":
        cells = [(i, j) for j in range(7) for i in range(j + 1, 7) if (i + 2 * j) % 7 != 3]
    else:
        cells = [(i, j) for j in range(7) for i in range(j, 7) if (i + 2 * j) % 7 != 3]
    # Every number of the list is in the matrix.
    assert len(cells) >= len(values), (field, symmetry, len(cells))
    out = []
    for k, (i, j) in enumerate(cells):
        v = values[k % len(values)]
        if symmetry == "hermitian" and i == j:
            v = complex(v.real, 0.0)
        out.append((i, j, v))
    # A cell named twice, with parts that sum exactly in both precisions, and an explicit zero in
    # a cell that holds nothing else, which a reader must not store.
    i, j = cells[1]
    if field == "complex":
        extra = complex(0.5, 0.25) if not (symmetry == "hermitian" and i == j) else complex(0.5, 0)
    else:
        extra = 0.5 if field != "integer" else 5.0
    out.append((i, j, extra))
    first = {"general": 0, "skew-symmetric": 1}.get(symmetry, 0)
    zero_cell = next(
        (i, j)
        for j in range(cols)
        for i in range(j + first if square else 0, rows)
        if (i, j) not in cells
    )
    out.append((*zero_cell, 0.0 if field != "complex" else complex(0.0, 0.0)))
    return rows, cols, out


def mirror(symmetry, v):
    if symmetry == "skew-symmetric":
        return -v
    if symmetry == "hermitian":
        return v.conjugate()
    return v


def full_matrix(field, symmetry):
    """The dense matrix the triplets stand for, in numpy, the cells summed in order."""
    rows, cols, ts = triplets(field, symmetry)
    dtype = {"real": np.float64, "integer": np.int64, "complex": np.complex128,
             "pattern": np.float64}[field]
    m = np.zeros((rows, cols), dtype=dtype)
    seen = set()

    def add(i, j, v):
        # The first value of a cell as it is, and the others added: 0 + (−0) would lose a sign.
        v = v if field != "integer" else int(v)
        if (i, j) in seen:
            m[i, j] += v
        else:
            m[i, j] = v
            seen.add((i, j))

    for i, j, v in ts:
        add(i, j, v)
        if symmetry != "general" and i != j:
            add(j, i, mirror(symmetry, v))
    return m


def write_scipy(fmt, field, symmetry, path):
    rows, cols, ts = triplets(field, symmetry)
    if fmt == "array":
        a = full_matrix(field, symmetry)
    else:
        # The coordinate file holds the triplets themselves; with a symmetry, scipy writes the
        # entries on and below the diagonal of what it is given, so it is given the expansion.
        dtype = {"real": np.float64, "integer": np.int64, "complex": np.complex128,
                 "pattern": np.float64}[field]
        rr, cc, vv = [], [], []
        for i, j, v in ts:
            rr.append(i)
            cc.append(j)
            vv.append(v if field != "integer" else int(v))
            if symmetry != "general" and i != j:
                rr.append(j)
                cc.append(i)
                w = mirror(symmetry, v)
                vv.append(w if field != "integer" else int(w))
        a = sp.coo_matrix((np.array(vv, dtype=dtype), (rr, cc)), shape=(rows, cols))
    kwargs = {"comment": COMMENT, "symmetry": symmetry}
    if field == "pattern":
        kwargs["field"] = "pattern"
    elif field == "integer":
        kwargs["field"] = "integer"
    scipy.io.mmwrite(path, a, **kwargs)


def f64_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def complex_of_bits(re64, im64):
    """The complex number of the bits of its two parts as f64."""
    return complex(*(struct.unpack("<d", struct.pack("<Q", b))[0] for b in (re64, im64)))


def f32_bits_of_value(x):
    """The bits of the f32 `x`, which a Python float holds exactly."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_of_text(text):
    """The f32 nearest to the decimal number `text`, ties to even, subnormals and the overflow to
    infinity included, as a Python float: rounded once, exactly, from the digits."""
    t = text.strip().lower().replace("d", "e")
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    if t in ("inf", "infinity"):
        return sign * math.inf
    if t == "nan":
        return math.nan
    x = Fraction(t)
    if x == 0:
        return sign * 0.0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    q = max(e, -126) - 23
    m = x / Fraction(2) ** q
    n = math.floor(m)
    rest = m - n
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    value = Fraction(n) * Fraction(2) ** q
    if value >= Fraction(2) ** 128:
        return sign * math.inf
    return sign * float(value)


def entries_of_file(text):
    """The entries of a Matrix Market file as (format, field, symmetry, rows, cols, list of
    (i, j, tokens)), 0-based, in the order of the file, as written: the lower triangle alone
    where there is a symmetry."""
    lines = text.splitlines()
    banner = lines[0].split()
    fmt, field, symmetry = banner[2].lower(), banner[3].lower(), banner[4].lower()
    data = [line for line in lines[1:] if line.strip() and not line.lstrip().startswith("%")]
    size = [int(t) for t in data[0].split()]
    rows, cols = size[0], size[1]
    out = []
    if fmt == "coordinate":
        for line in data[1:]:
            tok = line.split()
            out.append((int(tok[0]) - 1, int(tok[1]) - 1, tok[2:]))
    else:
        cells = []
        for j in range(cols):
            first = {"general": 0, "skew-symmetric": j + 1}.get(symmetry, j)
            for i in range(first, rows):
                cells.append((i, j))
        assert len(cells) == len(data) - 1, "an array file of the wrong length"
        for (i, j), line in zip(cells, data[1:]):
            out.append((i, j, line.split()))
    return fmt, field, symmetry, rows, cols, out


def f32_sum(a, b):
    """`a + b` in f32, of two f32 values or of two complex numbers of f32 parts."""
    if isinstance(a, complex):
        return complex(f32_sum(a.real, b.real), f32_sum(a.imag, b.imag))
    return float(np.float32(a) + np.float32(b))


def cells_of_file(text):
    """The cells of the matrix a file holds, expanded, each as (f64 value, f32 value): the
    numbers of the file rounded once to each precision, a cell named twice summed in its own
    precision in the order of the file, the first value of a cell taken as it is. The values of
    a file that isn't complex are real numbers, mirrored as such; a complex reader gives them the
    imaginary part +0."""
    fmt, field, symmetry, rows, cols, entries = entries_of_file(text)
    cells64, cells32 = {}, {}

    def add(i, j, v64, v32):
        if (i, j) in cells64:
            cells64[(i, j)] += v64
            cells32[(i, j)] = f32_sum(cells32[(i, j)], v32)
        else:
            cells64[(i, j)] = v64
            cells32[(i, j)] = v32

    for i, j, tok in entries:
        if field == "pattern":
            v64, v32 = 1.0, 1.0
        elif field == "complex":
            v64 = complex(float(tok[0]), float(tok[1]))
            v32 = complex(f32_of_text(tok[0]), f32_of_text(tok[1]))
        else:
            v64, v32 = float(tok[0]), f32_of_text(tok[0])
        if fmt == "array" and v64 == 0:
            continue
        add(i, j, v64, v32)
        if symmetry != "general" and i != j:
            add(j, i, mirror(symmetry, v64), mirror(symmetry, v32))
    # Complex numbers from here on, the real ones with the imaginary part +0.
    cells64 = {k: complex(v) for k, v in cells64.items()}
    cells32 = {k: complex(v) for k, v in cells32.items()}
    return rows, cols, field, cells64, cells32


def read_by_scipy(path):
    """The shape of the matrix scipy.io.mmread reads from a file, and its cells {(i, j): value}:
    the entries scipy parsed, those of one cell summed with the first taken as it is (scipy's own
    toarray adds them to zeros, which drops the sign of a zero part), and of an array the
    nonzeros."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = scipy.io.mmread(path)
    cells = {}
    if sp.issparse(a):
        a = a.tocoo()
        for i, j, v in zip(a.row.tolist(), a.col.tolist(), a.data.tolist()):
            cells[(i, j)] = cells[(i, j)] + v if (i, j) in cells else v
    else:
        a = np.asarray(a)
        for (i, j), v in np.ndenumerate(a):
            if v != 0:
                cells[(i, j)] = v.item()
    return a.shape, cells


def scipy_dense(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = scipy.io.mmread(path)
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def parts64(v, signed_zeros=True):
    """The bits of the two parts of `v` as f64; with `signed_zeros` false, a zero part as +0."""
    v = complex(v)
    parts = (v.real, v.imag) if signed_zeros else (v.real or 0.0, v.imag or 0.0)
    return f64_bits(parts[0]), f64_bits(parts[1])


def same_cells(mine, theirs, signed_zeros=True):
    """Whether two dictionaries of cells hold the same matrix, bit for bit where a cell isn't
    zero: (True, None, None, None), or (False, the cell, the two values). scipy's reader of the
    array format reads -0 as +0 (its coordinate reader keeps the sign), so for an array file the
    sign of a zero part is not compared."""
    for key in sorted(set(mine) | set(theirs)):
        a, b = mine.get(key, 0), theirs.get(key, 0)
        if a == 0 and b == 0:
            continue
        if parts64(a, signed_zeros) != parts64(b, signed_zeros):
            return False, key, a, b
    return True, None, None, None


def write():
    os.makedirs(os.path.join(HERE, "scipy"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "expected"), exist_ok=True)
    differing = []
    for field, symmetry, formats in KINDS:
        tables = []
        for fmt in formats:
            path = os.path.join(HERE, "scipy", file_name(fmt, field, symmetry))
            write_scipy(fmt, field, symmetry, path)
            with open(path) as f:
                text = f.read()
            rows, cols, _, c64, c32 = cells_of_file(text)
            # The reading of the digits here must be scipy's: every cell, bit for bit.
            shape, read = read_by_scipy(path)
            assert shape == (rows, cols), path
            if field == "integer":
                read = {k: float(v) for k, v in read.items()}
            same, key, a, b = same_cells(c64, read, signed_zeros=fmt == "coordinate")
            assert same, (path, key, a, b)
            # And the matrix the triplets stand for.
            assert np.array_equal(scipy_dense(path), full_matrix(field, symmetry)), path
            table = []
            for (i, j) in sorted(c64):
                v64, v32 = c64[(i, j)], c32[(i, j)]
                if v64 == 0:
                    continue
                for x in (v32.real, v32.imag):
                    with np.errstate(over="ignore"):
                        assert math.isnan(x) or float(np.float32(x)) == x, ("not an f32", x)
                re32, im32 = f32_bits_of_value(v32.real), f32_bits_of_value(v32.imag)
                re64, im64 = parts64(v64)
                table.append((i, j, re64, im64, re32, im32))
                # Where the f32 of the digits isn't the f32 of the f64, a reader that goes
                # through f64 is wrong: the samples hold such numbers on purpose.
                for x64, x32 in ((v64.real, v32.real), (v64.imag, v32.imag)):
                    with np.errstate(over="ignore"):
                        through = float(np.float32(x64))
                    if not (math.isnan(x64) or through == x32):
                        differing.append((fmt, field, symmetry, i, j, x64, x32, through))
            tables.append((rows, cols, table))
        assert all(t == tables[0] for t in tables), (field, symmetry, "the formats differ")
        rows, cols, table = tables[0]
        name = os.path.join(HERE, "expected", kind_name(field, symmetry) + ".txt")
        with open(name, "w") as f:
            sources = " and ".join("scipy/" + file_name(fmt, field, symmetry) for fmt in formats)
            f.write(f"# {field} {symmetry}, {rows} x {cols}: the matrix scipy.io.mmread reads from\n")
            f.write(f"# {sources} (scipy {scipy.__version__}), by gen_matrix_market.py.\n")
            f.write("# Each nonzero cell: row and column from 0, the bits of the real and the imaginary\n")
            f.write("# part as f64, and the bits of the two parts of the nearest f32 to each number of\n")
            f.write("# the file (a cell named twice summed in f32), all in hexadecimal. The values of a\n")
            f.write("# file that isn't complex have the imaginary part +0.\n")
            f.write(f"{rows} {cols}\n")
            for i, j, re64, im64, re32, im32 in table:
                f.write(f"{i} {j} {re64:016x} {im64:016x} {re32:08x} {im32:08x}\n")
        print(f"{name}: {len(table)} cells")
    print(f"{len(differing)} numbers whose f32 is not the f32 of their f64:")
    for d in differing:
        print("   ", d)


def read_expected(field, symmetry):
    name = os.path.join(HERE, "expected", kind_name(field, symmetry) + ".txt")
    with open(name) as f:
        lines = [line for line in f if not line.startswith("#")]
    rows, cols = (int(t) for t in lines[0].split())
    cells = {}
    for line in lines[1:]:
        i, j, re64, im64, re32, im32 = line.split()
        cells[(int(i), int(j))] = (int(re64, 16), int(im64, 16), int(re32, 16), int(im32, 16))
    return rows, cols, cells


def check(directory):
    failures, files, compared, through_f64 = 0, 0, 0, 0
    for precision in ("f64", "f32"):
        for field, symmetry, formats in KINDS:
            rows, cols, expected = read_expected(field, symmetry)
            for fmt in formats:
                path = os.path.join(directory, precision, file_name(fmt, field, symmetry))
                with open(path) as f:
                    text = f.read()
                files += 1
                shape, read = read_by_scipy(path)
                if shape != (rows, cols):
                    print("FAIL shape", path, shape)
                    failures += 1
                    continue
                if field == "integer":
                    read = {k: float(v) for k, v in read.items()}
                # The f32 of the digits of the file, rounded exactly.
                _, _, _, _, c32 = cells_of_file(text)
                for key in sorted(set(read) | set(expected) | set(c32)):
                    compared += 1
                    want = expected.get(key)
                    if field == "pattern" and want is not None:
                        # A pattern holds no values: every cell it names is a 1.
                        want = (f64_bits(1.0), 0, f32_bits_of_value(1.0), 0)
                    if precision == "f64":
                        v = read.get(key, 0)
                        # scipy reads the -0 of an array file as +0: there, the sign of a
                        # zero part is not compared.
                        signed = fmt == "coordinate"
                        ok = v == 0 if want is None else (
                            parts64(v, signed) == parts64(complex_of_bits(*want[:2]), signed))
                    else:
                        v = complex(c32.get(key, 0))
                        zero32 = want is not None and want[2] & 0x7FFFFFFF == 0 \
                            and want[3] & 0x7FFFFFFF == 0
                        if want is None or zero32:
                            ok = v == 0 and read.get(key, 0) == 0
                        else:
                            ok = (f32_bits_of_value(v.real), f32_bits_of_value(v.imag)) == want[2:]
                            # numpy's float32 of scipy's float64, which may round twice.
                            s = complex(read.get(key, 0))
                            with np.errstate(over="ignore"):
                                cast = (f32_bits_of_value(float(np.float32(s.real))),
                                        f32_bits_of_value(float(np.float32(s.imag))))
                            if cast != want[2:]:
                                through_f64 += 1
                    if not ok:
                        failures += 1
                        print("FAIL", path, key, v, want)
    print(f"{files} files, {compared} cells compared, {failures} failures; {through_f64} cells "
          "of the f32 files where numpy's float32 of scipy's float64 is not the f32 written")
    return failures == 0


# The four banners the definition of the format leaves out, as the test of io.rs gives them.
LENIENT = [
    "%%MatrixMarket matrix coordinate real hermitian\n2 2 2\n1 1 1.0\n2 1 3.0\n",
    "%%MatrixMarket matrix coordinate pattern skew-symmetric\n2 2 1\n2 1\n",
    "%%MatrixMarket matrix coordinate pattern hermitian\n2 2 2\n1 1\n2 1\n",
    "%%MatrixMarket matrix coordinate integer hermitian\n2 2 1\n2 1 3\n",
]


def lenient():
    for text in LENIENT:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = scipy.io.mmread(io.StringIO(text))
        print(text.splitlines()[0], "->", a.toarray().tolist())


if __name__ == "__main__":
    print("numpy", np.__version__, "scipy", scipy.__version__)
    if len(sys.argv) >= 2 and sys.argv[1] == "write":
        write()
    elif len(sys.argv) >= 2 and sys.argv[1] == "lenient":
        lenient()
    elif len(sys.argv) >= 2 and sys.argv[1] == "check":
        directory = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ours")
        sys.exit(0 if check(directory) else 1)
    else:
        print(__doc__)
        sys.exit(2)
