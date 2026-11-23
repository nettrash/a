//! Matrix Market files. What `scipy.io.mmwrite` writes is read bit for bit, in every format, field
//! and symmetry; what the writer writes `scipy.io.mmread` reads bit for bit (the files of
//! `data/ours`, which `data/gen_matrix_market.py check` reads with scipy); a file read back is the
//! matrix written; and a malformed file is an error that names its line, never a panic.

use super::common::{Num, Rng, assert_csr, each_type, random_sparse, size};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{
    CooMatrix, CscMatrix, MatrixMarketError, MatrixMarketField, MatrixMarketOptions,
    MatrixMarketSymmetry, SparseMatrix, parse_matrix_market, read_matrix_market, to_matrix_market,
    to_matrix_market_with, write_matrix_market, write_matrix_market_with,
};
use std::io::{self, BufReader, Read, Write};

/// One sample of `data/`: a format, a field and a symmetry; the file `scipy.io.mmwrite` wrote; the
/// matrix `scipy.io.mmread` reads from it, which `data/gen_matrix_market.py write` recorded; and
/// the files this crate writes of that matrix from the `f64` types and from the `f32` ones, which
/// `data/gen_matrix_market.py check` reads with scipy.
struct Sample {
    /// `<format>_<field>_<symmetry>`.
    name: &'static str,
    /// `coordinate` or `array`.
    format: &'static str,
    field: MatrixMarketField,
    symmetry: MatrixMarketSymmetry,
    scipy: &'static str,
    expected: &'static str,
    ours64: &'static str,
    ours32: &'static str,
}

macro_rules! sample {
    ($format:literal, $field:literal, $symmetry:literal, $f:ident, $s:ident) => {
        Sample {
            name: concat!($format, "_", $field, "_", $symmetry),
            format: $format,
            field: MatrixMarketField::$f,
            symmetry: MatrixMarketSymmetry::$s,
            scipy: include_str!(concat!(
                "data/scipy/",
                $format,
                "_",
                $field,
                "_",
                $symmetry,
                ".mtx"
            )),
            expected: include_str!(concat!("data/expected/", $field, "_", $symmetry, ".txt")),
            ours64: include_str!(concat!(
                "data/ours/f64/",
                $format,
                "_",
                $field,
                "_",
                $symmetry,
                ".mtx"
            )),
            ours32: include_str!(concat!(
                "data/ours/f32/",
                $format,
                "_",
                $field,
                "_",
                $symmetry,
                ".mtx"
            )),
        }
    };
}

/// Every combination of format, field and symmetry the definition of the format allows.
const SAMPLES: [Sample; 22] = [
    sample!("coordinate", "real", "general", Real, General),
    sample!("coordinate", "real", "symmetric", Real, Symmetric),
    sample!("coordinate", "real", "skew-symmetric", Real, SkewSymmetric),
    sample!("coordinate", "integer", "general", Integer, General),
    sample!("coordinate", "integer", "symmetric", Integer, Symmetric),
    sample!(
        "coordinate",
        "integer",
        "skew-symmetric",
        Integer,
        SkewSymmetric
    ),
    sample!("coordinate", "complex", "general", Complex, General),
    sample!("coordinate", "complex", "symmetric", Complex, Symmetric),
    sample!(
        "coordinate",
        "complex",
        "skew-symmetric",
        Complex,
        SkewSymmetric
    ),
    sample!("coordinate", "complex", "hermitian", Complex, Hermitian),
    sample!("coordinate", "pattern", "general", Pattern, General),
    sample!("coordinate", "pattern", "symmetric", Pattern, Symmetric),
    sample!("array", "real", "general", Real, General),
    sample!("array", "real", "symmetric", Real, Symmetric),
    sample!("array", "real", "skew-symmetric", Real, SkewSymmetric),
    sample!("array", "integer", "general", Integer, General),
    sample!("array", "integer", "symmetric", Integer, Symmetric),
    sample!("array", "integer", "skew-symmetric", Integer, SkewSymmetric),
    sample!("array", "complex", "general", Complex, General),
    sample!("array", "complex", "symmetric", Complex, Symmetric),
    sample!("array", "complex", "skew-symmetric", Complex, SkewSymmetric),
    sample!("array", "complex", "hermitian", Complex, Hermitian),
];

/// A nonzero cell of an expected matrix: its row and column, and the bits of its two parts as
/// `f64` and as the `f32` nearest to the digits of the file.
struct Cell {
    i: usize,
    j: usize,
    re64: u64,
    im64: u64,
    re32: u32,
    im32: u32,
}

/// The shape and the cells of a file of `data/expected`.
fn expected_cells(text: &str) -> (usize, usize, Vec<Cell>) {
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let size: Vec<usize> = lines
        .next()
        .expect("a size line")
        .split_whitespace()
        .map(|t| t.parse().expect("a count"))
        .collect();
    let cells = lines
        .map(|line| {
            let t: Vec<&str> = line.split_whitespace().collect();
            Cell {
                i: t[0].parse().expect("a row"),
                j: t[1].parse().expect("a column"),
                re64: u64::from_str_radix(t[2], 16).expect("bits"),
                im64: u64::from_str_radix(t[3], 16).expect("bits"),
                re32: u32::from_str_radix(t[4], 16).expect("bits"),
                im32: u32::from_str_radix(t[5], 16).expect("bits"),
            }
        })
        .collect();
    (size[0], size[1], cells)
}

/// Whether the element type is of single precision.
fn single<T: Num>() -> bool {
    T::EPS == f64::from(f32::EPSILON)
}

/// The element of a cell, in the precision of the type.
fn element<T: Num>(c: &Cell) -> T {
    let part = |b64: u64, b32: u32| {
        if single::<T>() {
            f64::from(f32::from_bits(b32))
        } else {
            f64::from_bits(b64)
        }
    };
    T::from_parts(
        T::Real::from_f64(part(c.re64, c.re32)),
        T::Real::from_f64(part(c.im64, c.im32)),
    )
}

/// The expected matrix, in the precision of the type: the cells whose element rounds to zero in
/// it are not stored.
fn expected_matrix<T: Num>(text: &str) -> SparseMatrix<T> {
    let (rows, cols, cells) = expected_cells(text);
    let triplets: Vec<(usize, usize, T)> =
        cells.iter().map(|c| (c.i, c.j, element::<T>(c))).collect();
    SparseMatrix::from_coo(rows, cols, &triplets)
}

/// The bits of the two parts of an element, as `f64`, which holds an `f32` exactly.
fn bits<T: Num>(x: T) -> (u64, u64) {
    (x.re().to_f64().to_bits(), x.im().to_f64().to_bits())
}

/// The two matrices are the same, bit for bit, the signs of zero parts included.
#[track_caller]
fn assert_same<T: Num>(got: &SparseMatrix<T>, want: &SparseMatrix<T>, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: the shapes differ");
    assert_eq!(
        got.row_pointers(),
        want.row_pointers(),
        "{what}: the patterns differ"
    );
    assert_eq!(
        got.col_indices(),
        want.col_indices(),
        "{what}: the patterns differ"
    );
    for (k, (&x, &y)) in got.values().iter().zip(want.values()).enumerate() {
        assert_eq!(bits(x), bits(y), "{what}: element {k} is {x:?}, not {y:?}");
    }
}

/// Whether a file with the field `field` can be read into the type: a complex one wants a
/// complex type.
fn readable<T: Num>(field: MatrixMarketField) -> bool {
    field != MatrixMarketField::Complex || T::IS_COMPLEX
}

fn reads_the_scipy_files_into<T: Num>() {
    for s in &SAMPLES {
        let read = parse_matrix_market::<T>(s.scipy);
        if !readable::<T>(s.field) {
            assert!(
                matches!(read, Err(MatrixMarketError::ComplexIntoReal { line: 1 })),
                "{}: {read:?}",
                s.name
            );
            continue;
        }
        let want = expected_matrix::<T>(s.expected);
        let coo = read.unwrap_or_else(|e| panic!("{}: {e}", s.name));
        let csr = coo.to_csr();
        assert_csr(&csr);
        assert_same(&csr, &want, s.name);
        assert_same(&coo.to_csc().to_csr(), &want, s.name);
        // The triplets are the entries of the file, a mirror after each one off the diagonal,
        // and of an array the nonzeros alone, so never fewer than the stored elements.
        assert!(coo.nnz() >= want.nnz(), "{}", s.name);
        assert_eq!(coo.shape(), want.shape());
        // Straight into compressed rows, from a string and from a reader.
        let direct = SparseMatrix::<T>::from_matrix_market(s.scipy).expect("readable");
        assert_same(&direct, &want, s.name);
        let reader = BufReader::new(s.scipy.as_bytes());
        let buffered = SparseMatrix::<T>::read_matrix_market(reader).expect("readable");
        assert_same(&buffered, &want, s.name);
    }
}

#[test]
fn the_files_scipy_writes_are_read_bit_for_bit_in_every_format_field_and_symmetry() {
    each_type!(reads_the_scipy_files_into);
}

#[test]
fn a_reader_of_f32_rounds_the_digits_once_and_not_through_f64() {
    // 1.0000000596046448 is the shortest decimal of the f64 1 + 2⁻²⁴, a tie between two f32s:
    // through that f64 it rounds to 1, but the decimal itself lies above the tie and is 1 + 2⁻²³.
    // 3.4028235677973366E38, the decimal of the f64 2¹²⁸ − 2¹⁰³, lies below the threshold of the
    // f32 overflow, where the f64 lies on it: the largest f32, not infinity.
    let text = "%%MatrixMarket matrix coordinate real general\n1 2 2\n\
                1 1 1.0000000596046448\n1 2 3.4028235677973366E38\n";
    let a = SparseMatrix::<f32>::from_matrix_market(text).expect("readable");
    assert_eq!(a.values(), [1.0 + f32::EPSILON, f32::MAX]);
    assert_eq!(1.000_000_059_604_644_8_f64 as f32, 1.0);
    assert_eq!(3.402_823_567_797_336_6e38_f64 as f32, f32::INFINITY);
    let c = SparseMatrix::<Complex<f32>>::from_matrix_market(
        "%%MatrixMarket matrix coordinate complex general\n1 1 1\n\
         1 1 3.4028235677973366E38 1.0000000596046448\n",
    )
    .expect("readable");
    assert_eq!(c.values(), [Complex::new(f32::MAX, 1.0 + f32::EPSILON)]);
}

/// The options a sample's file of this crate is written with: its symmetry, the field when it
/// isn't the element type's own, and a comment.
fn options_of(s: &Sample) -> MatrixMarketOptions {
    let options = MatrixMarketOptions::new()
        .with_symmetry(s.symmetry)
        .with_comment("written by a::math::sparse");
    match s.field {
        MatrixMarketField::Integer | MatrixMarketField::Pattern => options.with_field(s.field),
        _ => options,
    }
}

/// Where the files of `data/ours` are written instead of compared: the directory the variable
/// `A_MATRIX_MARKET_OUT` names, when it is set. `data/gen_matrix_market.py check <directory>` then
/// reads them with scipy.
fn out_directory() -> Option<std::path::PathBuf> {
    std::env::var_os("A_MATRIX_MARKET_OUT").map(Into::into)
}

fn writes_what_scipy_reads_from<T: Num>() {
    for s in &SAMPLES {
        // The files are written from the type of the field: real and complex numbers apart.
        if T::IS_COMPLEX != (s.field == MatrixMarketField::Complex) {
            continue;
        }
        let a = expected_matrix::<T>(s.expected);
        let options = options_of(s);
        let text = if s.format == "array" {
            to_matrix_market_with(&a.to_matrix(), &options)
        } else {
            to_matrix_market_with(&a, &options)
        }
        .unwrap_or_else(|e| panic!("{}: {e}", s.name));
        let (golden, precision) = if single::<T>() {
            (s.ours32, "f32")
        } else {
            (s.ours64, "f64")
        };
        match out_directory() {
            Some(directory) => {
                let directory = directory.join(precision);
                std::fs::create_dir_all(&directory).expect("a directory");
                std::fs::write(directory.join(format!("{}.mtx", s.name)), &text).expect("a file");
            }
            None => assert_eq!(text, golden, "{} from {precision}", s.name),
        }
        // Read back, the matrix written, bit for bit; a pattern as its pattern of ones.
        let back = SparseMatrix::<T>::from_matrix_market(&text).expect("readable");
        let want = if s.field == MatrixMarketField::Pattern {
            a.map(|_, _, _| T::one())
        } else {
            a
        };
        assert_same(&back, &want, s.name);
    }
}

#[test]
fn the_files_written_are_those_scipy_reads_bit_for_bit() {
    each_type!(writes_what_scipy_reads_from);
}

/// The lines of entries of a file: the lines after the size line that hold data.
fn entry_lines(text: &str) -> usize {
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('%'))
        .count()
        - 1
}

fn round_trips_of<T: Num>() {
    let mut r = Rng(0x5EED_2240);
    for case in 0..size(60, 6) {
        let (rows, cols) = (r.below(8), r.below(8));
        let a: SparseMatrix<T> = if rows * cols == 0 {
            SparseMatrix::zeros(rows, cols)
        } else {
            let count = r.below(rows * cols + 1);
            random_sparse(&mut r, rows, cols, count)
        };
        // Every form, written with the defaults and read back: the matrix, bit for bit.
        let texts = [
            to_matrix_market(&a),
            to_matrix_market(&a.to_csc()),
            to_matrix_market(&a.to_coo()),
            to_matrix_market(&a.to_matrix()),
        ];
        for text in &texts {
            let back = SparseMatrix::<T>::from_matrix_market(text).expect("readable");
            assert_same(&back, &a, &format!("case {case}"));
        }
        // The compressed rows and the triplets write the same file, the columns another order.
        assert_eq!(texts[0], texts[2]);
        assert_eq!(entry_lines(&texts[0]), a.nnz());
        assert_eq!(entry_lines(&texts[1]), a.nnz());
        assert_eq!(entry_lines(&texts[3]), rows * cols);
        // Through a writer, the same text.
        let mut bytes = Vec::new();
        write_matrix_market(&a, &mut bytes).expect("written");
        assert_eq!(String::from_utf8(bytes).expect("text"), texts[0]);
        if rows != cols {
            continue;
        }
        // A symmetric, a skew-symmetric and a Hermitian matrix, of which the lower triangle alone
        // is written, by every form, and read back whole.
        let kinds = [
            (&a + &a.transpose(), MatrixMarketSymmetry::Symmetric),
            (&a - &a.transpose(), MatrixMarketSymmetry::SkewSymmetric),
            (
                &a + &a.conjugate_transpose(),
                MatrixMarketSymmetry::Hermitian,
            ),
        ];
        for (m, symmetry) in kinds {
            let options = MatrixMarketOptions::new().with_symmetry(symmetry);
            let lower = if symmetry == MatrixMarketSymmetry::SkewSymmetric {
                m.lower_triangle().nnz() - m.diagonal().iter().filter(|x| !x.is_zero()).count()
            } else {
                m.lower_triangle().nnz()
            };
            let texts = [
                to_matrix_market_with(&m, &options),
                to_matrix_market_with(&m.to_csc(), &options),
                to_matrix_market_with(&m.to_coo(), &options),
                to_matrix_market_with(&m.to_matrix(), &options),
            ];
            for (k, text) in texts.iter().enumerate() {
                let text = text
                    .as_ref()
                    .unwrap_or_else(|e| panic!("case {case}, {symmetry}, form {k}: {e}"));
                if k < 3 {
                    assert_eq!(entry_lines(text), lower, "case {case}, {symmetry}");
                }
                let back = SparseMatrix::<T>::from_matrix_market(text).expect("readable");
                assert_same(&back, &m, &format!("case {case}, {symmetry}, form {k}"));
            }
        }
    }
}

#[test]
fn a_matrix_written_and_read_back_is_the_same_bit_for_bit_in_every_form() {
    each_type!(round_trips_of);
}

#[test]
fn the_numbers_at_the_edges_of_a_type_are_written_and_read_back_unchanged() {
    let numbers = [
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.1,
        1.0 / 3.0,
        -2.5e-7,
        123_456_789.0,
        1e300,
        -0.0,
    ];
    let dense = Matrix::from_column_major(1, numbers.len(), numbers.to_vec());
    for text in [
        to_matrix_market(&dense),
        to_matrix_market(&SparseMatrix::from_matrix(&dense)),
    ] {
        let back = read_matrix_market::<f64>(text.as_bytes())
            .expect("readable")
            .to_matrix();
        for (x, y) in back.as_slice().iter().zip(&numbers) {
            // A zero is not stored, and comes back as one, of either sign.
            assert!(
                x.to_bits() == y.to_bits() || (*x == 0.0 && *y == 0.0),
                "{x} for {y}"
            );
        }
    }
    let singles = [
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::INFINITY,
        0.1,
        1.0 / 3.0,
        16_777_216.0,
    ];
    let a = SparseMatrix::from_matrix(&Matrix::from_column_major(1, 7, singles.to_vec()));
    let text = to_matrix_market(&a);
    // The shortest digits of each f32, not of the f64 it widens to.
    assert!(
        text.contains(" 1e-1\n") && text.contains(" 3.3333334e-1\n"),
        "{text}"
    );
    let back = SparseMatrix::<f32>::from_matrix_market(&text).expect("readable");
    assert_eq!(back.values(), singles);
    // A NaN is written as one and read back as one.
    let n = SparseMatrix::from_coo(1, 2, &[(0, 0, f64::NAN), (0, 1, 1.0)]);
    let back = SparseMatrix::<f64>::from_matrix_market(&to_matrix_market(&n)).expect("readable");
    assert!(back.values()[0].is_nan() && back.values()[1] == 1.0);
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(f32::NAN, -0.0))]);
    let back =
        SparseMatrix::<Complex<f32>>::from_matrix_market(&to_matrix_market(&c)).expect("readable");
    assert!(back.values()[0].re.is_nan());
    assert_eq!(back.values()[0].im.to_bits(), (-0.0f32).to_bits());
}

#[test]
fn the_writer_lays_out_each_form_as_it_is_held() {
    let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
    let banner = "%%MatrixMarket matrix coordinate real general\n";
    // By rows from compressed rows and from triplets, by columns from compressed columns.
    assert_eq!(
        to_matrix_market(&a),
        format!("{banner}2 3 3\n1 3 1.5e0\n2 1 -2e0\n2 3 4e0\n")
    );
    assert_eq!(
        to_matrix_market(&a.to_csc()),
        format!("{banner}2 3 3\n2 1 -2e0\n1 3 1.5e0\n2 3 4e0\n")
    );
    // Triplets of one cell are summed, and a zero is not written.
    let mut coo = CooMatrix::new(2, 3);
    coo.push(1, 2, 3.0);
    coo.push(0, 2, 1.5);
    coo.push(1, 0, -2.0);
    coo.push(1, 2, 1.0);
    coo.push(0, 0, 0.0);
    assert_eq!(to_matrix_market(&coo), to_matrix_market(&a));
    // The dense form: every element, column after column.
    assert_eq!(
        to_matrix_market(&a.to_matrix()),
        "%%MatrixMarket matrix array real general\n2 3\n0e0\n-2e0\n0e0\n0e0\n1.5e0\n4e0\n"
    );
    // Integers in full, without a point or an exponent.
    let big = SparseMatrix::from_coo(1, 2, &[(0, 0, 1e20), (0, 1, -3.0)]);
    let integers = MatrixMarketOptions::new().with_field(MatrixMarketField::Integer);
    assert_eq!(
        to_matrix_market_with(&big, &integers).expect("integers"),
        "%%MatrixMarket matrix coordinate integer general\n1 2 2\n\
         1 1 100000000000000000000\n1 2 -3\n"
    );
    // A pattern: the coordinates alone.
    let pattern = MatrixMarketOptions::new().with_field(MatrixMarketField::Pattern);
    assert_eq!(
        to_matrix_market_with(&a, &pattern).expect("a pattern"),
        "%%MatrixMarket matrix coordinate pattern general\n2 3 3\n1 3\n2 1\n2 3\n"
    );
    // The empty matrices.
    assert_eq!(
        to_matrix_market(&SparseMatrix::<f64>::zeros(0, 0)),
        "%%MatrixMarket matrix coordinate real general\n0 0 0\n"
    );
    assert_eq!(
        to_matrix_market(&Matrix::<f32>::zeros(2, 0)),
        "%%MatrixMarket matrix array real general\n2 0\n"
    );
}

#[test]
fn the_writer_declares_the_symmetry_of_the_lower_triangle_it_writes() {
    let options = |symmetry| MatrixMarketOptions::new().with_symmetry(symmetry);
    // A real Hermitian matrix is a symmetric one: the format keeps `hermitian` for complex values.
    let s = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, 3.0), (1, 1, -1.0)]);
    assert_eq!(
        to_matrix_market_with(&s, &options(MatrixMarketSymmetry::Hermitian)).expect("symmetric"),
        "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n2 1 3e0\n2 2 -1e0\n"
    );
    // A complex Hermitian matrix of real values written in the real field: symmetric as well.
    let h = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 1, Complex::new(3.0, 0.0)),
            (1, 0, Complex::new(3.0, 0.0)),
            (1, 1, Complex::new(-1.0, 0.0)),
        ],
    );
    let real = options(MatrixMarketSymmetry::Hermitian).with_field(MatrixMarketField::Real);
    assert_eq!(
        to_matrix_market_with(&h, &real).expect("real"),
        "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n2 1 3e0\n2 2 -1e0\n"
    );
    // A complex symmetric matrix is not conjugated.
    let c = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 1, Complex::new(1.0, 2.0)),
            (1, 0, Complex::new(1.0, 2.0)),
        ],
    );
    assert_eq!(
        to_matrix_market_with(&c, &options(MatrixMarketSymmetry::Symmetric)).expect("symmetric"),
        "%%MatrixMarket matrix coordinate complex symmetric\n2 2 1\n2 1 1e0 2e0\n"
    );
    // Skew-symmetric: below the diagonal alone, in both layouts.
    let k = SparseMatrix::from_coo(
        3,
        3,
        &[(0, 2, -4.0), (2, 0, 4.0), (1, 2, 0.5), (2, 1, -0.5)],
    );
    let skew = options(MatrixMarketSymmetry::SkewSymmetric);
    assert_eq!(
        to_matrix_market_with(&k, &skew).expect("skew"),
        "%%MatrixMarket matrix coordinate real skew-symmetric\n3 3 2\n3 1 4e0\n3 2 -5e-1\n"
    );
    assert_eq!(
        to_matrix_market_with(&k.to_matrix(), &skew).expect("skew"),
        "%%MatrixMarket matrix array real skew-symmetric\n3 3\n0e0\n4e0\n-5e-1\n"
    );
    // A pattern needs only its pattern to be symmetric.
    let lopsided = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, 7.0)]);
    let pattern = options(MatrixMarketSymmetry::Symmetric).with_field(MatrixMarketField::Pattern);
    assert_eq!(
        to_matrix_market_with(&lopsided, &pattern).expect("a symmetric pattern"),
        "%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n2 1\n"
    );
    // A comment of several lines, each after a `%`, whatever ends them.
    let commented = MatrixMarketOptions::new().with_comment("one\ntwo\r\nthree\rfour");
    assert_eq!(
        to_matrix_market_with(&SparseMatrix::<f64>::zeros(1, 1), &commented).expect("written"),
        "%%MatrixMarket matrix coordinate real general\n%one\n%two\n%three\n%four\n1 1 0\n"
    );
}

#[test]
fn the_writer_refuses_what_the_options_cannot_hold() {
    let options = |symmetry| MatrixMarketOptions::new().with_symmetry(symmetry);
    let not_symmetric = |e: MatrixMarketError| matches!(e, MatrixMarketError::NotSymmetric { .. });
    // Not the symmetry asked for: a lopsided element, a missing mirror, a rectangle, a diagonal
    // that a skew-symmetric matrix can't have and one that a Hermitian matrix can't have.
    let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, 3.5)]);
    for symmetry in [
        MatrixMarketSymmetry::Symmetric,
        MatrixMarketSymmetry::SkewSymmetric,
        MatrixMarketSymmetry::Hermitian,
    ] {
        assert!(not_symmetric(
            to_matrix_market_with(&a, &options(symmetry)).unwrap_err()
        ));
        assert!(not_symmetric(
            to_matrix_market_with(&a.to_matrix(), &options(symmetry)).unwrap_err()
        ));
        assert!(not_symmetric(
            to_matrix_market_with(&SparseMatrix::<f64>::zeros(2, 3), &options(symmetry))
                .unwrap_err()
        ));
    }
    let half = SparseMatrix::from_coo(2, 2, &[(1, 0, 3.0)]);
    assert!(not_symmetric(
        to_matrix_market_with(&half.to_csc(), &options(MatrixMarketSymmetry::Symmetric))
            .unwrap_err()
    ));
    let diagonal = SparseMatrix::from_coo(2, 2, &[(1, 1, 1.0)]);
    assert!(not_symmetric(
        to_matrix_market_with(&diagonal, &options(MatrixMarketSymmetry::SkewSymmetric))
            .unwrap_err()
    ));
    let complex_diagonal = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(1.0, 1.0))]);
    let error = to_matrix_market_with(&complex_diagonal, &options(MatrixMarketSymmetry::Hermitian))
        .unwrap_err();
    assert!(not_symmetric(error));
    // A NaN mirrors nothing, not even another NaN.
    let nan = SparseMatrix::from_coo(2, 2, &[(0, 1, f64::NAN), (1, 0, f64::NAN)]);
    assert!(not_symmetric(
        to_matrix_market_with(&nan, &options(MatrixMarketSymmetry::Symmetric)).unwrap_err()
    ));
    let error = to_matrix_market_with(&a, &options(MatrixMarketSymmetry::Symmetric)).unwrap_err();
    assert_eq!(error.to_string(), "the matrix is not symmetric");
    assert_eq!(error.line(), None);
    // Values the field can't hold: an imaginary part in a real or integer field, a fraction, an
    // infinity or a NaN in an integer field.
    let not_in_field =
        |e: MatrixMarketError| matches!(e, MatrixMarketError::ValueNotInField { .. });
    let c = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(1.0, 0.5))]);
    for field in [MatrixMarketField::Real, MatrixMarketField::Integer] {
        let options = MatrixMarketOptions::new().with_field(field);
        assert!(not_in_field(
            to_matrix_market_with(&c, &options).unwrap_err()
        ));
    }
    let integers = MatrixMarketOptions::new().with_field(MatrixMarketField::Integer);
    for x in [0.5, f64::INFINITY, f64::NAN, -1e-300] {
        let m = SparseMatrix::from_coo(1, 1, &[(0, 0, x)]);
        let error = to_matrix_market_with(&m, &integers).unwrap_err();
        assert_eq!(
            error.to_string(),
            "an element cannot be written in the integer field"
        );
        assert!(not_in_field(
            to_matrix_market_with(&m.to_matrix(), &integers).unwrap_err()
        ));
    }
    // What the format has no words for: a dense pattern, and a skew-symmetric one.
    let pattern = MatrixMarketOptions::new().with_field(MatrixMarketField::Pattern);
    assert!(matches!(
        to_matrix_market_with(&Matrix::<f64>::identity(2), &pattern).unwrap_err(),
        MatrixMarketError::UnsupportedOptions
    ));
    let k = SparseMatrix::from_coo(2, 2, &[(0, 1, -1.0), (1, 0, 1.0)]);
    assert!(matches!(
        to_matrix_market_with(
            &k,
            &pattern.with_symmetry(MatrixMarketSymmetry::SkewSymmetric)
        )
        .unwrap_err(),
        MatrixMarketError::UnsupportedOptions
    ));
    // A failing writer is an error of its own, which carries the error of the writer.
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("the disk is full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = write_matrix_market(&a, Broken).unwrap_err();
    assert!(matches!(error, MatrixMarketError::Io(_)));
    assert_eq!(
        error.to_string(),
        "reading or writing failed: the disk is full"
    );
    assert!(std::error::Error::source(&error).is_some());
    assert!(write_matrix_market_with(&a.to_matrix(), Broken, &MatrixMarketOptions::new()).is_err());
}

#[test]
fn the_reader_takes_what_the_format_allows_in_any_spelling() {
    let read = |text: &str| {
        parse_matrix_market::<f64>(text)
            .unwrap_or_else(|e| panic!("{e}: {text:?}"))
            .to_csr()
    };
    let want = SparseMatrix::from_coo(2, 3, &[(0, 0, 150.0), (1, 2, -0.5)]);
    for text in [
        // Keywords in any case, a byte order mark, any whitespace, the ends of a Windows file.
        "%%matrixmarket MATRIX Coordinate REAL General\n2 3 2\n1 1 150\n2 3 -0.5\n",
        "\u{FEFF}%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 150\n2 3 -0.5\n",
        "  %%MatrixMarket\tmatrix  coordinate real   general  \n2\t3\t2\n 1 1\t150 \n2 3 -0.5",
        "%%MatrixMarket matrix coordinate real general\r\n2 3 2\r\n1 1 150\r\n2 3 -0.5\r\n",
        // Comments and blank lines anywhere after the banner.
        "%%MatrixMarket matrix coordinate real general\n% a\n\n  % b\n2 3 2\n%\n\n1 1 150\n\n% c\n2 3 -0.5\n%\n",
        // Exponents by e, E, d or D, signs, points without digits on one side.
        "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 1.5D+02\n+2 +3 -5d-1\n",
        "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 15E1\n2 3 -.5\n",
        "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 150.\n2 3 -5e-1\n",
        // The words some programs write for `real` and for an `integer` without a sign.
        "%%MatrixMarket matrix coordinate double general\n2 3 2\n1 1 150\n2 3 -0.5\n",
        // An explicit zero, which is not stored, and a cell given twice, which is summed.
        "%%MatrixMarket matrix coordinate real general\n2 3 4\n1 1 100\n2 2 0\n2 3 -0.5\n1 1 50\n",
    ] {
        assert_eq!(read(text), want, "{text:?}");
    }
    let unsigned = "%%MatrixMarket matrix coordinate unsigned-integer general\n1 1 1\n1 1 5\n";
    assert_eq!(read(unsigned).values(), [5.0]);
    // Infinities and NaNs are numbers.
    let special = read(
        "%%MatrixMarket matrix coordinate real general\n1 4 4\n1 1 inf\n1 2 -Infinity\n1 3 NaN\n1 4 nan\n",
    );
    assert_eq!(special.values()[..2], [f64::INFINITY, f64::NEG_INFINITY]);
    assert!(special.values()[2].is_nan() && special.values()[3].is_nan());
    // The triplets of a symmetric file: each entry, then its mirror; of an array, the nonzeros
    // column after column.
    let coo = parse_matrix_market::<f64>(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n3 3 2\n3 1 4\n2 1 -1\n",
    )
    .expect("readable");
    assert_eq!(
        coo.iter().collect::<Vec<_>>(),
        [(2, 0, 4.0), (0, 2, -4.0), (1, 0, -1.0), (0, 1, 1.0)]
    );
    let coo = parse_matrix_market::<f32>(
        "%%MatrixMarket matrix array real symmetric\n3 3\n1\n0\n2\n3\n0\n6\n",
    )
    .expect("readable");
    assert_eq!(
        coo.iter().collect::<Vec<_>>(),
        [
            (0, 0, 1.0),
            (2, 0, 2.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 2, 6.0)
        ]
    );
    // A zero on the diagonal of a skew-symmetric file says nothing, and is no error.
    let zero = parse_matrix_market::<f64>(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 2\n1 1 0\n2 1 3\n",
    )
    .expect("readable");
    assert_eq!(
        zero.to_csr(),
        SparseMatrix::from_coo(2, 2, &[(1, 0, 3.0), (0, 1, -3.0)])
    );
    // A real file read into a complex type: the imaginary parts are +0, the mirrors' too.
    let complex = parse_matrix_market::<Complex<f64>>(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 3\n",
    )
    .expect("readable");
    let parts: Vec<(u64, u64)> = complex
        .iter()
        .map(|(_, _, x)| (x.re.to_bits(), x.im.to_bits()))
        .collect();
    assert_eq!(
        parts,
        [(3f64.to_bits(), 0), ((-3f64).to_bits(), 0)],
        "{complex:?}"
    );
    // The empty matrices, and a 0 × 0 skew-symmetric array, which holds nothing.
    for text in [
        "%%MatrixMarket matrix coordinate real general\n0 0 0\n",
        "%%MatrixMarket matrix array real skew-symmetric\n0 0\n",
        "%%MatrixMarket matrix array complex hermitian\n0 0\n",
    ] {
        assert_eq!(
            SparseMatrix::<Complex<f32>>::from_matrix_market(text).expect("readable"),
            SparseMatrix::zeros(0, 0),
            "{text:?}"
        );
    }
    assert_eq!(
        SparseMatrix::<f64>::from_matrix_market(
            "%%MatrixMarket matrix array real skew-symmetric\n1 1\n"
        )
        .expect("readable"),
        SparseMatrix::zeros(1, 1)
    );
}

#[test]
fn the_banners_the_format_leaves_out_are_read_as_scipy_reads_them() {
    // As scipy.io.mmread (scipy 1.18) reads them: `data/gen_matrix_market.py lenient` prints its
    // readings of these four texts.
    let read = |text: &str| -> Matrix<f64> {
        SparseMatrix::from_matrix_market(text)
            .unwrap_or_else(|e| panic!("{e}: {text:?}"))
            .to_matrix()
    };
    let real_hermitian =
        "%%MatrixMarket matrix coordinate real hermitian\n2 2 2\n1 1 1.0\n2 1 3.0\n";
    assert_eq!(
        read(real_hermitian),
        Matrix::from_rows(&[[1.0, 3.0], [3.0, 0.0]])
    );
    let pattern_skew = "%%MatrixMarket matrix coordinate pattern skew-symmetric\n2 2 1\n2 1\n";
    assert_eq!(
        read(pattern_skew),
        Matrix::from_rows(&[[0.0, -1.0], [1.0, 0.0]])
    );
    let pattern_hermitian = "%%MatrixMarket matrix coordinate pattern hermitian\n2 2 2\n1 1\n2 1\n";
    assert_eq!(
        read(pattern_hermitian),
        Matrix::from_rows(&[[1.0, 1.0], [1.0, 0.0]])
    );
    let integer_hermitian = "%%MatrixMarket matrix coordinate integer hermitian\n2 2 1\n2 1 3\n";
    assert_eq!(
        read(integer_hermitian),
        Matrix::from_rows(&[[0.0, 3.0], [3.0, 0.0]])
    );
}

/// The error a text gives read into `f64`.
#[track_caller]
fn error_of(text: &str) -> MatrixMarketError {
    match parse_matrix_market::<f64>(text) {
        Ok(a) => panic!("read {a:?} from {text:?}"),
        Err(e) => e,
    }
}

#[test]
fn a_malformed_banner_or_size_line_is_an_error_at_its_line() {
    use MatrixMarketError as E;
    for text in [
        "",
        "\n",
        "%%MatrixMarket matrix coordinate real\n2 2 0\n",
        "%%MatrixMarket matrix coordinate real general extra\n2 2 0\n",
        "% a comment first\n%%MatrixMarket matrix coordinate real general\n2 2 0\n",
        "%MatrixMarket matrix coordinate real general\n2 2 0\n",
        "MatrixMarket matrix coordinate real general\n2 2 0\n",
    ] {
        assert!(
            matches!(error_of(text), E::BadBanner { line: 1 }),
            "{text:?}"
        );
    }
    for (text, word) in [
        ("%%MatrixMarket vector coordinate real general\n", "vector"),
        ("%%MatrixMarket matrix sparse real general\n", "sparse"),
        ("%%MatrixMarket matrix coordinate float general\n", "float"),
        ("%%MatrixMarket matrix coordinate real upper\n", "upper"),
    ] {
        match error_of(text) {
            E::UnknownKeyword { line: 1, keyword } => assert_eq!(keyword, word),
            other => panic!("{other:?} for {text:?}"),
        }
    }
    assert_eq!(
        error_of("%%MatrixMarket vector coordinate real general\n").to_string(),
        "line 1: unknown keyword `vector`"
    );
    assert!(matches!(
        error_of("%%MatrixMarket matrix array pattern general\n2 2\n"),
        E::UnsupportedCombination { line: 1 }
    ));
    assert!(matches!(
        error_of("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 2\n"),
        E::ComplexIntoReal { line: 1 }
    ));
    assert!(matches!(
        parse_matrix_market::<f32>("%%MatrixMarket matrix array complex hermitian\n1 1\n1 0\n"),
        Err(E::ComplexIntoReal { line: 1 })
    ));
    // The size line: missing, too few or too many counts, or something else than counts.
    let banner = "%%MatrixMarket matrix coordinate real general\n";
    assert!(matches!(error_of(banner), E::BadSize { line: 2 }));
    assert!(matches!(
        error_of(&format!("{banner}% nothing but comments\n\n")),
        E::BadSize { line: 4 }
    ));
    for size_line in [
        "2 2", "2 2 1 1", "2 x 1", "-2 2 1", "2 2 1.0", "2 2 +", "2 2 1e1",
    ] {
        let text = format!("{banner}%\n{size_line}\n1 1 1\n");
        assert!(
            matches!(error_of(&text), E::BadSize { line: 3 }),
            "{text:?}"
        );
    }
    assert!(matches!(
        error_of("%%MatrixMarket matrix array real general\n2 2 4\n1\n2\n3\n4\n"),
        E::BadSize { line: 2 }
    ));
    // A count no `usize` holds, and an array more elements than one can count.
    assert!(matches!(
        error_of(&format!("{banner}99999999999999999999999 2 1\n1 1 1\n")),
        E::TooLarge { line: 2 }
    ));
    assert!(matches!(
        error_of(&format!(
            "%%MatrixMarket matrix array real general\n{} 2\n",
            usize::MAX
        )),
        E::TooLarge { line: 2 }
    ));
    assert!(matches!(
        error_of(&format!(
            "%%MatrixMarket matrix array real symmetric\n{0} {0}\n",
            usize::MAX
        )),
        E::TooLarge { line: 2 }
    ));
    assert!(matches!(
        error_of("%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n"),
        E::NotSquare { line: 2 }
    ));
    assert_eq!(
        error_of("%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n").to_string(),
        "line 2: a matrix with a symmetry must be square"
    );
}

#[test]
fn a_malformed_entry_is_an_error_at_its_line() {
    use MatrixMarketError as E;
    let file =
        |head: &str, entry: &str| format!("%%MatrixMarket matrix {head}\n% c\n2 2 1\n{entry}\n");
    let real = |entry: &str| error_of(&file("coordinate real general", entry));
    // The wrong number of fields.
    for entry in ["1 1", "1", "1 1 2.0 3.0"] {
        assert!(matches!(real(entry), E::BadEntry { line: 4 }), "{entry:?}");
    }
    assert!(matches!(
        error_of(&file("coordinate pattern general", "1 1 5.0")),
        E::BadEntry { line: 4 }
    ));
    assert!(matches!(
        parse_matrix_market::<Complex<f64>>(&file("coordinate complex general", "1 1 2.0")),
        Err(E::BadEntry { line: 4 })
    ));
    assert!(matches!(
        error_of("%%MatrixMarket matrix array real general\n1 2\n1 2\n3\n"),
        E::BadEntry { line: 3 }
    ));
    // Fields that are not numbers, or not integers where the file holds integers.
    for entry in [
        "1 1 abc",
        "1 1 0x1p3",
        "1 1 1d",
        "1 1 1.5.5",
        "1.0 1 2",
        "a 1 2",
        "1 1 --1",
    ] {
        assert!(matches!(real(entry), E::BadNumber { line: 4 }), "{entry:?}");
    }
    for entry in ["1 1 1.5", "1 1 1e3", "1 1 1.0", "1 1 inf"] {
        assert!(
            matches!(
                error_of(&file("coordinate integer general", entry)),
                E::BadNumber { line: 4 }
            ),
            "{entry:?}"
        );
    }
    let integers = parse_matrix_market::<f64>(&file("coordinate integer general", "1 1 -0042"))
        .expect("an integer");
    assert_eq!(integers.values(), [-42.0]);
    assert_eq!(real("1 1 abc").to_string(), "line 4: not a number");
    let bytes = b"%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 \xff\n";
    assert!(matches!(
        read_matrix_market::<f64>(&bytes[..]),
        Err(E::BadNumber { line: 3 })
    ));
    // Indices from 1 up to the shape.
    for entry in [
        "0 1 2",
        "3 1 2",
        "1 3 2",
        "-1 1 2",
        "1 -0 2",
        "99999999999999999999999 1 2",
    ] {
        assert!(
            matches!(real(entry), E::IndexOutOfRange { line: 4 }),
            "{entry:?}"
        );
    }
    assert_eq!(
        real("3 1 2").to_string(),
        "line 4: an index is outside the matrix"
    );
    // Entries that contradict the symmetry.
    assert!(matches!(
        error_of(&file("coordinate real symmetric", "1 2 3")),
        E::AboveDiagonal { line: 4 }
    ));
    assert!(matches!(
        error_of(&file("coordinate real skew-symmetric", "1 1 3")),
        E::SkewDiagonal { line: 4 }
    ));
    assert!(matches!(
        parse_matrix_market::<Complex<f64>>(&file("coordinate complex hermitian", "1 1 1 2")),
        Err(E::HermitianDiagonal { line: 4 })
    ));
    assert!(matches!(
        parse_matrix_market::<Complex<f32>>(
            "%%MatrixMarket matrix array complex hermitian\n2 2\n1 0\n2 3\n4 1\n"
        ),
        Err(E::HermitianDiagonal { line: 5 })
    ));
    assert_eq!(
        error_of(&file("coordinate real skew-symmetric", "1 1 3")).to_string(),
        "line 4: a nonzero entry on the diagonal of a skew-symmetric file"
    );
}

#[test]
fn more_or_fewer_entries_than_declared_is_an_error() {
    use MatrixMarketError as E;
    let banner = "%%MatrixMarket matrix coordinate real general\n";
    match error_of(&format!("{banner}2 2 2\n1 1 1\n% the end\n")) {
        E::EntryCount {
            line: 4,
            expected: 2,
            found: 1,
        } => {}
        other => panic!("{other:?}"),
    }
    let too_many = format!("{banner}2 2 1\n1 1 1\n2 2 1\n\n2 1 3\n");
    match error_of(&too_many) {
        E::EntryCount {
            line: 4,
            expected: 1,
            found: 3,
        } => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(
        error_of(&too_many).to_string(),
        "line 4: the file holds 3 entries where 1 are declared"
    );
    let array = "%%MatrixMarket matrix array real general\n2 2\n";
    assert!(matches!(
        error_of(&format!("{array}1\n2\n3\n")),
        E::EntryCount {
            line: 5,
            expected: 4,
            found: 3
        }
    ));
    assert!(matches!(
        error_of(&format!("{array}1\n2\n3\n4\n5\n")),
        E::EntryCount {
            line: 7,
            expected: 4,
            found: 5
        }
    ));
    let skew = "%%MatrixMarket matrix array real skew-symmetric\n3 3\n";
    assert!(matches!(
        error_of(&format!("{skew}1\n2\n3\n4\n")),
        E::EntryCount {
            line: 6,
            expected: 3,
            found: 4
        }
    ));
    // A file may declare far more entries than it holds without room being made for them.
    assert!(matches!(
        error_of(&format!("{banner}2 2 {}\n1 1 1\n", usize::MAX)),
        E::EntryCount { found: 1, .. }
    ));
}

#[test]
fn a_matrix_too_large_to_hold_by_rows_is_an_error_and_not_an_abort() {
    use MatrixMarketError as E;
    // Triplets hold any shape; compressed rows need an offset for every row.
    // Under Miri, which may not survive asking for petabytes, the first alone.
    let shapes = [usize::MAX.to_string(), "1000000000000000".to_owned()];
    for rows in &shapes[..size(2, 1)] {
        let text = format!("%%MatrixMarket matrix coordinate real general\n{rows} 1 1\n1 1 2\n");
        let coo = parse_matrix_market::<f64>(&text).expect("triplets");
        assert_eq!(coo.nnz(), 1);
        assert!(matches!(
            SparseMatrix::<f64>::from_matrix_market(&text),
            Err(E::TooLarge { line: 2 })
        ));
    }
}

#[test]
fn a_failing_reader_is_an_error_that_carries_its_error() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("the disk is gone"))
        }
    }
    let start = "%%MatrixMarket matrix coordinate real general\n2 2 1\n".as_bytes();
    for reader in [
        BufReader::new(Box::new(Broken) as Box<dyn Read>),
        BufReader::new(Box::new(start.chain(Broken)) as Box<dyn Read>),
    ] {
        let error = read_matrix_market::<f64>(reader).unwrap_err();
        assert!(matches!(error, MatrixMarketError::Io(_)), "{error:?}");
        assert_eq!(error.line(), None);
        assert_eq!(
            error.to_string(),
            "reading or writing failed: the disk is gone"
        );
        assert!(std::error::Error::source(&error).is_some());
    }
}

#[test]
fn a_reader_that_gives_one_byte_at_a_time_reads_the_same_matrix() {
    struct Trickle<'a>(&'a [u8]);
    impl Read for Trickle<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            match (self.0.split_first(), buffer.first_mut()) {
                (Some((&byte, rest)), Some(slot)) => {
                    *slot = byte;
                    self.0 = rest;
                    Ok(1)
                }
                _ => Ok(0),
            }
        }
    }
    for s in &SAMPLES[..12] {
        let slowly = read_matrix_market::<Complex<f64>>(BufReader::with_capacity(
            1,
            Trickle(s.scipy.as_bytes()),
        ))
        .expect("readable");
        let at_once = parse_matrix_market::<Complex<f64>>(s.scipy).expect("readable");
        assert!(slowly == at_once, "{}", s.name);
    }
}

#[test]
fn no_file_makes_the_reader_panic() {
    // Samples edited at random — a byte changed, removed or added, the file cut short — and read
    // into every type: each read returns, a matrix or an error. The bytes are those a parser trips
    // on: digits, signs, points, exponents, whitespace, comment marks, and some that are none.
    const BYTES: &[u8] = b"0123456789 \t\n\r%-+.eEdD:x\xff\x00";
    let mut r = Rng(0x5EED_2241);
    let mut errors = 0usize;
    let reads = size(4_000, 40);
    for _ in 0..reads {
        let mut bytes = SAMPLES[r.below(SAMPLES.len())].scipy.as_bytes().to_vec();
        for _ in 0..1 + r.below(4) {
            let at = r.below(bytes.len() + 1);
            match r.below(4) {
                0 if at < bytes.len() => bytes[at] = BYTES[r.below(BYTES.len())],
                1 if at < bytes.len() => {
                    bytes.remove(at);
                }
                2 => bytes.insert(at, BYTES[r.below(BYTES.len())]),
                _ => bytes.truncate(at),
            }
        }
        errors += usize::from(read_matrix_market::<f32>(bytes.as_slice()).is_err());
        let _ = read_matrix_market::<f64>(bytes.as_slice());
        let _ = read_matrix_market::<Complex<f32>>(bytes.as_slice());
        let _ = read_matrix_market::<Complex<f64>>(bytes.as_slice());
        let _ = SparseMatrix::<f64>::read_matrix_market(bytes.as_slice());
    }
    // Most edits break a file, and some leave it readable.
    assert!(
        errors > 0 && errors < reads,
        "{errors} errors in {reads} reads"
    );
    // Every prefix of a file.
    let text = SAMPLES[9].scipy.as_bytes();
    for end in (0..text.len()).step_by(size(1, 23)) {
        let _ = read_matrix_market::<Complex<f64>>(&text[..end]);
    }
}

#[test]
fn a_large_matrix_goes_through_a_file_unchanged() {
    let mut r = Rng(0x5EED_2242);
    let n = size(3_000, 30);
    let a: SparseMatrix<f64> = random_sparse(&mut r, n, n, 10 * n);
    let text = to_matrix_market(&a);
    let back = SparseMatrix::<f64>::from_matrix_market(&text).expect("readable");
    assert_same(&back, &a, "a large matrix");
    let c: SparseMatrix<Complex<f32>> = random_sparse(&mut r, n, n / 2 + 1, 10 * n);
    let mut file = Vec::new();
    write_matrix_market(&c.to_csc(), &mut file).expect("written");
    let back = read_matrix_market::<Complex<f32>>(file.as_slice())
        .expect("readable")
        .to_csr();
    assert_same(&back, &c, "a large complex matrix");
}

#[test]
fn the_fields_and_symmetries_print_the_words_of_the_banner() {
    let fields = [
        (MatrixMarketField::Real, "real"),
        (MatrixMarketField::Integer, "integer"),
        (MatrixMarketField::Complex, "complex"),
        (MatrixMarketField::Pattern, "pattern"),
    ];
    for (field, word) in fields {
        assert_eq!(field.to_string(), word);
    }
    let symmetries = [
        (MatrixMarketSymmetry::General, "general"),
        (MatrixMarketSymmetry::Symmetric, "symmetric"),
        (MatrixMarketSymmetry::SkewSymmetric, "skew-symmetric"),
        (MatrixMarketSymmetry::Hermitian, "hermitian"),
    ];
    for (symmetry, word) in symmetries {
        assert_eq!(symmetry.to_string(), word);
    }
    let options = MatrixMarketOptions::new();
    assert_eq!(options.symmetry, MatrixMarketSymmetry::General);
    assert_eq!(options.field, None);
    assert!(options.comment.is_empty());
    assert_eq!(options, MatrixMarketOptions::default());
    let set = options
        .with_symmetry(MatrixMarketSymmetry::Hermitian)
        .with_field(MatrixMarketField::Complex)
        .with_comment("c");
    assert_eq!(set.symmetry, MatrixMarketSymmetry::Hermitian);
    assert_eq!(set.field, Some(MatrixMarketField::Complex));
    assert_eq!(set.comment, "c");
    // A CscMatrix and a CooMatrix read from their file are what they were.
    let a = SparseMatrix::from_coo(3, 2, &[(0, 1, 2.0f32), (2, 0, -1.0)]);
    let csc: CscMatrix<f32> = parse_matrix_market(&to_matrix_market(&a.to_csc()))
        .expect("readable")
        .to_csc();
    assert_eq!(csc, a.to_csc());
}
