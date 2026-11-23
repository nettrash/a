//! Reading and writing Matrix Market files, the exchange format of the NIST Matrix Market
//! (Boisvert, Pozo and Remington, *The Matrix Market exchange formats: initial design*, NISTIR
//! 5935, 1996) and of the SuiteSparse collection.
//!
//! A file is text. Its first line, the banner, is `%%MatrixMarket matrix`, then the format, the
//! field and the symmetry:
//!
//! * the format is `coordinate`, one stored element to a line as `row column value` with indices
//!   from 1, after a size line `rows columns entries`; or `array`, every element on a line of its
//!   own, column after column, after a size line `rows columns`;
//! * the field is `real`, `integer`, `complex` — two numbers to a value, the real and the
//!   imaginary part — or `pattern`, coordinates with no value at all, which stand for ones;
//! * the symmetry is `general`, or `symmetric`, `skew-symmetric` or `hermitian`, of which only the
//!   elements on and below the diagonal are in the file (below it alone for `skew-symmetric`).
//!
//! Lines that begin with `%` are comments. [`read_matrix_market`] and [`parse_matrix_market`]
//! read a file into a [`CooMatrix`], a symmetric one expanded to both triangles;
//! [`write_matrix_market`] and [`to_matrix_market`] write any of the three sparse forms in
//! coordinate format and a dense [`Matrix`] in array format. Whatever is wrong with a file is a
//! [`MatrixMarketError`] with the number of the line: nothing in a file makes the reader panic.

use super::{CooMatrix, CscMatrix, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use core::fmt;
use std::io::{self, BufRead, Write};

/// How a file lays its elements out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// The stored elements with their coordinates.
    Coordinate,
    /// Every element, column after column.
    Array,
}

/// What the values of a Matrix Market file are: the fourth word of its banner.
///
/// # Examples
///
/// ```
/// use a::math::sparse::MatrixMarketField;
///
/// assert_eq!(MatrixMarketField::Pattern.to_string(), "pattern");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatrixMarketField {
    /// Real numbers, one to a value.
    Real,
    /// Integers, one to a value.
    Integer,
    /// Complex numbers: the real and the imaginary part of each value.
    Complex,
    /// No values: the file gives the pattern alone, every stored element standing for 1.
    Pattern,
}

impl MatrixMarketField {
    /// The word of the banner.
    fn keyword(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Integer => "integer",
            Self::Complex => "complex",
            Self::Pattern => "pattern",
        }
    }
}

impl fmt::Display for MatrixMarketField {
    /// Writes the word of the banner: `real`, `integer`, `complex` or `pattern`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.keyword())
    }
}

/// The symmetry a Matrix Market file declares, by which it leaves out the elements above the
/// diagonal: the fifth word of its banner.
///
/// # Examples
///
/// ```
/// use a::math::sparse::MatrixMarketSymmetry;
///
/// assert_eq!(MatrixMarketSymmetry::default(), MatrixMarketSymmetry::General);
/// assert_eq!(MatrixMarketSymmetry::SkewSymmetric.to_string(), "skew-symmetric");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatrixMarketSymmetry {
    /// None: every element is in the file.
    #[default]
    General,
    /// `A = Aᵀ`: the file holds the diagonal and what is below it.
    Symmetric,
    /// `A = −Aᵀ`: the file holds what is below the diagonal, which is zero itself.
    SkewSymmetric,
    /// `A = Aᴴ`: the file holds the diagonal, which is real, and what is below it.
    Hermitian,
}

impl MatrixMarketSymmetry {
    /// The word of the banner.
    fn keyword(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Symmetric => "symmetric",
            Self::SkewSymmetric => "skew-symmetric",
            Self::Hermitian => "hermitian",
        }
    }
}

impl fmt::Display for MatrixMarketSymmetry {
    /// Writes the word of the banner: `general`, `symmetric`, `skew-symmetric` or `hermitian`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.keyword())
    }
}

/// What is wrong with a Matrix Market file that could not be read, or with a matrix that could
/// not be written as one. An error of reading carries the number of the line, from 1, which
/// [`line`](Self::line) gives.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{MatrixMarketError, parse_matrix_market};
///
/// let text = "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1.5\n";
/// let error = parse_matrix_market::<f64>(text).unwrap_err();
/// assert!(matches!(error, MatrixMarketError::IndexOutOfRange { line: 3 }));
/// assert_eq!(error.line(), Some(3));
/// assert_eq!(error.to_string(), "line 3: an index is outside the matrix");
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum MatrixMarketError {
    /// The reader or the writer failed.
    Io(io::Error),
    /// The first line isn't the banner, `%%MatrixMarket` and four words, or there is no first
    /// line.
    BadBanner {
        /// The line.
        line: usize,
    },
    /// A word of the banner is none the format has: the object isn't `matrix`, or the format, the
    /// field or the symmetry isn't one of theirs.
    UnknownKeyword {
        /// The line.
        line: usize,
        /// The word.
        keyword: String,
    },
    /// The banner puts together a format, a field and a symmetry that don't go together: a
    /// pattern in array format.
    UnsupportedCombination {
        /// The line.
        line: usize,
    },
    /// The file is of complex numbers and was to be read into a matrix of real ones.
    ComplexIntoReal {
        /// The line.
        line: usize,
    },
    /// The size line is missing, or isn't the two counts of an array file or the three of a
    /// coordinate file.
    BadSize {
        /// The line.
        line: usize,
    },
    /// The file declares a symmetry and a shape that isn't square.
    NotSquare {
        /// The line.
        line: usize,
    },
    /// The matrix has more elements or rows than this machine can count or hold: a count of the
    /// size line beyond a `usize`, or more elements than an array file can hold.
    TooLarge {
        /// The line.
        line: usize,
    },
    /// A line of the entries hasn't the fields its format and field take: too few, or too many.
    BadEntry {
        /// The line.
        line: usize,
    },
    /// A field that should be a number isn't one; in an `integer` file, one that isn't an
    /// integer, such as `1.5` or `1e3`.
    BadNumber {
        /// The line.
        line: usize,
    },
    /// An index is 0, or beyond the rows or the columns the size line declares.
    IndexOutOfRange {
        /// The line.
        line: usize,
    },
    /// The file holds fewer entries than its size line declares, or more.
    EntryCount {
        /// The line of the first entry too many, or the last line of a file with too few.
        line: usize,
        /// The number of entries the size line declares, or the shape and the symmetry imply.
        expected: usize,
        /// The number of entries the file holds.
        found: usize,
    },
    /// An entry lies above the diagonal in a file that declares a symmetry, which holds the lower
    /// triangle alone.
    AboveDiagonal {
        /// The line.
        line: usize,
    },
    /// A nonzero entry lies on the diagonal of a skew-symmetric file, whose diagonal is zero.
    SkewDiagonal {
        /// The line.
        line: usize,
    },
    /// An entry on the diagonal of a Hermitian file has an imaginary part, where the diagonal of
    /// a Hermitian matrix is real.
    HermitianDiagonal {
        /// The line.
        line: usize,
    },
    /// The matrix to write hasn't the symmetry the options declare: it isn't square, or an
    /// element doesn't mirror the one across the diagonal.
    NotSymmetric {
        /// The symmetry asked for.
        symmetry: MatrixMarketSymmetry,
    },
    /// An element of the matrix to write can't be written in the field the options ask for: one
    /// with an imaginary part in a real or integer field, or one that isn't an integer in an
    /// integer field.
    ValueNotInField {
        /// The field asked for.
        field: MatrixMarketField,
    },
    /// The options ask for what the format doesn't have: a pattern in array format, or a
    /// skew-symmetric pattern.
    UnsupportedOptions,
}

impl MatrixMarketError {
    /// The number of the line of the file the error is at, from 1; `None` for an error of the
    /// reader or the writer itself, and for the errors of writing.
    #[must_use]
    pub fn line(&self) -> Option<usize> {
        match *self {
            Self::BadBanner { line }
            | Self::UnknownKeyword { line, .. }
            | Self::UnsupportedCombination { line }
            | Self::ComplexIntoReal { line }
            | Self::BadSize { line }
            | Self::NotSquare { line }
            | Self::TooLarge { line }
            | Self::BadEntry { line }
            | Self::BadNumber { line }
            | Self::IndexOutOfRange { line }
            | Self::EntryCount { line, .. }
            | Self::AboveDiagonal { line }
            | Self::SkewDiagonal { line }
            | Self::HermitianDiagonal { line } => Some(line),
            Self::Io(_)
            | Self::NotSymmetric { .. }
            | Self::ValueNotInField { .. }
            | Self::UnsupportedOptions => None,
        }
    }
}

impl fmt::Display for MatrixMarketError {
    /// Writes what was wrong, after the number of the line where there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(line) = self.line() {
            write!(f, "line {line}: ")?;
        }
        match self {
            Self::Io(error) => write!(f, "reading or writing failed: {error}"),
            Self::BadBanner { .. } => f.write_str("not a Matrix Market banner"),
            Self::UnknownKeyword { keyword, .. } => write!(f, "unknown keyword `{keyword}`"),
            Self::UnsupportedCombination { .. } => {
                f.write_str("the format, the field and the symmetry don't go together")
            }
            Self::ComplexIntoReal { .. } => {
                f.write_str("a complex file cannot be read into a real matrix")
            }
            Self::BadSize { .. } => f.write_str("the size line is missing or malformed"),
            Self::NotSquare { .. } => f.write_str("a matrix with a symmetry must be square"),
            Self::TooLarge { .. } => f.write_str("the matrix is too large for this machine"),
            Self::BadEntry { .. } => f.write_str("the entry has the wrong number of fields"),
            Self::BadNumber { .. } => f.write_str("not a number"),
            Self::IndexOutOfRange { .. } => f.write_str("an index is outside the matrix"),
            Self::EntryCount {
                expected, found, ..
            } => write!(
                f,
                "the file holds {found} entries where {expected} are declared"
            ),
            Self::AboveDiagonal { .. } => {
                f.write_str("an entry above the diagonal of a file with a symmetry")
            }
            Self::SkewDiagonal { .. } => {
                f.write_str("a nonzero entry on the diagonal of a skew-symmetric file")
            }
            Self::HermitianDiagonal { .. } => {
                f.write_str("an entry on the diagonal of a Hermitian file is not real")
            }
            Self::NotSymmetric { symmetry } => write!(f, "the matrix is not {symmetry}"),
            Self::ValueNotInField { field } => {
                write!(f, "an element cannot be written in the {field} field")
            }
            Self::UnsupportedOptions => {
                f.write_str("the format has no such combination of format, field and symmetry")
            }
        }
    }
}

impl std::error::Error for MatrixMarketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for MatrixMarketError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The lines of a file, read one at a time into one buffer, and counted.
struct Lines<R> {
    /// Where the lines come from.
    reader: R,
    /// The line last read, without its end.
    buffer: Vec<u8>,
    /// The number of lines read so far, which is the number of the line in the buffer.
    line: usize,
}

impl<R: BufRead> Lines<R> {
    /// The lines of `reader`, none read yet.
    fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: Vec::new(),
            line: 0,
        }
    }

    /// Reads the next line into the buffer; `false` at the end of the file.
    fn advance(&mut self) -> Result<bool, MatrixMarketError> {
        self.buffer.clear();
        if self.reader.read_until(b'\n', &mut self.buffer)? == 0 {
            return Ok(false);
        }
        self.line += 1;
        Ok(true)
    }

    /// Reads on to the next line that holds data, one that is neither blank nor a comment;
    /// `false` at the end of the file.
    fn advance_to_data(&mut self) -> Result<bool, MatrixMarketError> {
        while self.advance()? {
            match self.buffer.iter().find(|b| !b.is_ascii_whitespace()) {
                None | Some(b'%') => {}
                Some(_) => return Ok(true),
            }
        }
        Ok(false)
    }

    /// The fields of the line in the buffer: what whitespace of any kind and length separates.
    fn fields(&self) -> impl Iterator<Item = &[u8]> {
        self.buffer
            .split(|b| b.is_ascii_whitespace())
            .filter(|field| !field.is_empty())
    }
}

/// Whether a field is an integer: an optional sign, then decimal digits and nothing else.
fn is_integer(field: &[u8]) -> bool {
    let digits = match field.first() {
        Some(b'+' | b'-') => &field[1..],
        _ => field,
    };
    !digits.is_empty() && digits.iter().all(u8::is_ascii_digit)
}

/// What is wrong with a field that should hold a count or an index.
enum CountError {
    /// It isn't an integer without a minus sign.
    NotACount,
    /// It is one, too large for a `usize`.
    TooLarge,
}

/// The count or index a field holds: an integer without a minus sign that fits a `usize`.
fn parse_count(field: &[u8]) -> Result<usize, CountError> {
    if !is_integer(field) || field[0] == b'-' {
        return Err(CountError::NotACount);
    }
    // Digits alone, which fail to parse only by being too many for the type.
    core::str::from_utf8(field)
        .ok()
        .and_then(|text| text.parse().ok())
        .ok_or(CountError::TooLarge)
}

/// The 0-based index a field holds, which is 1-based in the file and must be at most `bound`.
fn parse_index(field: &[u8], bound: usize, line: usize) -> Result<usize, MatrixMarketError> {
    match parse_count(field) {
        Ok(index) if index >= 1 && index <= bound => Ok(index - 1),
        // An integer all the same, negative or beyond the machine's, is an index out of range;
        // anything else isn't a number.
        Ok(_) | Err(CountError::TooLarge) => Err(MatrixMarketError::IndexOutOfRange { line }),
        Err(CountError::NotACount) if is_integer(field) => {
            Err(MatrixMarketError::IndexOutOfRange { line })
        }
        Err(CountError::NotACount) => Err(MatrixMarketError::BadNumber { line }),
    }
}

/// The real number a field holds, rounded to the type, or `None` if it isn't one. The exponent
/// may be marked by `e` or `E`, as C writes it, or by `d` or `D`, as Fortran writes that of a
/// double precision number.
fn parse_real<R: Float>(field: &[u8]) -> Option<R> {
    let text = core::str::from_utf8(field).ok()?;
    let fortran;
    let text = if text.contains(['d', 'D']) {
        fortran = text.replace(['d', 'D'], "e");
        fortran.as_str()
    } else {
        text
    };
    // Parsed in the type itself: through an `f64`, a decimal that lies next to the midpoint of
    // two `f32`s could be rounded twice, and to the wrong one.
    if R::MANTISSA_DIGITS == f32::MANTISSA_DIGITS {
        text.parse::<f32>().ok().map(|x| R::from_f64(f64::from(x)))
    } else {
        text.parse::<f64>().ok().map(R::from_f64)
    }
}

/// The integer a field of an `integer` file holds, as the nearest number of the type, or `None`
/// if it isn't an integer: an optional sign and decimal digits. A point or an exponent is not in
/// such a file, and rather than be truncated to a whole number, it is an error.
fn parse_integer<R: Float>(field: &[u8]) -> Option<R> {
    if is_integer(field) {
        parse_real(field)
    } else {
        None
    }
}

/// The value of an entry from the fields left on its line: none for a pattern, which is a 1, one
/// for a real number or an integer, and two for a complex number.
fn parse_value<'a, T: Scalar>(
    mut fields: impl Iterator<Item = &'a [u8]>,
    field: MatrixMarketField,
    line: usize,
) -> Result<T, MatrixMarketError> {
    let mut number = |parse: fn(&[u8]) -> Option<T::Real>| {
        let text = fields.next().ok_or(MatrixMarketError::BadEntry { line })?;
        parse(text).ok_or(MatrixMarketError::BadNumber { line })
    };
    let value = match field {
        MatrixMarketField::Pattern => T::one(),
        MatrixMarketField::Real => T::from_real(number(parse_real)?),
        MatrixMarketField::Integer => T::from_real(number(parse_integer)?),
        MatrixMarketField::Complex => {
            let re = number(parse_real)?;
            T::from_parts(re, number(parse_real)?)
        }
    };
    if fields.next().is_some() {
        return Err(MatrixMarketError::BadEntry { line });
    }
    Ok(value)
}

/// The banner of a file.
struct Banner {
    /// Coordinates or an array.
    format: Format,
    /// What the values are.
    field: MatrixMarketField,
    /// The symmetry declared.
    symmetry: MatrixMarketSymmetry,
}

impl Banner {
    /// The banner in the first line, whose words are matched in any case.
    fn parse(first_line: &[u8]) -> Result<Self, MatrixMarketError> {
        let line = 1;
        // A file saved by an editor may begin with a byte order mark.
        let text = first_line
            .strip_prefix(b"\xEF\xBB\xBF")
            .unwrap_or(first_line);
        let words: Vec<&[u8]> = text
            .split(|b| b.is_ascii_whitespace())
            .filter(|word| !word.is_empty())
            .collect();
        if words.len() != 5 || !words[0].eq_ignore_ascii_case(b"%%MatrixMarket") {
            return Err(MatrixMarketError::BadBanner { line });
        }
        let unknown = |word: &[u8]| MatrixMarketError::UnknownKeyword {
            line,
            keyword: String::from_utf8_lossy(word).into_owned(),
        };
        let is = |word: &[u8], keyword: &str| word.eq_ignore_ascii_case(keyword.as_bytes());
        if !is(words[1], "matrix") {
            return Err(unknown(words[1]));
        }
        let format = if is(words[2], "coordinate") {
            Format::Coordinate
        } else if is(words[2], "array") {
            Format::Array
        } else {
            return Err(unknown(words[2]));
        };
        // `double` and `unsigned-integer` are not in the definition of the format, and are what
        // some programs write for `real` and for an `integer` without a sign.
        let field = if is(words[3], "real") || is(words[3], "double") {
            MatrixMarketField::Real
        } else if is(words[3], "integer") || is(words[3], "unsigned-integer") {
            MatrixMarketField::Integer
        } else if is(words[3], "complex") {
            MatrixMarketField::Complex
        } else if is(words[3], "pattern") {
            MatrixMarketField::Pattern
        } else {
            return Err(unknown(words[3]));
        };
        let symmetry = if is(words[4], "general") {
            MatrixMarketSymmetry::General
        } else if is(words[4], "symmetric") {
            MatrixMarketSymmetry::Symmetric
        } else if is(words[4], "skew-symmetric") {
            MatrixMarketSymmetry::SkewSymmetric
        } else if is(words[4], "hermitian") {
            MatrixMarketSymmetry::Hermitian
        } else {
            return Err(unknown(words[4]));
        };
        if format == Format::Array && field == MatrixMarketField::Pattern {
            return Err(MatrixMarketError::UnsupportedCombination { line });
        }
        Ok(Self {
            format,
            field,
            symmetry,
        })
    }
}

/// Adds the element `x` at `(i, j)` of a file with the symmetry `symmetry` to `matrix`, and with
/// it its mirror across the diagonal: itself in a symmetric file, its negation in a
/// skew-symmetric one and its conjugate in a Hermitian one. The value of a file that isn't
/// `complex` is real, and so is its mirror: read into a complex type, both have the imaginary
/// part `+0`, where negating or conjugating the complex number would give `−0`.
fn push_entry<T: Scalar>(
    matrix: &mut CooMatrix<T>,
    symmetry: MatrixMarketSymmetry,
    field: MatrixMarketField,
    (i, j): (usize, usize),
    x: T,
    line: usize,
) -> Result<(), MatrixMarketError> {
    let real = field != MatrixMarketField::Complex;
    if symmetry == MatrixMarketSymmetry::General {
        matrix.push(i, j, x);
        return Ok(());
    }
    if i < j {
        return Err(MatrixMarketError::AboveDiagonal { line });
    }
    if i == j {
        match symmetry {
            // A zero on the diagonal of a skew-symmetric file says nothing, and is no error.
            MatrixMarketSymmetry::SkewSymmetric if x.is_zero() => return Ok(()),
            MatrixMarketSymmetry::SkewSymmetric => {
                return Err(MatrixMarketError::SkewDiagonal { line });
            }
            MatrixMarketSymmetry::Hermitian if x.im() != T::Real::ZERO => {
                return Err(MatrixMarketError::HermitianDiagonal { line });
            }
            _ => {}
        }
        matrix.push(i, i, x);
        return Ok(());
    }
    matrix.push(i, j, x);
    matrix.push(
        j,
        i,
        match symmetry {
            MatrixMarketSymmetry::SkewSymmetric if real => T::from_real(-x.re()),
            MatrixMarketSymmetry::SkewSymmetric => -x,
            MatrixMarketSymmetry::Hermitian if !real => x.conj(),
            _ => x,
        },
    );
    Ok(())
}

/// The number of elements an array file of an `n × n` matrix with a symmetry holds: the lower
/// triangle, with the diagonal or without it; `None` if it overflows.
fn triangle_count(n: usize, with_diagonal: bool) -> Option<usize> {
    if n == 0 {
        return Some(0);
    }
    // n(n + 1)/2 with the diagonal and (n − 1)n/2 without: the product of two consecutive
    // numbers, of which the even one is halved first.
    let (a, b) = if with_diagonal {
        (n, n.checked_add(1)?)
    } else {
        (n.checked_sub(1)?, n)
    };
    if a % 2 == 0 {
        (a / 2).checked_mul(b)
    } else {
        a.checked_mul(b / 2)
    }
}

/// The first `wanted` counts of the size line on the line `line`, two or three of them, the rest
/// zeros. A line without exactly that many fields, or with a field that isn't a count, is
/// [`BadSize`](MatrixMarketError::BadSize); a count too large for a `usize` is
/// [`TooLarge`](MatrixMarketError::TooLarge).
fn parse_size<'a>(
    mut fields: impl Iterator<Item = &'a [u8]>,
    wanted: usize,
    line: usize,
) -> Result<[usize; 3], MatrixMarketError> {
    let mut counts = [0usize; 3];
    let mut too_large = false;
    for count in &mut counts[..wanted] {
        match parse_count(fields.next().ok_or(MatrixMarketError::BadSize { line })?) {
            Ok(value) => *count = value,
            Err(CountError::TooLarge) => too_large = true,
            Err(CountError::NotACount) => return Err(MatrixMarketError::BadSize { line }),
        }
    }
    if fields.next().is_some() {
        Err(MatrixMarketError::BadSize { line })
    } else if too_large {
        Err(MatrixMarketError::TooLarge { line })
    } else {
        Ok(counts)
    }
}

/// The number of triplets room is made for before the entries of a file are seen. The file says
/// how many follow, and it may not say the truth.
const ROOM: usize = 1 << 20;

/// The matrix a file holds, in coordinate form, and the number of its size line.
fn read<T: Scalar>(reader: impl BufRead) -> Result<(CooMatrix<T>, usize), MatrixMarketError> {
    let mut lines = Lines::new(reader);
    if !lines.advance()? {
        return Err(MatrixMarketError::BadBanner { line: 1 });
    }
    let banner = Banner::parse(&lines.buffer)?;
    if banner.field == MatrixMarketField::Complex && !T::IS_COMPLEX {
        return Err(MatrixMarketError::ComplexIntoReal { line: 1 });
    }
    if !lines.advance_to_data()? {
        return Err(MatrixMarketError::BadSize {
            line: lines.line + 1,
        });
    }
    let size_line = lines.line;
    let wanted = if banner.format == Format::Coordinate {
        3
    } else {
        2
    };
    let [rows, cols, entries] = parse_size(lines.fields(), wanted, size_line)?;
    let symmetry = banner.symmetry;
    if symmetry != MatrixMarketSymmetry::General && rows != cols {
        return Err(MatrixMarketError::NotSquare { line: size_line });
    }
    let mut matrix = CooMatrix::with_capacity(rows, cols, entries.min(ROOM));
    match banner.format {
        Format::Coordinate => {
            let mut found = 0;
            while lines.advance_to_data()? {
                let line = lines.line;
                if found == entries {
                    // One entry too many: the rest are counted for the message.
                    found += 1;
                    while lines.advance_to_data()? {
                        found += 1;
                    }
                    return Err(MatrixMarketError::EntryCount {
                        line,
                        expected: entries,
                        found,
                    });
                }
                let mut fields = lines.fields();
                let mut index = |bound| match fields.next() {
                    Some(field) => parse_index(field, bound, line),
                    None => Err(MatrixMarketError::BadEntry { line }),
                };
                let cell = (index(rows)?, index(cols)?);
                let x = parse_value::<T>(fields, banner.field, line)?;
                push_entry(&mut matrix, symmetry, banner.field, cell, x, line)?;
                found += 1;
            }
            if found < entries {
                return Err(MatrixMarketError::EntryCount {
                    line: lines.line,
                    expected: entries,
                    found,
                });
            }
        }
        Format::Array => {
            let skew = symmetry == MatrixMarketSymmetry::SkewSymmetric;
            let expected = match symmetry {
                MatrixMarketSymmetry::General => rows.checked_mul(cols),
                _ => triangle_count(rows, !skew),
            }
            .ok_or(MatrixMarketError::TooLarge { line: size_line })?;
            // The cell of the next entry: down a column, then to the top of the next, which a
            // symmetry puts on the diagonal, or under it.
            let (mut i, mut j) = (usize::from(skew), 0);
            let mut found = 0;
            while lines.advance_to_data()? {
                let line = lines.line;
                if found == expected {
                    found += 1;
                    while lines.advance_to_data()? {
                        found += 1;
                    }
                    return Err(MatrixMarketError::EntryCount {
                        line,
                        expected,
                        found,
                    });
                }
                let x = parse_value::<T>(lines.fields(), banner.field, line)?;
                // An array holds every element, the zeros too, which a sparse matrix has no use
                // for; a zero on the diagonal of a skew-symmetric file isn't in it at all.
                if !x.is_zero() {
                    push_entry(&mut matrix, symmetry, banner.field, (i, j), x, line)?;
                }
                found += 1;
                i += 1;
                if i == rows {
                    j += 1;
                    i = match symmetry {
                        MatrixMarketSymmetry::General => 0,
                        MatrixMarketSymmetry::SkewSymmetric => j + 1,
                        _ => j,
                    };
                }
            }
            if found < expected {
                return Err(MatrixMarketError::EntryCount {
                    line: lines.line,
                    expected,
                    found,
                });
            }
        }
    }
    Ok((matrix, size_line))
}

/// Reads a Matrix Market file from `reader` into a matrix in coordinate form.
///
/// Both formats are read, `coordinate` and `array`, and the four fields: the values of a `real` or
/// `integer` file are rounded to the element type, a `pattern` file gives ones, and a `complex`
/// file wants a complex element type. A file that declares a symmetry is expanded: every element
/// below the diagonal comes with its mirror above it, so that the result is the whole matrix. The
/// triplets are in the order of the file; a cell the file names twice has two triplets, which the
/// conversions sum; and of an `array` file the zeros are left out.
///
/// The words of the banner are read in any case, `%` comment lines and blank lines are skipped
/// wherever they are, fields are separated by any whitespace, and the exponent of a number may be
/// marked by `e`, `E`, `d` or `D`. A number is rounded to the element type once, from its decimal
/// digits, and `inf` and `nan` are numbers too; an `integer` file holds integers alone. Each entry
/// is on a line of its own, as the format has it. The reader should be buffered, as a
/// [`BufReader`](std::io::BufReader) or a slice of bytes is.
///
/// Three banners the definition of the format leaves out are read for what they can only mean:
/// a `hermitian` file of real or integer values is a symmetric one, a `pattern` file may be
/// `hermitian`, which is `symmetric` again, or `skew-symmetric`, whose elements above the diagonal
/// are then −1. A `pattern` file in `array` format means nothing and is an error.
///
/// # Errors
///
/// A [`MatrixMarketError`] for a file that isn't well formed — a bad banner or size line, an
/// entry with the wrong fields, a number that isn't one, an index outside the matrix, more or
/// fewer entries than declared, an entry above the diagonal of a file with a symmetry, a nonzero
/// on the diagonal of a skew-symmetric one — for a complex file read into a real type, and for a
/// failure of the reader.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{CooMatrix, read_matrix_market};
///
/// let file = "\
/// %%MatrixMarket matrix coordinate real symmetric
/// % the lower triangle of a 3 x 3 matrix
/// 3 3 4
/// 1 1 2.0
/// 2 1 -1.0
/// 2 2 2.0
/// 3 3 5e-1
/// ";
/// let a: CooMatrix<f64> = read_matrix_market(file.as_bytes()).unwrap();
/// assert_eq!(a.shape(), (3, 3));
/// assert_eq!(a.nnz(), 5); // the element below the diagonal, and its mirror
/// assert_eq!(a.to_matrix().as_slice(), &[2.0, -1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
/// ```
pub fn read_matrix_market<T: Scalar>(
    reader: impl BufRead,
) -> Result<CooMatrix<T>, MatrixMarketError> {
    read(reader).map(|(matrix, _)| matrix)
}

/// Reads the text of a Matrix Market file into a matrix in coordinate form:
/// [`read_matrix_market`] of a string.
///
/// # Errors
///
/// A [`MatrixMarketError`] for a file that isn't well formed, and for a complex file read into a
/// real type.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::sparse::parse_matrix_market;
///
/// let text = "%%MatrixMarket matrix coordinate complex hermitian\n2 2 2\n1 1 3 0\n2 1 1 -2\n";
/// let a = parse_matrix_market::<Complex<f64>>(text).unwrap().to_csr();
/// assert_eq!(a[(1, 0)], Complex::new(1.0, -2.0));
/// assert_eq!(a[(0, 1)], Complex::new(1.0, 2.0));
/// assert!(a.is_hermitian());
/// ```
pub fn parse_matrix_market<T: Scalar>(text: &str) -> Result<CooMatrix<T>, MatrixMarketError> {
    read_matrix_market(text.as_bytes())
}

impl<T: Scalar> SparseMatrix<T> {
    /// Reads a Matrix Market file from `reader` into a matrix in compressed sparse row form:
    /// [`read_matrix_market`], converted. The entries of a cell the file names more than once are
    /// summed, in the order of the file.
    ///
    /// # Errors
    ///
    /// A [`MatrixMarketError`], as [`read_matrix_market`] gives; and
    /// [`TooLarge`](MatrixMarketError::TooLarge) for a file that declares more rows than can be
    /// held, which is an error here rather than an abort in the conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let file = "%%MatrixMarket matrix coordinate real general\n2 2 2\n2 1 3.5\n1 2 -1\n";
    /// let a = SparseMatrix::<f64>::read_matrix_market(file.as_bytes()).unwrap();
    /// assert_eq!(a, SparseMatrix::from_coo(2, 2, &[(1, 0, 3.5), (0, 1, -1.0)]));
    /// ```
    pub fn read_matrix_market(reader: impl BufRead) -> Result<Self, MatrixMarketError> {
        let (matrix, size_line) = read::<T>(reader)?;
        // A file can declare any number of rows in a few bytes: whether their offsets can be held
        // is found out here, where it is an error, rather than in the conversion, where it would
        // stop the program.
        let too_large = MatrixMarketError::TooLarge { line: size_line };
        let Some(offsets) = matrix.rows().checked_add(1) else {
            return Err(too_large);
        };
        if Vec::<usize>::new().try_reserve_exact(offsets).is_err() {
            return Err(too_large);
        }
        Ok(matrix.to_csr())
    }

    /// Reads the text of a Matrix Market file into a matrix in compressed sparse row form.
    ///
    /// # Errors
    ///
    /// A [`MatrixMarketError`], as [`SparseMatrix::read_matrix_market`] gives.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let text = "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n2 1\n3 3\n";
    /// let a = SparseMatrix::<f32>::from_matrix_market(text).unwrap();
    /// assert_eq!(a, SparseMatrix::from_coo(3, 3, &[(0, 1, 1.0), (1, 0, 1.0), (2, 2, 1.0)]));
    /// ```
    pub fn from_matrix_market(text: &str) -> Result<Self, MatrixMarketError> {
        Self::read_matrix_market(text.as_bytes())
    }
}

/// How a matrix is written as a Matrix Market file: the symmetry to declare, the field of the
/// values, and a comment.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{
///     MatrixMarketOptions, MatrixMarketSymmetry, SparseMatrix, to_matrix_market_with,
/// };
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 2], &[2.0; 3], &[-1.0; 2]);
/// let options = MatrixMarketOptions::new()
///     .with_symmetry(MatrixMarketSymmetry::Symmetric)
///     .with_comment("the second difference");
/// let text = to_matrix_market_with(&a, &options).unwrap();
/// assert_eq!(
///     text,
///     "%%MatrixMarket matrix coordinate real symmetric\n\
///      %the second difference\n\
///      3 3 5\n1 1 2e0\n2 1 -1e0\n2 2 2e0\n3 2 -1e0\n3 3 2e0\n"
/// );
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatrixMarketOptions {
    /// The symmetry declared in the banner, by which only the lower triangle is written:
    /// [`General`](MatrixMarketSymmetry::General) by default, which writes every element.
    pub symmetry: MatrixMarketSymmetry,
    /// The field of the values; `None`, the default, is [`Real`](MatrixMarketField::Real) for a
    /// matrix of real numbers and [`Complex`](MatrixMarketField::Complex) for one of complex
    /// numbers.
    pub field: Option<MatrixMarketField>,
    /// A comment, written after the banner, each of its lines after a `%`; none when empty.
    pub comment: String,
}

impl MatrixMarketOptions {
    /// The defaults: no symmetry, the field of the element type, and no comment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `self`, declaring the symmetry `symmetry`: only the lower triangle of the matrix is
    /// written, and the matrix must have the symmetry, or writing it is an error.
    ///
    /// [`Hermitian`](MatrixMarketSymmetry::Hermitian) with a field that isn't complex is written
    /// as `symmetric`, which is what a Hermitian matrix of real numbers is, the format keeping
    /// `hermitian` for complex values.
    #[must_use]
    pub fn with_symmetry(mut self, symmetry: MatrixMarketSymmetry) -> Self {
        self.symmetry = symmetry;
        self
    }

    /// `self`, writing the values in the field `field`. An [`Integer`](MatrixMarketField::Integer)
    /// field takes a matrix whose elements are integers, a [`Real`](MatrixMarketField::Real) one
    /// a matrix without imaginary parts, and a [`Pattern`](MatrixMarketField::Pattern) writes no
    /// values at all.
    #[must_use]
    pub fn with_field(mut self, field: MatrixMarketField) -> Self {
        self.field = Some(field);
        self
    }

    /// `self`, with the comment `comment`, which may have several lines.
    #[must_use]
    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comment = comment.to_owned();
        self
    }
}

/// Seals [`MatrixMarketSource`].
mod sealed {
    use super::{CooMatrix, CscMatrix, Matrix, SparseMatrix};

    /// The four types that can be written, borrowed.
    pub enum View<'a, T> {
        /// By rows.
        Csr(&'a SparseMatrix<T>),
        /// By columns.
        Csc(&'a CscMatrix<T>),
        /// As triplets.
        Coo(&'a CooMatrix<T>),
        /// Every element.
        Dense(&'a Matrix<T>),
    }

    /// A matrix that says which of the four it is.
    pub trait Source<T> {
        /// The matrix as one of the four.
        fn view(&self) -> View<'_, T>;
    }
}

use sealed::View;

/// A matrix that can be written as a Matrix Market file: [`SparseMatrix`], [`CscMatrix`] and
/// [`CooMatrix`], in coordinate format, and the dense [`Matrix`], in array format. The trait is
/// sealed.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{SparseMatrix, to_matrix_market};
///
/// let dense = Matrix::from_rows(&[[1.0, 0.0], [0.0, 2.0]]);
/// assert!(to_matrix_market(&dense).starts_with("%%MatrixMarket matrix array real general\n"));
/// let sparse = SparseMatrix::from_matrix(&dense);
/// assert!(to_matrix_market(&sparse).starts_with("%%MatrixMarket matrix coordinate real general\n"));
/// ```
pub trait MatrixMarketSource<T: Scalar>: sealed::Source<T> {}

impl<T: Scalar> sealed::Source<T> for SparseMatrix<T> {
    fn view(&self) -> View<'_, T> {
        View::Csr(self)
    }
}

impl<T: Scalar> sealed::Source<T> for CscMatrix<T> {
    fn view(&self) -> View<'_, T> {
        View::Csc(self)
    }
}

impl<T: Scalar> sealed::Source<T> for CooMatrix<T> {
    fn view(&self) -> View<'_, T> {
        View::Coo(self)
    }
}

impl<T: Scalar> sealed::Source<T> for Matrix<T> {
    fn view(&self) -> View<'_, T> {
        View::Dense(self)
    }
}

impl<T: Scalar> MatrixMarketSource<T> for SparseMatrix<T> {}
impl<T: Scalar> MatrixMarketSource<T> for CscMatrix<T> {}
impl<T: Scalar> MatrixMarketSource<T> for CooMatrix<T> {}
impl<T: Scalar> MatrixMarketSource<T> for Matrix<T> {}

/// Whether the value `x` can be written in the field `field`: anything as a complex number or as
/// a mark of the pattern, a number without an imaginary part as a real one, and such a number
/// that is finite and has no fraction as an integer.
fn fits<T: Scalar>(x: T, field: MatrixMarketField) -> bool {
    let real = x.im() == T::Real::ZERO;
    match field {
        MatrixMarketField::Complex | MatrixMarketField::Pattern => true,
        MatrixMarketField::Real => real,
        // Of an infinity and of a NaN the fraction is a NaN, which is not 0.
        MatrixMarketField::Integer => real && x.re().to_f64().fract() == 0.0,
    }
}

/// Checks that every one of `values` can be written in the field `field`, before anything is
/// written.
fn check_field<T: Scalar>(values: &[T], field: MatrixMarketField) -> Result<(), MatrixMarketError> {
    if values.iter().all(|&x| fits(x, field)) {
        Ok(())
    } else {
        Err(MatrixMarketError::ValueNotInField { field })
    }
}

/// Writes the value `x`, which fits the field `field`, after a space when `spaced`: nothing for
/// a pattern, and otherwise the shortest digits that read back as the same number.
fn write_value<T: Scalar>(
    writer: &mut impl Write,
    x: T,
    field: MatrixMarketField,
    spaced: bool,
) -> io::Result<()> {
    let space = if spaced { " " } else { "" };
    match field {
        MatrixMarketField::Pattern => Ok(()),
        MatrixMarketField::Complex => write!(writer, "{space}{:e} {:e}", x.re(), x.im()),
        MatrixMarketField::Real => write!(writer, "{space}{:e}", x.re()),
        // All the digits of an integer, with no exponent and no point.
        MatrixMarketField::Integer => write!(writer, "{space}{:.0}", x.re()),
    }
}

/// Writes the banner, the comment and the beginning of the size line: everything up to the
/// number of entries of a coordinate file.
fn write_head(
    writer: &mut impl Write,
    format: &str,
    field: MatrixMarketField,
    symmetry: MatrixMarketSymmetry,
    comment: &str,
    (rows, cols): (usize, usize),
) -> Result<(), MatrixMarketError> {
    writeln!(writer, "%%MatrixMarket matrix {format} {field} {symmetry}")?;
    // A line of the comment that held the end of a line of any kind would end the comment early.
    for line in comment.lines().flat_map(|line| line.split('\r')) {
        writeln!(writer, "%{line}")?;
    }
    write!(writer, "{rows} {cols}")?;
    Ok(())
}

/// The symmetry the banner declares for the symmetry asked for: `hermitian` is the format's word
/// for complex values alone, and a Hermitian matrix of anything else is `symmetric`.
fn declared(symmetry: MatrixMarketSymmetry, field: MatrixMarketField) -> MatrixMarketSymmetry {
    if symmetry == MatrixMarketSymmetry::Hermitian && field != MatrixMarketField::Complex {
        MatrixMarketSymmetry::Symmetric
    } else {
        symmetry
    }
}

/// Writes, in coordinate format, the matrix whose compressed rows are `matrix`, or, when
/// `by_columns`, the matrix whose compressed columns they are.
fn write_coordinate<T: Scalar>(
    writer: &mut impl Write,
    matrix: &SparseMatrix<T>,
    by_columns: bool,
    field: MatrixMarketField,
    options: &MatrixMarketOptions,
) -> Result<(), MatrixMarketError> {
    let pattern = field == MatrixMarketField::Pattern;
    let symmetry = options.symmetry;
    // A matrix and its transpose have each of the symmetries together, so the rows are asked
    // whichever of the two they are of.
    let symmetric = match symmetry {
        MatrixMarketSymmetry::General => true,
        MatrixMarketSymmetry::SkewSymmetric if pattern => {
            return Err(MatrixMarketError::UnsupportedOptions);
        }
        MatrixMarketSymmetry::Symmetric | MatrixMarketSymmetry::Hermitian if pattern => {
            matrix.mirrored(|_, _| true)
        }
        MatrixMarketSymmetry::Symmetric => matrix.mirrored(|x, y| x == y),
        MatrixMarketSymmetry::SkewSymmetric => matrix.mirrored(|x, y| x == -y),
        MatrixMarketSymmetry::Hermitian => matrix.mirrored(|x, y| x == y.conj()),
    };
    if !symmetric {
        return Err(MatrixMarketError::NotSymmetric { symmetry });
    }
    check_field(matrix.values(), field)?;
    // Whether the element in the row `r` and the column `c` of the compressed rows is written:
    // everything, or the lower triangle of the matrix, which is the upper one of the rows when
    // they are its columns.
    let written = |r: usize, c: usize| match symmetry {
        MatrixMarketSymmetry::General => true,
        MatrixMarketSymmetry::SkewSymmetric => r != c && ((c < r) != by_columns),
        _ => r == c || ((c < r) != by_columns),
    };
    let (row_ptrs, col_indices, values) =
        (matrix.row_pointers(), matrix.col_indices(), matrix.values());
    let rows_of = |r: usize| row_ptrs[r]..row_ptrs[r + 1];
    let entries = if symmetry == MatrixMarketSymmetry::General {
        values.len()
    } else {
        (0..matrix.rows())
            .map(|r| rows_of(r).filter(|&k| written(r, col_indices[k])).count())
            .sum()
    };
    let shape = if by_columns {
        (matrix.cols(), matrix.rows())
    } else {
        matrix.shape()
    };
    write_head(
        writer,
        "coordinate",
        field,
        declared(symmetry, field),
        &options.comment,
        shape,
    )?;
    writeln!(writer, " {entries}")?;
    for r in 0..matrix.rows() {
        for k in rows_of(r) {
            let c = col_indices[k];
            if written(r, c) {
                let (i, j) = if by_columns { (c, r) } else { (r, c) };
                write!(writer, "{} {}", i + 1, j + 1)?;
                write_value(writer, values[k], field, true)?;
                writeln!(writer)?;
            }
        }
    }
    Ok(())
}

/// Writes the dense `matrix` in array format.
fn write_array<T: Scalar>(
    writer: &mut impl Write,
    matrix: &Matrix<T>,
    field: MatrixMarketField,
    options: &MatrixMarketOptions,
) -> Result<(), MatrixMarketError> {
    if field == MatrixMarketField::Pattern {
        return Err(MatrixMarketError::UnsupportedOptions);
    }
    let symmetry = options.symmetry;
    let (rows, cols) = matrix.shape();
    if symmetry != MatrixMarketSymmetry::General {
        // Every element on or below the diagonal against its mirror; the diagonal against
        // itself, which finds a nonzero in a skew-symmetric matrix and an imaginary part in a
        // Hermitian one.
        let mirrors = |x: T, y: T| match symmetry {
            MatrixMarketSymmetry::SkewSymmetric => x == -y,
            MatrixMarketSymmetry::Hermitian => x == y.conj(),
            _ => x == y,
        };
        let symmetric = rows == cols
            && (0..cols).all(|j| (j..rows).all(|i| mirrors(matrix[(i, j)], matrix[(j, i)])));
        if !symmetric {
            return Err(MatrixMarketError::NotSymmetric { symmetry });
        }
    }
    check_field(matrix.as_slice(), field)?;
    write_head(
        writer,
        "array",
        field,
        declared(symmetry, field),
        &options.comment,
        (rows, cols),
    )?;
    writeln!(writer)?;
    for j in 0..cols {
        let first = match symmetry {
            MatrixMarketSymmetry::General => 0,
            MatrixMarketSymmetry::SkewSymmetric => j + 1,
            _ => j,
        };
        for &x in matrix.column(j).iter().skip(first) {
            write_value(writer, x, field, false)?;
            writeln!(writer)?;
        }
    }
    Ok(())
}

/// Writes `matrix` to `writer` as a Matrix Market file, with the default options: every element,
/// in the field of the element type. [`write_matrix_market_with`] takes options.
///
/// A sparse matrix is written in coordinate format, its stored elements in the order they are
/// held in — by rows from a [`SparseMatrix`], by columns from a [`CscMatrix`] — and a
/// [`CooMatrix`] as the matrix it stands for, the triplets of a cell summed. A dense [`Matrix`]
/// is written in array format. Each number is written with the fewest digits that read back as
/// the same number, so that a file read again gives the matrix bit for bit.
///
/// # Errors
///
/// [`MatrixMarketError::Io`] if the writer fails.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, read_matrix_market, write_matrix_market};
///
/// let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 0.1), (1, 0, -2.5e-7)]);
/// let mut file = Vec::new();
/// write_matrix_market(&a, &mut file).unwrap();
/// assert_eq!(
///     String::from_utf8(file.clone()).unwrap(),
///     "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 3 1e-1\n2 1 -2.5e-7\n"
/// );
/// assert_eq!(read_matrix_market::<f64>(file.as_slice()).unwrap().to_csr(), a);
/// ```
pub fn write_matrix_market<T: Scalar, M: MatrixMarketSource<T> + ?Sized>(
    matrix: &M,
    writer: impl Write,
) -> Result<(), MatrixMarketError> {
    write_matrix_market_with(matrix, writer, &MatrixMarketOptions::new())
}

/// Writes `matrix` to `writer` as a Matrix Market file, as the options say: with a symmetry, of
/// which the lower triangle alone is written, in another field than the element type's, or with
/// a comment. Otherwise as [`write_matrix_market`].
///
/// # Errors
///
/// * [`NotSymmetric`](MatrixMarketError::NotSymmetric) if the matrix hasn't the symmetry the
///   options declare — for a [`Pattern`](MatrixMarketField::Pattern) field, if its pattern
///   hasn't.
/// * [`ValueNotInField`](MatrixMarketError::ValueNotInField) if an element can't be written in
///   the field: an imaginary part in a real or integer field, or a fraction, an infinity or a
///   NaN in an integer field.
/// * [`UnsupportedOptions`](MatrixMarketError::UnsupportedOptions) for a pattern of a dense
///   matrix, and for a skew-symmetric pattern.
/// * [`Io`](MatrixMarketError::Io) if the writer fails.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{
///     MatrixMarketError, MatrixMarketField, MatrixMarketOptions, MatrixMarketSymmetry,
///     SparseMatrix, write_matrix_market_with,
/// };
///
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, 3.0), (1, 1, 7.0)]);
/// let options = MatrixMarketOptions::new()
///     .with_symmetry(MatrixMarketSymmetry::Symmetric)
///     .with_field(MatrixMarketField::Integer);
/// let mut file = Vec::new();
/// write_matrix_market_with(&a, &mut file, &options).unwrap();
/// assert_eq!(
///     String::from_utf8(file).unwrap(),
///     "%%MatrixMarket matrix coordinate integer symmetric\n2 2 2\n2 1 3\n2 2 7\n"
/// );
///
/// // A matrix that isn't symmetric is not written as one.
/// let b = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0)]);
/// let error = write_matrix_market_with(&b, Vec::new(), &options).unwrap_err();
/// assert!(matches!(error, MatrixMarketError::NotSymmetric { .. }));
/// ```
pub fn write_matrix_market_with<T: Scalar, M: MatrixMarketSource<T> + ?Sized>(
    matrix: &M,
    writer: impl Write,
    options: &MatrixMarketOptions,
) -> Result<(), MatrixMarketError> {
    // The entries are many short writes, which a writer of its own would make one by one.
    let mut writer = io::BufWriter::new(writer);
    let field = options.field.unwrap_or(if T::IS_COMPLEX {
        MatrixMarketField::Complex
    } else {
        MatrixMarketField::Real
    });
    match sealed::Source::view(matrix) {
        View::Csr(a) => write_coordinate(&mut writer, a, false, field, options)?,
        View::Csc(a) => write_coordinate(&mut writer, a.transposed(), true, field, options)?,
        View::Coo(a) => write_coordinate(&mut writer, &a.to_csr(), false, field, options)?,
        View::Dense(a) => write_array(&mut writer, a, field, options)?,
    }
    writer.flush()?;
    Ok(())
}

/// The text of `matrix` as a Matrix Market file, with the default options:
/// [`write_matrix_market`] into a string.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{CooMatrix, to_matrix_market};
///
/// let mut a = CooMatrix::new(2, 2);
/// a.push(1, 0, 0.5);
/// assert_eq!(
///     to_matrix_market(&a),
///     "%%MatrixMarket matrix coordinate real general\n2 2 1\n2 1 5e-1\n"
/// );
///
/// let dense = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(
///     to_matrix_market(&dense),
///     "%%MatrixMarket matrix array real general\n2 2\n1e0\n3e0\n2e0\n4e0\n"
/// );
/// ```
#[must_use]
pub fn to_matrix_market<T: Scalar, M: MatrixMarketSource<T> + ?Sized>(matrix: &M) -> String {
    // Nothing the default options ask of a matrix can fail, and neither can writing to memory.
    to_matrix_market_with(matrix, &MatrixMarketOptions::new()).unwrap_or_default()
}

/// The text of `matrix` as a Matrix Market file, as the options say:
/// [`write_matrix_market_with`] into a string.
///
/// # Errors
///
/// A [`MatrixMarketError`] if the matrix can't be written as the options ask, as
/// [`write_matrix_market_with`] gives.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::sparse::{
///     MatrixMarketOptions, MatrixMarketSymmetry, SparseMatrix, to_matrix_market_with,
/// };
///
/// let a = SparseMatrix::from_coo(
///     2,
///     2,
///     &[(0, 1, Complex::new(1.0, 2.0)), (1, 0, Complex::new(1.0, -2.0))],
/// );
/// let options = MatrixMarketOptions::new().with_symmetry(MatrixMarketSymmetry::Hermitian);
/// assert_eq!(
///     to_matrix_market_with(&a, &options).unwrap(),
///     "%%MatrixMarket matrix coordinate complex hermitian\n2 2 1\n2 1 1e0 -2e0\n"
/// );
/// ```
pub fn to_matrix_market_with<T: Scalar, M: MatrixMarketSource<T> + ?Sized>(
    matrix: &M,
    options: &MatrixMarketOptions,
) -> Result<String, MatrixMarketError> {
    let mut bytes = Vec::new();
    write_matrix_market_with(matrix, &mut bytes, options)?;
    // Digits, the words of the banner, and a comment that came in as a string.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
