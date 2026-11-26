//! Matrices over `GF(2)`, 64 elements a word, each row packed (as M4RI has them).
//!
//! A [`BitMatrix`] holds its rows one after another, each in `⌈cols / 64⌉` words, the element of
//! column `j` bit `j mod 64` of word `j / 64`; the bits of a row's last word beyond its last
//! column are always zero, so that two matrices are equal, and hash alike, exactly when their
//! elements are.
//!
//! - Products are by M4RM, the method of the Four Russians (Arlazarov, Dinic, Kronrod and
//!   Faradzev, 1970), as Albrecht, Bard and Hart arrange it ("Algorithm 898: Efficient
//!   multiplication of dense matrices over GF(2)", ACM TOMS 37, 2010): the `2^k` sums of `k` rows
//!   of the right operand in a table built in Gray-code order, one addition each, and each row of
//!   the product the sum of one entry of each table, about `n³ / (64·k)` word operations for
//!   `k = ⌊log₂ n⌋ − 1`, at most 8. From 8192 rows and columns, Strassen's method in Winograd's form sits on top,
//!   seven products of halves and fifteen sums, as M4RI's `mzd_mul` does.
//! - Echelon forms, ranks, inverses, solutions and null spaces are by M4RI, Bard's method of the
//!   Four Russians for elimination ("Accelerating cryptanalysis with the Method of Four
//!   Russians", 2006): `k` pivots found at a time in a strip of `k` columns, then the other rows
//!   cleared of those columns by one entry each of a Gray-code table of the pivot rows' sums,
//!   `O(n³ / log n)` word operations; classical Gaussian elimination, a column at a time, below
//!   32 rows.
//! - Transposes are by 64 × 64 blocks, each transposed in registers in six rounds of swaps of
//!   halves (Warren, "Hacker's Delight", 7–3).

mod echelon;
mod mul;
mod transpose;

use super::Matrix;
use crate::math::numbers::Number;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign};

/// A matrix over `GF(2)`: each row packed 64 elements a word, the element of column `j` bit
/// `j mod 64` of word `j / 64`, the bits beyond the last column zero.
///
/// `+` is the sum (exclusive or) and `*` the product, of owned or borrowed operands, as for
/// [`Matrix`]. Two matrices are equal, and hash alike, when they have the same shape and the
/// same elements.
///
/// # Examples
///
/// ```
/// use a::math::matrix::BitMatrix;
///
/// let a = BitMatrix::from_rows(&[[true, true], [false, true]]);
/// // Over GF(2), [[1, 1], [0, 1]] is its own inverse.
/// assert_eq!(&a * &a, BitMatrix::identity(2));
/// assert_eq!(a.inverse(), Some(a.clone()));
/// assert_eq!(&a + &a, BitMatrix::zeros(2, 2));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BitMatrix {
    /// The number of rows.
    rows: usize,
    /// The number of columns.
    cols: usize,
    /// The words of a row: `⌈cols / 64⌉`.
    stride: usize,
    /// The words, row after row; the bits beyond the last column are zero.
    data: Vec<u64>,
}

/// The bits of a row's last word that hold columns, all ones when the columns fill it.
fn last_word_mask(cols: usize) -> u64 {
    match cols % 64 {
        0 => u64::MAX,
        r => (1u64 << r) - 1,
    }
}

impl BitMatrix {
    /// The `rows × cols` zero matrix.
    ///
    /// # Panics
    ///
    /// Panics if its words would overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let z = BitMatrix::zeros(2, 70);
    /// assert_eq!((z.rows(), z.cols(), z.words_per_row()), (2, 70, 2));
    /// assert!(!z.get(1, 69));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> BitMatrix {
        let stride = cols.div_ceil(64);
        BitMatrix {
            rows,
            cols,
            stride,
            data: vec![0; rows.checked_mul(stride).expect("the matrix is too large")],
        }
    }

    /// The `n × n` identity matrix.
    ///
    /// # Panics
    ///
    /// Panics if its words would overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let i = BitMatrix::identity(3);
    /// assert!(i.get(2, 2) && !i.get(2, 1));
    /// assert!(i.det());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn identity(n: usize) -> BitMatrix {
        let mut m = BitMatrix::zeros(n, n);
        for i in 0..n {
            m.data[i * m.stride + i / 64] |= 1 << (i % 64);
        }
        m
    }

    /// The matrix whose element in row `i` and column `j` is `f(i, j)`, computed row after row.
    ///
    /// # Panics
    ///
    /// Panics if its words would overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// // The upper triangle of ones.
    /// let u = BitMatrix::from_fn(4, 4, |i, j| i <= j);
    /// assert!(u.get(0, 3) && !u.get(3, 0));
    /// assert_eq!(u.rank(), 4);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> bool) -> BitMatrix {
        let mut m = BitMatrix::zeros(rows, cols);
        for i in 0..rows {
            let row = &mut m.data[i * m.stride..(i + 1) * m.stride];
            for j in 0..cols {
                row[j / 64] |= u64::from(f(i, j)) << (j % 64);
            }
        }
        m
    }

    /// The matrix with the given rows of `bool`s.
    ///
    /// # Panics
    ///
    /// Panics if the rows are not all of the same length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_rows(&[[true, false, true], [false, true, true]]);
    /// assert_eq!((a.rows(), a.cols()), (2, 3));
    /// assert!(a.get(0, 2) && !a.get(1, 0));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_rows<R: AsRef<[bool]>>(rows: &[R]) -> BitMatrix {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        assert!(
            rows.iter().all(|r| r.as_ref().len() == cols),
            "the rows are not all of the same length"
        );
        BitMatrix::from_fn(rows.len(), cols, |i, j| rows[i].as_ref()[j])
    }

    /// The matrix of `rows × cols` elements given as words: row after row, each in
    /// [`words_per_row`](BitMatrix::words_per_row) words, the element of column `j` bit `j mod 64`
    /// of word `j / 64` of its row.
    ///
    /// # Panics
    ///
    /// Panics if `words` doesn't hold `rows · ⌈cols / 64⌉` words, or if a bit beyond the last
    /// column of a row is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_words(2, 3, vec![0b101, 0b010]);
    /// assert!(a.get(0, 0) && !a.get(0, 1) && a.get(0, 2) && a.get(1, 1));
    /// assert_eq!(a.row_words(1), &[0b010]);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_words(rows: usize, cols: usize, words: Vec<u64>) -> BitMatrix {
        let stride = cols.div_ceil(64);
        assert_eq!(
            Some(words.len()),
            rows.checked_mul(stride),
            "the words are not rows · ⌈cols / 64⌉"
        );
        if stride > 0 {
            let mask = last_word_mask(cols);
            assert!(
                words
                    .chunks_exact(stride)
                    .all(|r| r[stride - 1] & !mask == 0),
                "a bit beyond the last column is set"
            );
        }
        BitMatrix {
            rows,
            cols,
            stride,
            data: words,
        }
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Whether it has as many rows as columns.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The words of a row, `⌈cols / 64⌉`.
    #[must_use]
    pub fn words_per_row(&self) -> usize {
        self.stride
    }

    /// The element in row `i` and column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// assert!(BitMatrix::identity(100).get(99, 99));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> bool {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        self.data[i * self.stride + j / 64] >> (j % 64) & 1 == 1
    }

    /// Sets the element in row `i` and column `j` to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::zeros(2, 2);
    /// a.set(1, 0, true);
    /// assert!(a.get(1, 0));
    /// a.set(1, 0, false);
    /// assert_eq!(a, BitMatrix::zeros(2, 2));
    /// ```
    #[track_caller]
    pub fn set(&mut self, i: usize, j: usize, value: bool) {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        let w = &mut self.data[i * self.stride + j / 64];
        *w = *w & !(1 << (j % 64)) | u64::from(value) << (j % 64);
    }

    /// Flips the element in row `i` and column `j`: adds 1 to it.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::identity(2);
    /// a.flip(0, 0);
    /// a.flip(0, 1);
    /// assert!(!a.get(0, 0) && a.get(0, 1));
    /// ```
    #[track_caller]
    pub fn flip(&mut self, i: usize, j: usize) {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        self.data[i * self.stride + j / 64] ^= 1 << (j % 64);
    }

    /// The words of row `i`: the element of column `j` is bit `j mod 64` of word `j / 64`, and
    /// the bits beyond the last column are zero.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_fn(2, 66, |_, j| j % 2 == 0);
    /// assert_eq!(a.row_words(0), &[0x5555_5555_5555_5555, 0b01]);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn row_words(&self, i: usize) -> &[u64] {
        assert!(i < self.rows, "the row {i} is out of range");
        &self.data[i * self.stride..(i + 1) * self.stride]
    }

    /// Sets row `i` from its words, laid out as [`row_words`](BitMatrix::row_words) gives them.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range, if `words` is not
    /// [`words_per_row`](BitMatrix::words_per_row) long, or if a bit beyond the last column is
    /// set.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::zeros(2, 3);
    /// a.set_row_words(1, &[0b110]);
    /// assert!(a.get(1, 1) && a.get(1, 2) && !a.get(1, 0));
    /// ```
    #[track_caller]
    pub fn set_row_words(&mut self, i: usize, words: &[u64]) {
        assert!(i < self.rows, "the row {i} is out of range");
        assert_eq!(words.len(), self.stride, "the words are not ⌈cols / 64⌉");
        if let Some(&last) = words.last() {
            assert!(
                last & !last_word_mask(self.cols) == 0,
                "a bit beyond the last column is set"
            );
        }
        self.data[i * self.stride..(i + 1) * self.stride].copy_from_slice(words);
    }

    /// All the words, row after row, each row [`words_per_row`](BitMatrix::words_per_row) words
    /// as [`row_words`](BitMatrix::row_words) gives it.
    #[must_use]
    pub fn as_words(&self) -> &[u64] {
        &self.data
    }

    /// Swaps rows `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if either is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::identity(2);
    /// a.swap_rows(0, 1);
    /// assert!(a.get(0, 1) && a.get(1, 0));
    /// ```
    #[track_caller]
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(a < self.rows && b < self.rows, "the row is out of range");
        if a != b {
            let (lo, hi) = (a.min(b), a.max(b));
            let (left, right) = self.data.split_at_mut(hi * self.stride);
            left[lo * self.stride..(lo + 1) * self.stride]
                .swap_with_slice(&mut right[..self.stride]);
        }
    }

    /// Adds row `source` to row `target`, word by word: the row operation of elimination over
    /// `GF(2)`. A row added to itself becomes zero.
    ///
    /// # Panics
    ///
    /// Panics if either is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::identity(2);
    /// a.add_row(1, 0);
    /// assert_eq!(a, BitMatrix::from_rows(&[[true, false], [true, true]]));
    /// ```
    #[track_caller]
    pub fn add_row(&mut self, target: usize, source: usize) {
        assert!(
            target < self.rows && source < self.rows,
            "the row is out of range"
        );
        if target == source {
            self.data[target * self.stride..(target + 1) * self.stride].fill(0);
        } else {
            xor_rows(&mut self.data, self.stride, target, source, 0);
        }
    }

    /// The matrix as one of [`Number`]s, each element 0 or 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{BitMatrix, Matrix};
    /// use a::math::numbers::Number;
    ///
    /// let a = BitMatrix::identity(2).to_matrix();
    /// assert_eq!(a, Matrix::from_rows(&[[1u8, 0], [0, 1]]).map(|&x| Number::from(x)));
    /// ```
    #[must_use]
    pub fn to_matrix(&self) -> Matrix<Number> {
        Matrix::from_fn(self.rows, self.cols, |i, j| {
            Number::from(u8::from(
                self.data[i * self.stride + j / 64] >> (j % 64) & 1 == 1,
            ))
        })
    }

    /// The product `A·B` by Strassen's method in Winograd's form, seven products of halves and
    /// fifteen sums, recursing while all three dimensions are at least `cutoff`, and by M4RM
    /// below: M4RI's `mzd_mul`, whose last argument is the same cutoff. A `cutoff` below 128
    /// counts as 128, so that every half has whole words. `&a * &b` is this with the cutoff
    /// 8192, so that no half is below 4096 rows or columns; a smaller one can pay where the
    /// caches are smaller.
    ///
    /// # Panics
    ///
    /// Panics if the columns of `self` are not as many as the rows of `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_fn(300, 200, |i, j| (i * j) % 7 < 3);
    /// let b = BitMatrix::from_fn(200, 260, |i, j| (i + 2 * j) % 5 == 0);
    /// // One level of halves, the rows and columns past them peeled off, then M4RM.
    /// assert_eq!(a.mul_strassen(&b, 128), &a * &b);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn mul_strassen(&self, rhs: &BitMatrix, cutoff: usize) -> BitMatrix {
        assert!(
            self.cols == rhs.rows,
            "the columns of the left matrix are not as many as the rows of the right one"
        );
        mul::product_from(self, rhs, cutoff.max(128))
    }

    /// The transpose `Aᵀ`, by 64 × 64 blocks transposed in registers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_fn(3, 100, |i, j| j == 2 * i + 1);
    /// let t = a.transpose();
    /// assert_eq!((t.rows(), t.cols()), (100, 3));
    /// assert!(t.get(5, 2) && !t.get(2, 2));
    /// assert_eq!(t.transpose(), a);
    /// ```
    #[must_use]
    pub fn transpose(&self) -> BitMatrix {
        transpose::transpose(self)
    }

    /// The rank, by M4RI elimination to an echelon form (classical elimination below 32 rows).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// // The third row is the sum of the first two.
    /// let a = BitMatrix::from_rows(&[[true, true, false], [false, true, true], [true, false, true]]);
    /// assert_eq!(a.rank(), 2);
    /// assert_eq!(BitMatrix::zeros(3, 5).rank(), 0);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        let mut m = self.clone();
        echelon::echelonize(&mut m, false, self.cols).len()
    }

    /// The determinant, 1 (`true`) exactly when the matrix is invertible; `true` for `0 × 0`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// // det [[1, 1], [1, 1]] = 1 + 1 = 0.
    /// assert!(!BitMatrix::from_fn(2, 2, |_, _| true).det());
    /// assert!(BitMatrix::from_fn(5, 5, |i, j| i >= j).det());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn det(&self) -> bool {
        assert!(self.is_square(), "the matrix is not square");
        self.rank() == self.rows
    }

    /// The inverse, or `None` when the matrix is singular: the reduced echelon form of `[A | I]`
    /// by M4RI, whose right half is then `A⁻¹`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_fn(70, 70, |i, j| i <= j);
    /// let b = a.inverse().unwrap();
    /// assert_eq!(&a * &b, BitMatrix::identity(70));
    /// assert_eq!(BitMatrix::zeros(2, 2).inverse(), None);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn inverse(&self) -> Option<BitMatrix> {
        assert!(self.is_square(), "the matrix is not square");
        echelon::solve(self, &BitMatrix::identity(self.rows), true)
    }

    /// A solution `X` of `A·X = B`, for `A` of any shape, or `None` when there is none: the
    /// reduced echelon form of `[A | B]` by M4RI, its pivots chosen in the columns of `A` alone;
    /// the rows of `X` of the columns of `A` without a pivot are zero, and those of the pivots
    /// are the rows of the right half. The other solutions are `X` plus columns of
    /// [`null_space`](BitMatrix::null_space).
    ///
    /// # Panics
    ///
    /// Panics if `b` doesn't have as many rows as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_rows(&[[true, true, false], [false, true, true]]);
    /// let b = BitMatrix::from_rows(&[[true], [false]]);
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(&a * &x, b);
    /// // x + y = 1 and x + y = 0 have no solution.
    /// let c = BitMatrix::from_rows(&[[true, true], [true, true]]);
    /// assert_eq!(c.solve(&b), None);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn solve(&self, b: &BitMatrix) -> Option<BitMatrix> {
        assert_eq!(
            self.rows, b.rows,
            "the right-hand side doesn't have as many rows as the matrix"
        );
        echelon::solve(self, b, false)
    }

    /// A basis of the null space `{x : A·x = 0}`, as the columns of an `n × (n − rank)` matrix
    /// `N` with `A·N = 0`: for each column `f` of the reduced echelon form without a pivot, the
    /// vector with a 1 in row `f`, the element of column `f` of each pivot row in the row of its
    /// pivot, and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_rows(&[[true, true, false], [false, true, true]]);
    /// let n = a.null_space();
    /// assert_eq!((n.rows(), n.cols()), (3, 1));
    /// assert_eq!(&a * &n, BitMatrix::zeros(2, 1));
    /// assert!(n.get(0, 0) && n.get(1, 0) && n.get(2, 0));
    /// ```
    #[must_use]
    pub fn null_space(&self) -> BitMatrix {
        echelon::null_space(self)
    }

    /// The reduced row echelon form and its pivot columns, ascending: row `i` of the form is
    /// zero before column `pivots[i]`, 1 there, and the only row with a 1 in that column; the
    /// rows from `pivots.len()`, the rank, are zero. Both are unique. By M4RI (classical
    /// elimination below 32 rows).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let a = BitMatrix::from_rows(&[[false, true, true], [false, true, false], [false, false, true]]);
    /// let (r, pivots) = a.reduced_row_echelon_form();
    /// assert_eq!(pivots, [1, 2]);
    /// assert_eq!(r, BitMatrix::from_rows(&[[false, true, false], [false, false, true], [false; 3]]));
    /// ```
    #[must_use]
    pub fn reduced_row_echelon_form(&self) -> (BitMatrix, Vec<usize>) {
        let mut m = self.clone();
        let pivots = m.echelonize();
        (m, pivots)
    }

    /// Brings the matrix to its reduced row echelon form in place and returns the pivot columns,
    /// as [`reduced_row_echelon_form`](BitMatrix::reduced_row_echelon_form) does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// let mut a = BitMatrix::from_fn(3, 3, |_, _| true);
    /// assert_eq!(a.echelonize(), [0]);
    /// assert_eq!(a, BitMatrix::from_fn(3, 3, |i, _| i == 0));
    /// ```
    pub fn echelonize(&mut self) -> Vec<usize> {
        let cols = self.cols;
        echelon::echelonize(self, true, cols)
    }

    /// The words `[w0, w1)` of rows `r0..r1` as a matrix of `cols` columns, for the blocks of
    /// Strassen's method; `cols` is `64·(w1 − w0)`, or less for a block that ends at the last
    /// column, whose padding is already zero.
    fn block(&self, r0: usize, r1: usize, w0: usize, w1: usize, cols: usize) -> BitMatrix {
        let mut b = BitMatrix::zeros(r1 - r0, cols);
        debug_assert_eq!(b.stride, w1 - w0);
        for (i, row) in b.data.chunks_exact_mut(b.stride.max(1)).enumerate() {
            let s = (r0 + i) * self.stride;
            row.copy_from_slice(&self.data[s + w0..s + w1]);
        }
        b
    }

    /// Adds `b` to the block of rows from `r0` and words from `w0`.
    fn add_block(&mut self, r0: usize, w0: usize, b: &BitMatrix) {
        if b.stride == 0 {
            return;
        }
        for (i, row) in b.data.chunks_exact(b.stride).enumerate() {
            let s = (r0 + i) * self.stride + w0;
            for (x, y) in self.data[s..s + b.stride].iter_mut().zip(row) {
                *x ^= y;
            }
        }
    }
}

/// Adds row `src` to row `dst` of the words `data`, rows `stride` words apart, from word `from`.
fn xor_rows(data: &mut [u64], stride: usize, dst: usize, src: usize, from: usize) {
    debug_assert_ne!(dst, src);
    let (d, s) = if dst < src {
        let (lo, hi) = data.split_at_mut(src * stride);
        (
            &mut lo[dst * stride + from..(dst + 1) * stride],
            &hi[from..stride],
        )
    } else {
        let (lo, hi) = data.split_at_mut(dst * stride);
        (
            &mut hi[from..stride],
            &lo[src * stride + from..(src + 1) * stride],
        )
    };
    for (x, y) in d.iter_mut().zip(s) {
        *x ^= y;
    }
}

impl From<&Matrix<Number>> for BitMatrix {
    /// The matrix of integers modulo 2: each element's parity, negative ones too (`-3 ≡ 1`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{BitMatrix, Matrix};
    /// use a::math::numbers::Number;
    ///
    /// let m = Matrix::from_rows(&[[2, -3], [7, 0]]).map(|&x| Number::from(x));
    /// assert_eq!(BitMatrix::from(&m), BitMatrix::from_rows(&[[false, true], [true, false]]));
    /// ```
    fn from(m: &Matrix<Number>) -> BitMatrix {
        BitMatrix::from_fn(m.rows(), m.cols(), |i, j| m[(i, j)].bit(0))
    }
}

/// Panics unless the two matrices have the same shape.
#[track_caller]
fn same_shape(a: &BitMatrix, b: &BitMatrix) {
    assert!(
        a.rows == b.rows && a.cols == b.cols,
        "the matrices are not of the same shape"
    );
}

/// `a · b`.
#[track_caller]
fn product(a: &BitMatrix, b: &BitMatrix) -> BitMatrix {
    assert!(
        a.cols == b.rows,
        "the columns of the left matrix are not as many as the rows of the right one"
    );
    mul::product(a, b)
}

impl AddAssign<&BitMatrix> for BitMatrix {
    /// Adds `rhs`, element by element: an exclusive or of the words.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    #[allow(
        clippy::suspicious_op_assign_impl,
        reason = "the sum over GF(2) is the exclusive or"
    )]
    fn add_assign(&mut self, rhs: &BitMatrix) {
        same_shape(self, rhs);
        for (x, y) in self.data.iter_mut().zip(&rhs.data) {
            *x ^= y;
        }
    }
}

impl AddAssign<BitMatrix> for BitMatrix {
    #[track_caller]
    fn add_assign(&mut self, rhs: BitMatrix) {
        *self += &rhs;
    }
}

impl Add<&BitMatrix> for &BitMatrix {
    type Output = BitMatrix;
    /// The sum, element by element: an exclusive or of the words.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[track_caller]
    fn add(self, rhs: &BitMatrix) -> BitMatrix {
        let mut s = self.clone();
        s += rhs;
        s
    }
}

impl Add<BitMatrix> for &BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn add(self, mut rhs: BitMatrix) -> BitMatrix {
        rhs += self;
        rhs
    }
}

impl Add<&BitMatrix> for BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn add(mut self, rhs: &BitMatrix) -> BitMatrix {
        self += rhs;
        self
    }
}

impl Add<BitMatrix> for BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn add(mut self, rhs: BitMatrix) -> BitMatrix {
        self += &rhs;
        self
    }
}

impl Mul<&BitMatrix> for &BitMatrix {
    type Output = BitMatrix;
    /// The matrix product, by M4RM, with Strassen–Winograd above it for large matrices.
    ///
    /// # Panics
    ///
    /// Panics if the columns of `self` are not as many as the rows of `rhs`.
    #[track_caller]
    fn mul(self, rhs: &BitMatrix) -> BitMatrix {
        product(self, rhs)
    }
}

impl Mul<BitMatrix> for &BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn mul(self, rhs: BitMatrix) -> BitMatrix {
        product(self, &rhs)
    }
}

impl Mul<&BitMatrix> for BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn mul(self, rhs: &BitMatrix) -> BitMatrix {
        product(&self, rhs)
    }
}

impl Mul<BitMatrix> for BitMatrix {
    type Output = BitMatrix;
    #[track_caller]
    fn mul(self, rhs: BitMatrix) -> BitMatrix {
        product(&self, &rhs)
    }
}

impl MulAssign<&BitMatrix> for BitMatrix {
    #[track_caller]
    fn mul_assign(&mut self, rhs: &BitMatrix) {
        *self = product(self, rhs);
    }
}

impl MulAssign<BitMatrix> for BitMatrix {
    #[track_caller]
    fn mul_assign(&mut self, rhs: BitMatrix) {
        *self = product(self, &rhs);
    }
}

/// A row written as `0`s and `1`s.
struct Bits<'a>(&'a BitMatrix, usize);

impl fmt::Display for Bits<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row = &self.0.data[self.1 * self.0.stride..(self.1 + 1) * self.0.stride];
        for j in 0..self.0.cols {
            f.write_str(if row[j / 64] >> (j % 64) & 1 == 1 {
                "1"
            } else {
                "0"
            })?;
        }
        Ok(())
    }
}

impl fmt::Debug for Bits<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{self}\"")
    }
}

/// The rows of a matrix for [`fmt::Debug`].
struct Rows<'a>(&'a BitMatrix);

impl fmt::Debug for Rows<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.0.rows).map(|i| Bits(self.0, i)))
            .finish()
    }
}

impl fmt::Debug for BitMatrix {
    /// Writes the shape and the rows as strings of `0`s and `1`s:
    /// `BitMatrix { rows: 2, cols: 2, data: ["10", "01"] }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &Rows(self))
            .finish()
    }
}

impl fmt::Display for BitMatrix {
    /// Writes the rows on lines of their own, each as its `0`s and `1`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::BitMatrix;
    ///
    /// assert_eq!(BitMatrix::identity(2).to_string(), "10\n01");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", Bits(self, i))?;
        }
        Ok(())
    }
}
