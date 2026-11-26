//! Matrices over the prime field `GF(p)` for a prime `p < 2^64`, the modulus stored once and each
//! element a word in `[0, p)`, held row after row.
//!
//! The arithmetic is on words ([`field`]): sums with one comparison, products reduced by the
//! reciprocal of the normalised `p` (Möller and Granlund), products by a fixed multiplier with
//! Shoup's precomputed quotient, and dot products whose sums are held unreduced in one, two or
//! three words, as the length and `p` allow, and reduced once (as FLINT's `nmod_mat` does). The
//! product of matrices is classical below a crossover and Strassen–Winograd's recursion above it
//! ([`product`]). Elimination is the PLE decomposition ([`ple`]), recursive on the columns so that
//! its work is products, `O(n^ω)`; the rank, the determinant, the inverse, the solutions, the
//! reduced row echelon form and the null space come from it by triangular solves.
//!
//! The modulus is checked once, when a matrix is built, by the crate's primality test for words
//! (trial division and Baillie–PSW, which no composite below `2^64` passes).

mod crt;
mod doubles;
mod field;
mod ple;
mod primes;
mod product;

pub(super) use crt::Crt;
pub(super) use field::Field;
pub(super) use ple::decompose;
pub(super) use primes::TopPrimes;
pub(super) use product::{Mode as ProductMode, product as product_mod};

use super::Matrix;
use crate::math::numbers::{Magnitude, Number, is_prime_u64};
use core::ops::{Add, Index, Mul, Neg, Sub};
use product::Mode;

/// A matrix over `GF(p)`, `p` a prime below `2^64` stored once, its elements words in `[0, p)`.
///
/// # Examples
///
/// ```
/// use a::math::matrix::ModMatrix;
///
/// let a = ModMatrix::from_rows(&[[1, 2], [3, 4]], 7);
/// let inv = a.inverse().unwrap();
/// assert_eq!(&a * &inv, ModMatrix::identity(2, 7));
/// assert_eq!(a.det(), 5); // 1·4 - 2·3 = -2 = 5 (mod 7)
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ModMatrix {
    /// The number of rows.
    rows: usize,
    /// The number of columns.
    cols: usize,
    /// The prime.
    p: u64,
    /// The elements, row after row.
    data: Vec<u64>,
}

/// The number of elements of a `rows × cols` matrix.
///
/// # Panics
///
/// Panics if it overflows `usize`.
#[track_caller]
fn size(rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols)
        .expect("the matrix has more elements than a vector can hold")
}

/// Panics unless `p` is a prime.
#[track_caller]
fn check_prime(p: u64) {
    assert!(is_prime_u64(p), "the modulus {p} is not a prime");
}

/// The PLE decomposition of a matrix over `GF(p)`, `P·A = L·E` with `E` in row echelon form: for
/// an `m × n` matrix `A` of rank `r`, row `i` of `L·E` is row `perm[i]` of `A`, `L` is `m × r`
/// lower triangular with ones on its diagonal, and `E` is `r × n`, row `i` zero before its pivot
/// column `q_i`, the pivot columns increasing, so that they are the column rank profile of `A`.
/// It is what [`ModMatrix::ple`] returns.
///
/// # Examples
///
/// ```
/// use a::math::matrix::ModMatrix;
///
/// let a = ModMatrix::from_rows(&[[0, 2, 4], [1, 1, 1], [2, 4, 6]], 11);
/// let d = a.ple();
/// assert_eq!(d.rank(), 2);
/// assert_eq!(d.pivots(), &[0, 1]);
/// let le = d.l() * d.e();
/// for (i, &r) in d.permutation().iter().enumerate() {
///     assert_eq!(le.row(i), a.row(r));
/// }
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ModPle {
    /// Row `i` of `L·E` is row `perm[i]` of `A`.
    perm: Vec<usize>,
    /// `L`, `m × r`.
    l: ModMatrix,
    /// `E`, `r × n`.
    e: ModMatrix,
    /// The pivot columns.
    pivots: Vec<usize>,
}

impl ModPle {
    /// The permutation: row `i` of `L·E` is row `permutation()[i]` of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// // The first nonzero element of the first column is in row 2.
    /// let a = ModMatrix::from_rows(&[[0, 1], [0, 2], [1, 1]], 3);
    /// assert_eq!(a.ple().permutation(), &[2, 1, 0]);
    /// ```
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// `L`, `m × r`, lower triangular with ones on its diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2], [3, 4]], 5);
    /// // The second row less 3 times the first: `L` holds the 3.
    /// assert_eq!(a.ple().l(), &ModMatrix::from_rows(&[[1, 0], [3, 1]], 5));
    /// ```
    #[must_use]
    pub fn l(&self) -> &ModMatrix {
        &self.l
    }

    /// `E`, `r × n`, in row echelon form: row `i` zero before column `pivots()[i]`, not zero
    /// there.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2], [3, 4]], 5);
    /// // 4 - 3·2 = -2 = 3 (mod 5).
    /// assert_eq!(a.ple().e(), &ModMatrix::from_rows(&[[1, 2], [0, 3]], 5));
    /// ```
    #[must_use]
    pub fn e(&self) -> &ModMatrix {
        &self.e
    }

    /// The pivot columns of `E`, increasing: the first `r` columns of `A` that are independent of
    /// the columns before them.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// // The second column is twice the first.
    /// let a = ModMatrix::from_rows(&[[1, 2, 0], [2, 4, 1]], 7);
    /// assert_eq!(a.ple().pivots(), &[0, 2]);
    /// ```
    #[must_use]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// The rank `r`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::from_rows(&[[1, 1], [1, 1]], 2).ple().rank(), 1);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }
}

impl ModMatrix {
    /// The matrix of the given parts, the modulus already checked and the elements reduced.
    fn raw(rows: usize, cols: usize, p: u64, data: Vec<u64>) -> ModMatrix {
        debug_assert_eq!(data.len(), rows * cols);
        ModMatrix {
            rows,
            cols,
            p,
            data,
        }
    }

    /// The arithmetic modulo `p`.
    fn field(&self) -> Field {
        Field::new(self.p)
    }

    /// The `rows × cols` zero matrix over `GF(p)`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime, or `rows · cols` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let z = ModMatrix::zeros(2, 3, 5);
    /// assert_eq!((z.rows(), z.cols(), z.modulus()), (2, 3, 5));
    /// assert_eq!(z.rank(), 0);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn zeros(rows: usize, cols: usize, p: u64) -> ModMatrix {
        check_prime(p);
        ModMatrix::raw(rows, cols, p, vec![0; size(rows, cols)])
    }

    /// The `n × n` identity matrix over `GF(p)`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let i = ModMatrix::identity(3, 2);
    /// assert_eq!(i.det(), 1);
    /// assert_eq!(i.get(1, 1), 1);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn identity(n: usize, p: u64) -> ModMatrix {
        let mut a = ModMatrix::zeros(n, n, p);
        for i in 0..n {
            a.data[i * n + i] = 1;
        }
        a
    }

    /// The matrix whose element in row `i` and column `j` is `f(i, j)` reduced modulo `p`,
    /// computed row after row.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime, or `rows · cols` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_fn(2, 2, 3, |i, j| (i + 2 * j) as u64);
    /// assert_eq!(a, ModMatrix::from_rows(&[[0, 2], [1, 0]], 3));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_fn(
        rows: usize,
        cols: usize,
        p: u64,
        mut f: impl FnMut(usize, usize) -> u64,
    ) -> ModMatrix {
        check_prime(p);
        let field = Field::new(p);
        let mut data = Vec::with_capacity(size(rows, cols));
        for i in 0..rows {
            data.extend((0..cols).map(|j| field.reduce(f(i, j))));
        }
        ModMatrix::raw(rows, cols, p, data)
    }

    /// The matrix with the given rows, each element reduced modulo `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime, or the rows are not all of the same length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[5, 6, 7]], 5);
    /// assert_eq!(a.row(0), &[0, 1, 2]);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_rows<R: AsRef<[u64]>>(rows: &[R], p: u64) -> ModMatrix {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        assert!(
            rows.iter().all(|r| r.as_ref().len() == cols),
            "the rows are not all of the same length"
        );
        ModMatrix::from_fn(rows.len(), cols, p, |i, j| rows[i].as_ref()[j])
    }

    /// The matrix of the residues modulo `p`, in `[0, p)`, of the elements of a matrix of numbers
    /// of any size and sign.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, ModMatrix};
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[-1, 10], [7, 3]]).map(|&x| Number::from(x));
    /// let m = ModMatrix::from_matrix(&a, 7);
    /// assert_eq!(m, ModMatrix::from_rows(&[[6, 3], [0, 3]], 7));
    /// assert_eq!(a.det(), -73);
    /// assert_eq!(m.det(), 4); // -73 = 4 (mod 7)
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_matrix(a: &Matrix<Number>, p: u64) -> ModMatrix {
        check_prime(p);
        ModMatrix::from_matrix_prime(a, p)
    }

    /// [`from_matrix`](Self::from_matrix) for a `p` the caller knows to be a prime (one of
    /// [`TopPrimes`]), without testing it again.
    pub(super) fn from_matrix_prime(a: &Matrix<Number>, p: u64) -> ModMatrix {
        let f = Field::new(p);
        let (rows, cols) = a.shape();
        let mut data = vec![0; size(rows, cols)];
        // Each element's magnitude read in place: a word reduced once, a longer one limb by
        // limb against the residues of 2^(64·i), products independent of each other.
        let longest = a
            .as_slice()
            .iter()
            .map(|x| Magnitude::of(x).bytes.len().div_ceil(8))
            .max()
            .unwrap_or(0);
        let powers = f.powers(longest.max(1));
        let mut limbs: Vec<u64> = Vec::with_capacity(longest);
        for j in 0..cols {
            for (i, x) in a.column(j).iter().enumerate() {
                let m = Magnitude::of(x);
                let r = if m.bytes.len() <= 8 {
                    f.reduce(m.small() as u64)
                } else {
                    limbs.clear();
                    if m.big_endian {
                        limbs.extend(m.bytes.rchunks(8).map(|c| {
                            let mut b = [0u8; 8];
                            b[8 - c.len()..].copy_from_slice(c);
                            u64::from_be_bytes(b)
                        }));
                    } else {
                        limbs.extend(m.bytes.chunks(8).map(|c| {
                            let mut b = [0u8; 8];
                            b[..c.len()].copy_from_slice(c);
                            u64::from_le_bytes(b)
                        }));
                    }
                    f.of_limbs_by(false, &limbs, &powers)
                };
                data[i * cols + j] = if m.negative { f.neg(r) } else { r };
            }
        }
        ModMatrix::raw(rows, cols, p, data)
    }

    /// The matrix of numbers in `[0, p)` with the same elements, little-endian.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, ModMatrix};
    /// use a::math::numbers::Number;
    ///
    /// let m = ModMatrix::from_rows(&[[1, 2]], 3);
    /// assert_eq!(m.to_matrix(), Matrix::from_rows(&[[1, 2]]).map(|&x| Number::from(x)));
    /// ```
    #[must_use]
    pub fn to_matrix(&self) -> Matrix<Number> {
        Matrix::from_fn(self.rows, self.cols, |i, j| {
            Number::from(self.data[i * self.cols + j])
        })
    }

    /// The number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::zeros(2, 3, 5).rows(), 2);
    /// ```
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::zeros(2, 3, 5).cols(), 3);
    /// ```
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::zeros(2, 3, 5).shape(), (2, 3));
    /// ```
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Whether it has as many rows as columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert!(ModMatrix::identity(3, 5).is_square());
    /// assert!(!ModMatrix::zeros(2, 3, 5).is_square());
    /// ```
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The prime `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::zeros(1, 1, 65521).modulus(), 65521);
    /// ```
    #[must_use]
    pub fn modulus(&self) -> u64 {
        self.p
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
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2], [3, 4]], 3);
    /// assert_eq!(a.get(1, 0), 0);
    /// assert_eq!(a[(1, 1)], 1);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> u64 {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        self.data[i * self.cols + j]
    }

    /// Sets the element in row `i` and column `j` to `x` reduced modulo `p`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let mut a = ModMatrix::zeros(2, 2, 3);
    /// a.set(0, 1, 7);
    /// assert_eq!(a.get(0, 1), 1);
    /// ```
    #[track_caller]
    pub fn set(&mut self, i: usize, j: usize, x: u64) {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        self.data[i * self.cols + j] = self.field().reduce(x);
    }

    /// Row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not below the number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::identity(2, 7).row(1), &[0, 1]);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn row(&self, i: usize) -> &[u64] {
        assert!(i < self.rows, "the row is outside the matrix");
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The elements, row after row.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// assert_eq!(ModMatrix::identity(2, 7).as_slice(), &[1, 0, 0, 1]);
    /// ```
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    /// The transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2, 3]], 5);
    /// assert_eq!(a.transpose(), ModMatrix::from_rows(&[[1], [2], [3]], 5));
    /// ```
    #[must_use]
    pub fn transpose(&self) -> ModMatrix {
        let (m, n) = (self.rows, self.cols);
        let mut data = vec![0; m * n];
        // In tiles that the caches hold on both sides, 16 × 16 words. Measured on an Apple M4
        // Max against tiles of 4, 8, 32 and 64 on square and thin shapes from 64 to 4096: the
        // best tile depends on the shape (8 takes 0.59 of the time for 4096 × 4096, whose rows of
        // the transpose map to the same sets of the cache, but 1.08 to 1.12 for 1000 × 1000,
        // 2000 × 300 and 4096 × 64; 64 the reverse, 0.84 to 0.87 and 1.45 to 1.67): no other
        // tile is faster for every shape.
        const T: usize = 16;
        for i0 in (0..m).step_by(T) {
            for j0 in (0..n).step_by(T) {
                for i in i0..(i0 + T).min(m) {
                    let row = &self.data[i * n..(i + 1) * n];
                    for j in j0..(j0 + T).min(n) {
                        data[j * m + i] = row[j];
                    }
                }
            }
        }
        ModMatrix::raw(n, m, self.p, data)
    }

    /// The product of every element by `c` reduced modulo `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2]], 5);
    /// assert_eq!(a.scale(3), ModMatrix::from_rows(&[[3, 1]], 5));
    /// ```
    #[must_use]
    pub fn scale(&self, c: u64) -> ModMatrix {
        let f = self.field();
        let mut data = self.data.clone();
        f.scale(&mut data, f.reduce(c));
        ModMatrix::raw(self.rows, self.cols, self.p, data)
    }

    /// The PLE decomposition `P·A = L·E` ([`ModPle`]), by Gaussian elimination recursive on the
    /// columns, whose work is products of blocks: `O(n^ω)` operations on words, `ω = log₂ 7`
    /// from the crossover of Strassen–Winograd's product, and about `n³/3` products of words and
    /// as many sums for an `n × n` matrix below it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[0, 1], [1, 0]], 2);
    /// let d = a.ple();
    /// assert_eq!(d.permutation(), &[1, 0]);
    /// assert_eq!(d.l(), &ModMatrix::identity(2, 2));
    /// assert_eq!(d.e(), &ModMatrix::identity(2, 2));
    /// ```
    #[must_use]
    pub fn ple(&self) -> ModPle {
        let (m, n) = (self.rows, self.cols);
        let mut a = self.data.clone();
        let e = decompose(self.field(), &mut a, m, n);
        let r = e.rank;
        let mut l = vec![0; m * r];
        for i in 0..m {
            for (k, &q) in e.pivots.iter().enumerate().take(i.min(r)) {
                l[i * r + k] = a[i * n + q];
            }
            if i < r {
                l[i * r + i] = 1;
            }
        }
        let mut u = vec![0; r * n];
        for (i, &q) in e.pivots.iter().enumerate() {
            u[i * n + q..(i + 1) * n].copy_from_slice(&a[i * n + q..(i + 1) * n]);
        }
        ModPle {
            perm: e.perm,
            l: ModMatrix::raw(m, r, self.p, l),
            e: ModMatrix::raw(r, n, self.p, u),
            pivots: e.pivots,
        }
    }

    /// The rank, by the PLE decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// // Dependent modulo 3, independent over the integers.
    /// let a = ModMatrix::from_rows(&[[1, 2], [2, 1]], 3);
    /// assert_eq!(a.rank(), 1);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        let mut a = self.data.clone();
        decompose(self.field(), &mut a, self.rows, self.cols).rank
    }

    /// The determinant, the product of the pivots of the PLE decomposition with the sign of its
    /// permutation; 1 for the `0 × 0` matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[2, 3], [1, 4]], 13);
    /// assert_eq!(a.det(), 5);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn det(&self) -> u64 {
        assert!(self.is_square(), "the matrix is not square");
        let mut a = self.data.clone();
        det_in_place(self.field(), &mut a, self.rows)
    }

    /// The factors of a square matrix of full rank stored over a copy of it, and the
    /// permutation: `None` when it is singular.
    fn lu(&self) -> Option<(Vec<u64>, Vec<usize>)> {
        let n = self.rows;
        let mut a = self.data.clone();
        let e = decompose(self.field(), &mut a, n, n);
        (e.rank == n).then_some((a, e.perm))
    }

    /// `X ← U⁻¹·L⁻¹·X` for the `n × n` factors `lu` of [`lu`](Self::lu) and the `n × w` matrix
    /// `x`, which this changes; the factors' upper triangle is scaled to a unit diagonal on the
    /// way.
    fn substitute(f: Field, lu: &mut [u64], n: usize, x: &mut [u64], w: usize) {
        ple::lower_unit(f, lu, n, x, w, n, w);
        // `U = D·U'` with `U'` of a unit diagonal: `U'⁻¹·(D⁻¹·X)`.
        for i in 0..n {
            let d = f.inv(lu[i * n + i]);
            f.scale(&mut lu[i * n + i + 1..(i + 1) * n], d);
            f.scale(&mut x[i * w..(i + 1) * w], d);
        }
        ple::upper_unit(f, lu, n, x, w, n, w);
    }

    /// The inverse, or `None` when the matrix is singular: from the PLE decomposition, which is
    /// `P·A = L·U` for a matrix of full rank, `A⁻¹ = U⁻¹·L⁻¹·P` by two triangular solves.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 1], [0, 1]], 2);
    /// assert_eq!(a.inverse(), Some(a.clone()));
    /// assert_eq!(ModMatrix::from_rows(&[[1, 1], [1, 1]], 2).inverse(), None);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn inverse(&self) -> Option<ModMatrix> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows;
        let (mut lu, perm) = self.lu()?;
        let mut x = vec![0; n * n];
        for (i, &r) in perm.iter().enumerate() {
            x[i * n + r] = 1;
        }
        ModMatrix::substitute(self.field(), &mut lu, n, &mut x, n);
        Some(ModMatrix::raw(n, n, self.p, x))
    }

    /// The solution `X` of `A·X = B`, or `None` when `A` is singular: `X = U⁻¹·L⁻¹·P·B` from the
    /// PLE decomposition, by two triangular solves.
    ///
    /// # Panics
    ///
    /// Panics if `A` is not square, `B` has not as many rows, or the two are over different
    /// primes.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[2, 1], [1, 3]], 7);
    /// let b = ModMatrix::from_rows(&[[1], [2]], 7);
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(&a * &x, b);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn solve(&self, b: &ModMatrix) -> Option<ModMatrix> {
        assert!(self.is_square(), "the matrix is not square");
        assert_eq!(
            b.rows, self.rows,
            "the right-hand side has not as many rows"
        );
        same_field(self, b);
        let (n, w) = (self.rows, b.cols);
        let (mut lu, perm) = self.lu()?;
        let mut x = Vec::with_capacity(n * w);
        for &r in &perm {
            x.extend_from_slice(b.row(r));
        }
        ModMatrix::substitute(self.field(), &mut lu, n, &mut x, w);
        Some(ModMatrix::raw(n, w, self.p, x))
    }

    /// The pivot columns of the reduced row echelon form, and the rows of its other columns,
    /// `r × (n - r)`: from the PLE decomposition, `E`'s rows scaled to leading ones, then
    /// `U⁻¹·F` for the unit upper triangular `U` of `E`'s pivot columns and the others `F`.
    fn reduced(&self) -> (Vec<usize>, Vec<usize>, Vec<u64>) {
        let (m, n) = (self.rows, self.cols);
        let f = self.field();
        let mut a = self.data.clone();
        let e = decompose(f, &mut a, m, n);
        let r = e.rank;
        let mut is_pivot = vec![false; n];
        for &q in &e.pivots {
            is_pivot[q] = true;
        }
        let free: Vec<usize> = (0..n).filter(|&c| !is_pivot[c]).collect();
        for (i, &q) in e.pivots.iter().enumerate() {
            let row = &mut a[i * n..(i + 1) * n];
            row[..q].fill(0);
            let d = f.inv(row[q]);
            f.scale(&mut row[q..], d);
        }
        let u = ple::gather(&a, n, 0, r, &e.pivots);
        let mut x = ple::gather(&a, n, 0, r, &free);
        ple::upper_unit(f, &u, r, &mut x, free.len(), r, free.len());
        (e.pivots, free, x)
    }

    /// The reduced row echelon form and its pivot columns: each nonzero row's first nonzero
    /// element is a 1 and the only nonzero element of its column, those rows above the zero rows
    /// and their pivots increasing. From the PLE decomposition and a triangular solve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[2, 4, 1], [1, 2, 4]], 5);
    /// let (r, pivots) = a.rref();
    /// assert_eq!(pivots, [0, 2]);
    /// assert_eq!(r, ModMatrix::from_rows(&[[1, 2, 0], [0, 0, 1]], 5));
    /// ```
    #[must_use]
    pub fn rref(&self) -> (ModMatrix, Vec<usize>) {
        let (m, n) = (self.rows, self.cols);
        let (pivots, free, x) = self.reduced();
        let w = free.len();
        let mut out = vec![0; m * n];
        for (i, &q) in pivots.iter().enumerate() {
            out[i * n + q] = 1;
            for (t, &c) in free.iter().enumerate() {
                out[i * n + c] = x[i * w + t];
            }
        }
        (ModMatrix::raw(m, n, self.p, out), pivots)
    }

    /// A basis of the null space `{x : A·x = 0}`, as the columns of an `n × (n - rank)` matrix:
    /// one vector for each column `c` without a pivot in the reduced row echelon form `R`, with
    /// `x_c = 1`, the other such coordinates 0, and `x_(q_i) = -R[i][c]` at the pivot columns
    /// `q_i`, in the order of the columns `c` (FLINT's `nmod_mat_nullspace` gives the same).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::ModMatrix;
    ///
    /// let a = ModMatrix::from_rows(&[[1, 2, 3], [2, 4, 6]], 7);
    /// let n = a.null_space();
    /// assert_eq!((n.rows(), n.cols()), (3, 2));
    /// assert_eq!(&a * &n, ModMatrix::zeros(2, 2, 7));
    /// ```
    #[must_use]
    pub fn null_space(&self) -> ModMatrix {
        let n = self.cols;
        let f = self.field();
        let (pivots, free, x) = self.reduced();
        let w = free.len();
        let mut out = vec![0; n * w];
        for (t, &c) in free.iter().enumerate() {
            out[c * w + t] = 1;
            for (i, &q) in pivots.iter().enumerate() {
                out[q * w + t] = f.neg(x[i * w + t]);
            }
        }
        ModMatrix::raw(n, w, self.p, out)
    }
}

/// The determinant of the `n × n` row-major matrix `a` modulo `p`, which this overwrites.
pub(super) fn det_in_place(f: Field, a: &mut [u64], n: usize) -> u64 {
    if n == 0 {
        return f.reduce(1);
    }
    let e = decompose(f, a, n, n);
    if e.rank < n {
        return 0;
    }
    let d = (0..n).fold(1, |d, i| f.mul(d, a[i * n + i]));
    if e.odd { f.neg(d) } else { d }
}

/// Panics unless the two matrices are over the same prime.
#[track_caller]
fn same_field(a: &ModMatrix, b: &ModMatrix) {
    assert_eq!(a.p, b.p, "the matrices are over different primes");
}

/// Panics unless the two matrices have the same shape and prime.
#[track_caller]
fn same_shape(a: &ModMatrix, b: &ModMatrix) {
    same_field(a, b);
    assert!(
        a.shape() == b.shape(),
        "the matrices are not of the same shape"
    );
}

/// `a ± b` element by element.
#[track_caller]
fn elementwise(a: &ModMatrix, b: &ModMatrix, subtract: bool) -> ModMatrix {
    same_shape(a, b);
    let p = a.p;
    // Without a branch on the data, so that the loops are vectors: a difference that borrows
    // gets p back, a sum that carries out of the word or reaches p loses it.
    let data: Vec<u64> = if subtract {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| {
                let d = x.wrapping_sub(y);
                d.wrapping_add(p & ((x < y) as u64).wrapping_neg())
            })
            .collect()
    } else {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| {
                let (s, carry) = x.overflowing_add(y);
                let over = (carry | (s >= p)) as u64;
                s.wrapping_sub(p & over.wrapping_neg())
            })
            .collect()
    };
    ModMatrix::raw(a.rows, a.cols, a.p, data)
}

/// `a + b`.
#[track_caller]
fn sum(a: &ModMatrix, b: &ModMatrix) -> ModMatrix {
    elementwise(a, b, false)
}

/// `a - b`.
#[track_caller]
fn difference(a: &ModMatrix, b: &ModMatrix) -> ModMatrix {
    elementwise(a, b, true)
}

/// `a · b`: classical below Strassen–Winograd's crossover, by its recursion above.
#[track_caller]
fn multiply(a: &ModMatrix, b: &ModMatrix) -> ModMatrix {
    same_field(a, b);
    assert!(
        a.cols == b.rows,
        "the columns of the left matrix are not as many as the rows of the right one"
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut c = vec![0; size(m, n)];
    product::product(
        a.field(),
        &mut c,
        n,
        &a.data,
        k,
        &b.data,
        n,
        (m, k, n),
        Mode::Set,
    );
    ModMatrix::raw(m, n, a.p, c)
}

/// A binary operator on matrices, owned or borrowed, by the function `$f`.
macro_rules! binary {
    ($op:ident, $method:ident, $f:ident) => {
        impl $op<&ModMatrix> for &ModMatrix {
            type Output = ModMatrix;
            /// # Panics
            ///
            /// Panics if the shapes do not fit, or the matrices are over different primes.
            #[track_caller]
            fn $method(self, rhs: &ModMatrix) -> ModMatrix {
                $f(self, rhs)
            }
        }
        impl $op<ModMatrix> for &ModMatrix {
            type Output = ModMatrix;
            /// # Panics
            ///
            /// Panics if the shapes do not fit, or the matrices are over different primes.
            #[track_caller]
            fn $method(self, rhs: ModMatrix) -> ModMatrix {
                $f(self, &rhs)
            }
        }
        impl $op<&ModMatrix> for ModMatrix {
            type Output = ModMatrix;
            /// # Panics
            ///
            /// Panics if the shapes do not fit, or the matrices are over different primes.
            #[track_caller]
            fn $method(self, rhs: &ModMatrix) -> ModMatrix {
                $f(&self, rhs)
            }
        }
        impl $op<ModMatrix> for ModMatrix {
            type Output = ModMatrix;
            /// # Panics
            ///
            /// Panics if the shapes do not fit, or the matrices are over different primes.
            #[track_caller]
            fn $method(self, rhs: ModMatrix) -> ModMatrix {
                $f(&self, &rhs)
            }
        }
    };
}

binary!(Add, add, sum);
binary!(Sub, sub, difference);
binary!(Mul, mul, multiply);

impl Neg for &ModMatrix {
    type Output = ModMatrix;
    /// `-A`, element by element.
    fn neg(self) -> ModMatrix {
        let f = self.field();
        let data = self.data.iter().map(|&x| f.neg(x)).collect();
        ModMatrix::raw(self.rows, self.cols, self.p, data)
    }
}

impl Neg for ModMatrix {
    type Output = ModMatrix;
    /// `-A`, element by element.
    fn neg(self) -> ModMatrix {
        -&self
    }
}

impl Index<(usize, usize)> for ModMatrix {
    type Output = u64;

    /// The element in row `i` and column `j`.
    ///
    /// # Panics
    ///
    /// Panics outside the matrix.
    #[track_caller]
    fn index(&self, (i, j): (usize, usize)) -> &u64 {
        assert!(i < self.rows && j < self.cols, "({i}, {j}) is out of range");
        &self.data[i * self.cols + j]
    }
}
