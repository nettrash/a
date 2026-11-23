//! The operations on sparse matrices: the products, the sums, the norms, and the sub-matrices,
//! triangles and permutations taken out of one.
//!
//! The operators are the dense ones', on owned or borrowed operands: `&a + &b`, `&a * &b`, `&a *
//! alpha`. What differs is the cost, which follows the nonzeros — the sum of two matrices walks the
//! two rows of every row at once, and the product of two matrices walks the rows of the second
//! through the columns of the first, each row of the result built in an accumulator the length of a
//! row and read once.
//!
//! An element that is not stored is an exact zero and stays one: the product of a matrix by an
//! infinite or NaN element, and its quotient by zero, change the stored elements alone, where the
//! dense matrix would fill with NaNs, as `scipy.sparse` has it too. A stored element follows IEEE
//! arithmetic: an infinity or a NaN times zero is a NaN, and stays stored.

use super::{SparseMatrix, sort_rows, sorted_order};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Norm, Scalar};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign};

/// A sum of `f64`s by Neumaier's compensated summation: beside the running sum it keeps the
/// rounding errors of the additions, and adds them at the end, so that the error of the result is
/// of the order of one rounding rather than one for each term. The norms of a matrix of millions
/// of elements are sums that long.
#[derive(Clone, Copy)]
struct Sum {
    /// The running sum.
    sum: f64,
    /// The rounding errors of the additions so far.
    carry: f64,
}

impl Sum {
    /// The empty sum.
    fn new() -> Self {
        Self {
            sum: 0.0,
            carry: 0.0,
        }
    }

    /// Adds `x`.
    fn add(&mut self, x: f64) {
        let total = self.sum + x;
        // An infinite or NaN total has no rounding error to keep, and would poison the carry.
        if total.is_finite() {
            self.carry += if self.sum.abs() >= x.abs() {
                (self.sum - total) + x
            } else {
                (x - total) + self.sum
            };
        }
        self.sum = total;
    }

    /// The sum.
    fn value(self) -> f64 {
        if self.sum.is_finite() {
            self.sum + self.carry
        } else {
            self.sum
        }
    }
}

/// The larger of `m` and `x`, or a NaN when either is one: `f64::max` ignores NaNs, which would
/// let a norm of a matrix holding one be finite.
fn max_nan(m: f64, x: f64) -> f64 {
    if m.is_nan() || x.is_nan() {
        f64::NAN
    } else {
        m.max(x)
    }
}

/// The absolute value of `x` as an `f64`, in which the sums of the norms are kept whatever the
/// type of the elements: a sum of millions of `f32`s loses nothing that way.
fn magnitude<T: Scalar>(x: T) -> f64 {
    x.re().to_f64().hypot(x.im().to_f64())
}

/// The square root of the sum of the squares of the absolute values of `values`: the squares
/// scaled by a power of two so that they neither overflow nor underflow, and summed with
/// compensation. An infinite element gives infinity, and otherwise a NaN gives NaN, as `hypot`
/// has it.
fn frobenius<T: Scalar>(values: &[T]) -> T::Real {
    let mut largest = 0.0f64;
    let mut nan = false;
    for x in values {
        for part in [x.re().to_f64(), x.im().to_f64()] {
            if part.is_nan() {
                nan = true;
            } else {
                largest = largest.max(part.abs());
            }
        }
    }
    if largest.is_infinite() {
        return T::Real::INFINITY;
    }
    if nan {
        return T::Real::NAN;
    }
    if largest == 0.0 {
        return T::Real::ZERO;
    }
    // The power of two at or below the largest part, by which a division is exact; a subnormal
    // has none in range, and is divided by itself.
    let power = f64::from_bits(largest.to_bits() & !((1u64 << 52) - 1));
    let scale = if power > 0.0 { power } else { largest };
    let mut sum = Sum::new();
    for x in values {
        let re = x.re().to_f64() / scale;
        sum.add(re * re);
        if T::IS_COMPLEX {
            let im = x.im().to_f64() / scale;
            sum.add(im * im);
        }
    }
    T::Real::from_f64(scale * sum.value().sqrt())
}

/// The inverse of the permutation `p` of `0..n`, `inverse[old] = new` where `p[new] = old`.
///
/// # Panics
///
/// Panics if `p` isn't a permutation of `0..n`; `what` names the rows or the columns in the
/// message.
#[track_caller]
fn inverse_permutation(p: &[usize], n: usize, what: &str) -> Vec<usize> {
    assert_eq!(
        p.len(),
        n,
        "the permutation of the {what} isn't of the right length"
    );
    let mut inverse = vec![usize::MAX; n];
    for (new, &old) in p.iter().enumerate() {
        assert!(old < n, "the permutation of the {what} is out of range");
        assert!(
            inverse[old] == usize::MAX,
            "the permutation of the {what} repeats an index"
        );
        inverse[old] = new;
    }
    inverse
}

impl<T: Scalar> SparseMatrix<T> {
    /// The product by the dense `x`, which has as many rows as the matrix has columns: `a·x`, in
    /// the time the stored elements cost for each column of `x`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes don't fit.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
    /// let x = Matrix::from_rows(&[[1.0], [2.0]]);
    /// assert_eq!(a.mul_matrix(&x), Matrix::from_rows(&[[4.0], [6.0]]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_matrix(&self, x: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, x.rows(), "the shapes don't fit for a product");
        let mut product = Matrix::zeros(self.rows, x.cols());
        for c in 0..x.cols() {
            self.mul_vec_into(x.column(c), product.column_mut(c));
        }
        product
    }

    /// `y = a·x` written into `y`, for `x` of length `cols` and `y` of length `rows`: one pass over
    /// the stored elements and no allocation, which is what a Krylov method wants per iteration.
    ///
    /// # Panics
    ///
    /// Panics if `x` and `y` aren't of the right lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
    /// let mut y = [0.0; 2];
    /// a.mul_vec_into(&[1.0, 2.0], &mut y);
    /// assert_eq!(y, [4.0, 6.0]);
    /// ```
    #[track_caller]
    pub fn mul_vec_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.cols,
            "the vector has the wrong length for the matrix"
        );
        assert_eq!(
            y.len(),
            self.rows,
            "the result has the wrong length for the matrix"
        );
        for (i, y) in y.iter_mut().enumerate() {
            let mut sum = T::zero();
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            *y = sum;
        }
    }

    /// `y = a·x`, the vector form of [`mul_matrix`](Self::mul_matrix).
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 2.0), (0, 2, 1.0), (1, 1, 3.0)]);
    /// assert_eq!(a.mul_vec(&[1.0, 2.0, 3.0]), [5.0, 6.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.rows];
        self.mul_vec_into(x, &mut y);
        y
    }

    /// `y = aᵀ·x` written into `y`, for `x` of length `rows` and `y` of length `cols`, without the
    /// transpose being formed: every stored element adds to the entry of `y` its column names.
    ///
    /// # Panics
    ///
    /// Panics if `x` and `y` aren't of the right lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 2.0), (0, 2, 1.0), (1, 1, 3.0)]);
    /// let mut y = [0.0; 3];
    /// a.mul_vec_transpose_into(&[1.0, 2.0], &mut y);
    /// assert_eq!(y, [2.0, 6.0, 1.0]);
    /// ```
    #[track_caller]
    pub fn mul_vec_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.scatter(x, y, |a| a);
    }

    /// `aᵀ·x`, for `x` of length `rows`, without the transpose being formed.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `rows`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 2.0), (0, 2, 1.0), (1, 1, 3.0)]);
    /// assert_eq!(a.mul_vec_transpose(&[1.0, 2.0]), a.transpose().mul_vec(&[1.0, 2.0]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec_transpose(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.cols];
        self.mul_vec_transpose_into(x, &mut y);
        y
    }

    /// `y = aᴴ·x` written into `y`, for `x` of length `rows` and `y` of length `cols`, without the
    /// conjugate transpose being formed. For a real matrix it is
    /// [`mul_vec_transpose_into`](Self::mul_vec_transpose_into).
    ///
    /// # Panics
    ///
    /// Panics if `x` and `y` aren't of the right lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(1, 2, &[(0, 1, Complex::new(1.0, 2.0))]);
    /// let mut y = [Complex::ZERO; 2];
    /// a.mul_vec_adjoint_into(&[Complex::new(0.0, 1.0)], &mut y);
    /// // conj(1 + 2i)·i = 2 + i
    /// assert_eq!(y, [Complex::ZERO, Complex::new(2.0, 1.0)]);
    /// ```
    #[track_caller]
    pub fn mul_vec_adjoint_into(&self, x: &[T], y: &mut [T]) {
        self.scatter(x, y, |a| a.conj());
    }

    /// `aᴴ·x`, for `x` of length `rows`, without the conjugate transpose being formed.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `rows`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(1, 2, &[(0, 1, Complex::new(1.0, 2.0))]);
    /// let x = [Complex::new(0.0, 1.0)];
    /// assert_eq!(a.mul_vec_adjoint(&x), a.conjugate_transpose().mul_vec(&x));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec_adjoint(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.cols];
        self.mul_vec_adjoint_into(x, &mut y);
        y
    }

    /// `y = conj(a)·x` written into `y`: the product by the matrix of the conjugates, which is
    /// what the conjugate transpose of a matrix held by columns multiplies by.
    #[track_caller]
    pub(crate) fn mul_vec_conjugate_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.cols,
            "the vector has the wrong length for the matrix"
        );
        assert_eq!(
            y.len(),
            self.rows,
            "the result has the wrong length for the matrix"
        );
        for (i, y) in y.iter_mut().enumerate() {
            let mut sum = T::zero();
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                sum += self.values[k].conj() * x[self.col_indices[k]];
            }
            *y = sum;
        }
    }

    /// `y = op(a)ᵀ·x` written into `y`, where `op` is applied to each stored element: row `i` of
    /// the matrix is column `i` of its transpose, so its elements, times `x[i]`, are added to the
    /// entries of `y` their columns name.
    #[track_caller]
    fn scatter(&self, x: &[T], y: &mut [T], op: impl Fn(T) -> T) {
        assert_eq!(
            x.len(),
            self.rows,
            "the vector has the wrong length for the matrix"
        );
        assert_eq!(
            y.len(),
            self.cols,
            "the result has the wrong length for the matrix"
        );
        y.fill(T::zero());
        for (i, &xi) in x.iter().enumerate() {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                y[self.col_indices[k]] += op(self.values[k]) * xi;
            }
        }
    }

    /// The sum or difference, element by element: the two rows of a row are walked together, so the
    /// cost is what the two patterns together cost and a cell in one matrix alone keeps its element.
    #[track_caller]
    fn add_sub(&self, other: &Self, subtract: bool) -> Self {
        assert_eq!(
            self.rows, other.rows,
            "the matrices aren't of the same shape"
        );
        assert_eq!(
            self.cols, other.cols,
            "the matrices aren't of the same shape"
        );
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut row_ptrs = Vec::with_capacity(self.rows + 1);
        for i in 0..self.rows {
            row_ptrs.push(col_indices.len());
            let (a_begin, a_end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
            let (b_begin, b_end) = (other.row_ptrs[i], other.row_ptrs[i + 1]);
            let (mut p, mut q) = (a_begin, b_begin);
            while p < a_end || q < b_end {
                // Whichever row has the smaller column next contributes it alone; equal columns
                // meet and their elements combine, cancelling out of the pattern when they do.
                let (j, x) =
                    if q >= b_end || (p < a_end && self.col_indices[p] < other.col_indices[q]) {
                        let x = self.values[p];
                        p += 1;
                        (self.col_indices[p - 1], x)
                    } else if p >= a_end || other.col_indices[q] < self.col_indices[p] {
                        let x = other.values[q];
                        q += 1;
                        (other.col_indices[q - 1], if subtract { -x } else { x })
                    } else {
                        let x = if subtract {
                            self.values[p] - other.values[q]
                        } else {
                            self.values[p] + other.values[q]
                        };
                        let j = self.col_indices[p];
                        p += 1;
                        q += 1;
                        (j, x)
                    };
                if !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows: self.rows,
            cols: self.cols,
            row_ptrs,
            col_indices,
            values,
            zero: self.zero,
        }
    }

    /// The product of two sparse matrices, `a·b`, by Gustavson's method (1978): each row of the
    /// result is gathered into an accumulator as long as a row, over the rows of `b` that the row
    /// of `a` points at, and read once at the end. The time is the multiplications there are to
    /// do, and a sort of the columns of each row of the result.
    #[track_caller]
    fn product(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "the shapes don't fit for a product");
        let mut gather = vec![T::zero(); other.cols];
        // `seen[j]` is the row being built when column j was last touched, which says whether the
        // column is already in the row without a second pass over it.
        let mut seen = vec![usize::MAX; other.cols];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut row_ptrs = Vec::with_capacity(self.rows + 1);
        let mut touched: Vec<usize> = Vec::new();
        for i in 0..self.rows {
            row_ptrs.push(col_indices.len());
            touched.clear();
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let (p, aij) = (self.col_indices[k], self.values[k]);
                for m in other.row_ptrs[p]..other.row_ptrs[p + 1] {
                    let j = other.col_indices[m];
                    if seen[j] != i {
                        seen[j] = i;
                        gather[j] = aij * other.values[m];
                        touched.push(j);
                    } else {
                        gather[j] += aij * other.values[m];
                    }
                }
            }
            touched.sort_unstable();
            for &j in &touched {
                let x = gather[j];
                if !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows: self.rows,
            cols: other.cols,
            row_ptrs,
            col_indices,
            values,
            zero: self.zero,
        }
    }

    /// The matrix with every stored element multiplied by `alpha`; a product that is zero leaves
    /// the pattern. The elements that aren't stored stay zeros whatever `alpha` is.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.5), (1, 0, -2.0)]);
    /// assert_eq!(a.scale(2.0), SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, -4.0)]));
    /// assert_eq!(a.scale(0.0).nnz(), 0);
    /// ```
    #[must_use]
    pub fn scale(&self, alpha: T) -> Self {
        self.map(|_, _, x| x * alpha)
    }

    /// A norm of the matrix, of the same kinds as [`Matrix::norm`]: `One` the largest column sum of
    /// absolute values, `Infinity` the largest row sum, `Frobenius` the square root of the sum of
    /// the squares and `Max` the largest absolute value — each in what the nonzeros cost. `Two` is
    /// the largest singular value, which needs the whole matrix: it densifies. `Nuclear` is the sum
    /// of the singular values and densifies too.
    ///
    /// The sums are kept in `f64` whatever the type of the elements, with compensation for their
    /// rounding errors, and Frobenius's is scaled so that the squares neither overflow nor
    /// underflow: the norm of a matrix of many millions of `f32`s is as accurate as that of a
    /// small one. A matrix holding a NaN has the norm NaN, except that an infinite element makes
    /// Frobenius's infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Norm;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::<f64>::from_coo(2, 2, &[(0, 0, 3.0), (0, 1, 4.0), (1, 0, -1.0)]);
    /// assert_eq!(a.norm(Norm::Infinity), 7.0);
    /// assert_eq!(a.norm(Norm::One), 4.0);
    /// assert_eq!(a.norm(Norm::Max), 4.0);
    /// assert!((a.norm(Norm::Frobenius) - 5.0990195135927845).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn norm(&self, norm: Norm) -> T::Real {
        match norm {
            Norm::One => T::Real::from_f64(self.largest_column_sum()),
            Norm::Infinity => {
                let mut largest = 0.0;
                for i in 0..self.rows {
                    let mut sum = Sum::new();
                    for &x in &self.values[self.row_ptrs[i]..self.row_ptrs[i + 1]] {
                        sum.add(magnitude(x));
                    }
                    largest = max_nan(largest, sum.value());
                }
                T::Real::from_f64(largest)
            }
            Norm::Frobenius => frobenius(&self.values),
            Norm::Max => {
                let largest = self
                    .values
                    .iter()
                    .fold(0.0, |m, &x| max_nan(m, magnitude(x)));
                T::Real::from_f64(largest)
            }
            Norm::Two | Norm::Nuclear => self.to_matrix().norm(norm),
        }
    }

    /// The largest sum of the absolute values of a column.
    fn largest_column_sum(&self) -> f64 {
        let mut largest = 0.0;
        if self.cols <= self.values.len().max(256) {
            let mut sums = vec![Sum::new(); self.cols];
            for (&j, &x) in self.col_indices.iter().zip(&self.values) {
                sums[j].add(magnitude(x));
            }
            for sum in sums {
                largest = max_nan(largest, sum.value());
            }
        } else {
            // Far more columns than elements: the elements in the order of their columns, summed
            // a column at a time, rather than a sum for every column there is.
            let order = sorted_order(self.values.len(), self.cols, |k| self.col_indices[k]);
            let mut column = usize::MAX;
            let mut sum = Sum::new();
            for k in order {
                if self.col_indices[k] != column {
                    largest = max_nan(largest, sum.value());
                    column = self.col_indices[k];
                    sum = Sum::new();
                }
                sum.add(magnitude(self.values[k]));
            }
            largest = max_nan(largest, sum.value());
        }
        largest
    }

    /// The trace `tr(A)`, the sum of the diagonal, as [`Matrix::trace`] has it: `0` of the
    /// `0 × 0` matrix. A binary search of each row finds its diagonal element.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, as [`Matrix::trace`] does;
    /// [`diagonal`](Self::diagonal) gives the `min(rows, cols)` diagonal elements of any shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 3.0), (0, 1, 4.0), (1, 1, -1.0)]);
    /// assert_eq!(a.trace(), 2.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn trace(&self) -> T {
        assert!(self.is_square(), "the matrix is not square");
        (0..self.rows)
            .map(|i| self[(i, i)])
            .fold(T::zero(), |s, x| s + x)
    }

    /// The rows `rows` and the columns `cols` of the matrix, as a matrix of their own: the element
    /// `(i, j)` of the result is the element `(rows.start + i, cols.start + j)` of this one.
    ///
    /// The time is proportional to the rows taken, with a binary search in each for the first
    /// column, and to the elements copied.
    ///
    /// # Panics
    ///
    /// Panics if a range ends beyond the matrix or before it begins.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]);
    /// let b = a.submatrix(1..3, 2..5);
    /// assert_eq!(b.shape(), (2, 3));
    /// assert_eq!(b, SparseMatrix::from_coo(2, 3, &[(0, 0, -1.0), (1, 0, 2.0), (1, 1, -1.0)]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn submatrix(&self, rows: Range<usize>, cols: Range<usize>) -> Self {
        assert!(
            rows.start <= rows.end && rows.end <= self.rows,
            "the rows are outside the matrix"
        );
        assert!(
            cols.start <= cols.end && cols.end <= self.cols,
            "the columns are outside the matrix"
        );
        let mut row_ptrs = Vec::with_capacity(rows.len() + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in rows.clone() {
            row_ptrs.push(col_indices.len());
            let (begin, end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
            let row = &self.col_indices[begin..end];
            let first = begin + row.partition_point(|&j| j < cols.start);
            let last = begin + row.partition_point(|&j| j < cols.end);
            col_indices.extend(
                self.col_indices[first..last]
                    .iter()
                    .map(|&j| j - cols.start),
            );
            values.extend_from_slice(&self.values[first..last]);
        }
        row_ptrs.push(col_indices.len());
        Self::from_canonical(rows.len(), cols.len(), row_ptrs, col_indices, values)
    }

    /// The matrix of the rows `rows` and the columns `cols` of this one, in the order the two
    /// lists give them: the element `(i, j)` of the result is the element `(rows[i], cols[j])`.
    /// The lists are any indices in any order, and an index may come more than once, which repeats
    /// its row or column.
    ///
    /// The time is proportional to the two lists and the result, and, for each row taken, to the
    /// shorter of that row and the list of columns, times a logarithm for the searches: nothing
    /// is paid for the rows and the columns that are not taken.
    ///
    /// # Panics
    ///
    /// Panics if an index is outside the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(3, 3, &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0)]);
    /// // Rows 2 and 0, and column 0 twice with column 2 between.
    /// let b = a.select(&[2, 0], &[0, 2, 0]);
    /// assert_eq!(b.shape(), (2, 3));
    /// assert_eq!(b.to_matrix().as_slice(), &[4.0, 1.0, 0.0, 2.0, 4.0, 1.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Self {
        assert!(
            rows.iter().all(|&i| i < self.rows),
            "a row is outside the matrix"
        );
        assert!(
            cols.iter().all(|&j| j < self.cols),
            "a column is outside the matrix"
        );
        // The positions in `cols` in ascending order of the column of the matrix each one takes:
        // `picked` lists the distinct columns taken, ascending, and the positions that take
        // `picked[t]` are `order[first[t]..first[t + 1]]`.
        let order = sorted_order(cols.len(), self.cols, |p| cols[p]);
        let mut picked: Vec<usize> = Vec::new();
        let mut first: Vec<usize> = Vec::new();
        for (at, &p) in order.iter().enumerate() {
            if picked.last() != Some(&cols[p]) {
                picked.push(cols[p]);
                first.push(at);
            }
        }
        first.push(order.len());
        let mut row_ptrs = Vec::with_capacity(rows.len() + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for &i in rows {
            row_ptrs.push(col_indices.len());
            let (begin, end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
            let row = &self.col_indices[begin..end];
            // The element at the offset `k` is in the column `picked[t]`: it goes to every position
            // that takes that column.
            let mut take = |k: usize, t: usize| {
                for &p in &order[first[t]..first[t + 1]] {
                    col_indices.push(p);
                    values.push(self.values[k]);
                }
            };
            // The row and the columns taken both ascend, so the shorter of the two is searched for
            // in the longer, each search beginning where the one before ended.
            let mut from = 0;
            if row.len() <= picked.len() {
                for (offset, j) in row.iter().enumerate() {
                    match picked[from..].binary_search(j) {
                        Ok(t) => {
                            take(begin + offset, from + t);
                            from += t + 1;
                        }
                        Err(t) => from += t,
                    }
                }
            } else {
                for (t, j) in picked.iter().enumerate() {
                    match row[from..].binary_search(j) {
                        Ok(offset) => {
                            take(begin + from + offset, t);
                            from += offset + 1;
                        }
                        Err(offset) => from += offset,
                    }
                }
            }
        }
        row_ptrs.push(col_indices.len());
        // A list of columns that doesn't ascend leaves the rows out of order.
        sort_rows(&row_ptrs, cols.len(), &mut col_indices, &mut values);
        Self::from_canonical(rows.len(), cols.len(), row_ptrs, col_indices, values)
    }

    /// The upper triangle with the diagonal: the elements `(i, j)` with `j ≥ i`, of a matrix of
    /// any shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 2], &[2.0; 3], &[-3.0; 2]);
    /// let u = a.upper_triangle();
    /// assert_eq!(u.nnz(), 5);
    /// assert_eq!((u[(0, 1)], u[(1, 1)], u[(1, 0)]), (-3.0, 2.0, 0.0));
    /// ```
    #[must_use]
    pub fn upper_triangle(&self) -> Self {
        self.triangle(true)
    }

    /// The lower triangle with the diagonal: the elements `(i, j)` with `j ≤ i`, of a matrix of
    /// any shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 2], &[2.0; 3], &[-3.0; 2]);
    /// let l = a.lower_triangle();
    /// assert_eq!(l.nnz(), 5);
    /// assert_eq!((l[(1, 0)], l[(1, 1)], l[(0, 1)]), (-1.0, 2.0, 0.0));
    /// ```
    #[must_use]
    pub fn lower_triangle(&self) -> Self {
        self.triangle(false)
    }

    /// The elements on the diagonal and above it, or on it and below it: of each row the stretch
    /// from the diagonal on or up to it, which a binary search bounds.
    fn triangle(&self, upper: bool) -> Self {
        let mut row_ptrs = Vec::with_capacity(self.rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..self.rows {
            row_ptrs.push(col_indices.len());
            let (begin, end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
            let row = &self.col_indices[begin..end];
            let (first, last) = if upper {
                (begin + row.partition_point(|&j| j < i), end)
            } else {
                (begin, begin + row.partition_point(|&j| j <= i))
            };
            col_indices.extend_from_slice(&self.col_indices[first..last]);
            values.extend_from_slice(&self.values[first..last]);
        }
        row_ptrs.push(col_indices.len());
        Self::from_canonical(self.rows, self.cols, row_ptrs, col_indices, values)
    }

    /// The matrix with its rows and columns in another order: the element `(i, j)` of the result
    /// is the element `(row_perm[i], col_perm[j])` of this one, `p[new] = old` as everywhere in
    /// this module. With permutation matrices `P` and `Q` that have a 1 at `(i, row_perm[i])` and
    /// at `(j, col_perm[j])`, the result is `P·A·Qᵀ`; for a square matrix and one permutation `p`,
    /// `permute(p, p)` is `P·A·Pᵀ`.
    ///
    /// The time is proportional to the stored elements, the rows and the columns.
    ///
    /// # Panics
    ///
    /// Panics if `row_perm` isn't a permutation of `0..rows` or `col_perm` one of `0..cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)]);
    /// let b = a.permute(&[1, 0], &[2, 0, 1]);
    /// // Row 0 of the result is the old row 1, and its column 0 the old column 2.
    /// assert_eq!(b, SparseMatrix::from_coo(2, 3, &[(0, 2, 3.0), (1, 0, 2.0), (1, 1, 1.0)]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn permute(&self, row_perm: &[usize], col_perm: &[usize]) -> Self {
        inverse_permutation(row_perm, self.rows, "rows");
        let new_column = inverse_permutation(col_perm, self.cols, "columns");
        let mut row_ptrs = Vec::with_capacity(self.rows + 1);
        let mut col_indices = Vec::with_capacity(self.values.len());
        let mut values = Vec::with_capacity(self.values.len());
        for &old in row_perm {
            row_ptrs.push(col_indices.len());
            let (begin, end) = (self.row_ptrs[old], self.row_ptrs[old + 1]);
            col_indices.extend(self.col_indices[begin..end].iter().map(|&j| new_column[j]));
            values.extend_from_slice(&self.values[begin..end]);
        }
        row_ptrs.push(col_indices.len());
        // The new columns of a row come in the order of the old ones.
        sort_rows(&row_ptrs, self.cols, &mut col_indices, &mut values);
        Self::from_canonical(self.rows, self.cols, row_ptrs, col_indices, values)
    }
}

/// The product `y·a` of a dense matrix by a sparse one: column `k` of the first meets row `k` of
/// the second, so every stored element `(k, j)` adds that column, times itself, to column `j` of
/// the product. The time is the rows of `y` for each stored element of `a`.
#[track_caller]
fn dense_times_sparse<T: Scalar>(y: &Matrix<T>, a: &SparseMatrix<T>) -> Matrix<T> {
    assert_eq!(y.cols(), a.rows, "the shapes don't fit for a product");
    let mut product = Matrix::zeros(y.rows(), a.cols);
    for k in 0..a.rows {
        let column = y.column(k);
        for m in a.row_ptrs[k]..a.row_ptrs[k + 1] {
            let x = a.values[m];
            for (p, &c) in product.column_mut(a.col_indices[m]).iter_mut().zip(column) {
                *p += c * x;
            }
        }
    }
    product
}

impl<T: Scalar> Add<&SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The sum, element by element.
    ///
    /// # Panics
    ///
    /// Panics if the matrices aren't of the same shape.
    #[track_caller]
    fn add(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(rhs, false)
    }
}

impl<T: Scalar> Add<SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn add(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(&rhs, false)
    }
}

impl<T: Scalar> Add<&SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn add(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(rhs, false)
    }
}

impl<T: Scalar> Add<SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn add(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(&rhs, false)
    }
}

impl<T: Scalar> AddAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn add_assign(&mut self, rhs: &SparseMatrix<T>) {
        *self = (*self).add_sub(rhs, false);
    }
}

impl<T: Scalar> AddAssign<SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn add_assign(&mut self, rhs: SparseMatrix<T>) {
        *self = (*self).add_sub(&rhs, false);
    }
}

impl<T: Scalar> Sub<&SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The difference, element by element.
    ///
    /// # Panics
    ///
    /// Panics if the matrices aren't of the same shape.
    #[track_caller]
    fn sub(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(rhs, true)
    }
}

impl<T: Scalar> Sub<SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn sub(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(&rhs, true)
    }
}

impl<T: Scalar> Sub<&SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn sub(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(rhs, true)
    }
}

impl<T: Scalar> Sub<SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn sub(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.add_sub(&rhs, true)
    }
}

impl<T: Scalar> SubAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn sub_assign(&mut self, rhs: &SparseMatrix<T>) {
        *self = (*self).add_sub(rhs, true);
    }
}

impl<T: Scalar> SubAssign<SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn sub_assign(&mut self, rhs: SparseMatrix<T>) {
        *self = (*self).add_sub(&rhs, true);
    }
}

impl<T: Scalar> Neg for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The negation, of the same pattern.
    fn neg(self) -> SparseMatrix<T> {
        self.map(|_, _, x| -x)
    }
}

impl<T: Scalar> Neg for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn neg(self) -> SparseMatrix<T> {
        self.map(|_, _, x| -x)
    }
}

impl<T: Scalar> Mul<&SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The matrix product of two sparse matrices.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't agree.
    #[track_caller]
    fn mul(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.product(rhs)
    }
}

impl<T: Scalar> Mul<SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn mul(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.product(&rhs)
    }
}

impl<T: Scalar> Mul<&SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn mul(self, rhs: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.product(rhs)
    }
}

impl<T: Scalar> Mul<SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    #[track_caller]
    fn mul(self, rhs: SparseMatrix<T>) -> SparseMatrix<T> {
        self.product(&rhs)
    }
}

impl<T: Scalar> MulAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn mul_assign(&mut self, rhs: &SparseMatrix<T>) {
        *self = (*self).product(rhs);
    }
}

impl<T: Scalar> MulAssign<SparseMatrix<T>> for SparseMatrix<T> {
    #[track_caller]
    fn mul_assign(&mut self, rhs: SparseMatrix<T>) {
        *self = (*self).product(&rhs);
    }
}

impl<T: Scalar> Mul<&SparseMatrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    /// The product of a dense and a sparse matrix, in the time the stored elements of the second
    /// cost for each row of the first.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't agree.
    #[track_caller]
    fn mul(self, rhs: &SparseMatrix<T>) -> Matrix<T> {
        dense_times_sparse(self, rhs)
    }
}

impl<T: Scalar> Mul<SparseMatrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: SparseMatrix<T>) -> Matrix<T> {
        dense_times_sparse(self, &rhs)
    }
}

impl<T: Scalar> Mul<&SparseMatrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: &SparseMatrix<T>) -> Matrix<T> {
        dense_times_sparse(&self, rhs)
    }
}

impl<T: Scalar> Mul<SparseMatrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: SparseMatrix<T>) -> Matrix<T> {
        dense_times_sparse(&self, &rhs)
    }
}

impl<T: Scalar> Mul<Matrix<T>> for &SparseMatrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        self.mul_matrix(&rhs)
    }
}

impl<T: Scalar> Mul<&Matrix<T>> for &SparseMatrix<T> {
    type Output = Matrix<T>;

    /// The product of a sparse and a dense matrix: [`SparseMatrix::mul_matrix`].
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't agree.
    #[track_caller]
    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        self.mul_matrix(rhs)
    }
}

impl<T: Scalar> Mul<&Matrix<T>> for SparseMatrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        self.mul_matrix(rhs)
    }
}

impl<T: Scalar> Mul<Matrix<T>> for SparseMatrix<T> {
    type Output = Matrix<T>;
    #[track_caller]
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        self.mul_matrix(&rhs)
    }
}

impl<T: Scalar> Mul<T> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The matrix with every stored element multiplied by the element:
    /// [`SparseMatrix::scale`].
    fn mul(self, rhs: T) -> SparseMatrix<T> {
        self.scale(rhs)
    }
}

impl<T: Scalar> Mul<&T> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: &T) -> SparseMatrix<T> {
        self.scale(*rhs)
    }
}

impl<T: Scalar> Mul<T> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: T) -> SparseMatrix<T> {
        self.scale(rhs)
    }
}

impl<T: Scalar> Mul<&T> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn mul(self, rhs: &T) -> SparseMatrix<T> {
        self.scale(*rhs)
    }
}

impl<T: Scalar> MulAssign<T> for SparseMatrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.scale(rhs);
    }
}

impl<T: Scalar> MulAssign<&T> for SparseMatrix<T> {
    fn mul_assign(&mut self, rhs: &T) {
        *self = self.scale(*rhs);
    }
}

impl<T: Scalar> Div<T> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    /// The matrix with every stored element divided by the element; the elements that aren't
    /// stored stay zeros, also when the divisor is zero.
    fn div(self, rhs: T) -> SparseMatrix<T> {
        self.map(|_, _, x| x / rhs)
    }
}

impl<T: Scalar> Div<T> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn div(self, rhs: T) -> SparseMatrix<T> {
        self.map(|_, _, x| x / rhs)
    }
}

impl<T: Scalar> DivAssign<T> for SparseMatrix<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = self.map(|_, _, x| x / rhs);
    }
}
