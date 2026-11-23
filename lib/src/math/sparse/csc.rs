//! The compressed sparse column form.
//!
//! It is the compressed sparse row form of the transpose, array for array: the values in column
//! order, the row index of each, and where each column begins. A [`CscMatrix`] is held as exactly
//! that — the compressed rows of its transpose — so that every operation on it is the one on rows
//! that is already there, read the other way: its columns are the rows of the transpose, its
//! product by a vector is the transpose's product by rows scattered instead of gathered, and its
//! invariant, that the row indices of a column strictly ascend and no zero is stored, is the one
//! the row form keeps.

use super::{CooMatrix, LinearOperator, SparseMatrix};
use crate::math::matrix::{Matrix, Scalar};
use core::fmt;
use core::ops::Index;

/// A matrix of `rows × cols` elements of the type `T` that stores only its nonzero elements, in
/// compressed sparse column form: the values in column order, the row index of each, and the
/// offset in those two where each column begins.
///
/// Within a column the row indices strictly ascend and no zero is stored, as within a row of a
/// [`SparseMatrix`]. A column is read in the time its elements take, which is what the methods
/// that work by columns want; a row is not, and [`to_csr`](Self::to_csr) is the way to have the
/// rows.
///
/// # Examples
///
/// ```
/// use a::math::sparse::CscMatrix;
///
/// // [1.5  0  0]
/// // [ 0   0  4]
/// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
/// assert_eq!(a.shape(), (2, 3));
/// assert_eq!(a.column(2).collect::<Vec<_>>(), [(1, 4.0)]);
/// assert_eq!(a[(0, 0)], 1.5);
/// assert_eq!(a[(1, 1)], 0.0);
/// assert_eq!(a.mul_vec(&[2.0, 1.0, 1.0]), [3.0, 4.0]);
/// ```
#[derive(Clone, PartialEq)]
pub struct CscMatrix<T> {
    /// The transpose in compressed sparse row form, whose three arrays are this matrix's own: the
    /// offsets of its rows are the offsets of the columns here, and its column indices the row
    /// indices.
    transposed: SparseMatrix<T>,
}

impl<T: Scalar> CscMatrix<T> {
    /// The matrix whose transpose, by rows, is `transposed`.
    pub(crate) fn from_transposed(transposed: SparseMatrix<T>) -> Self {
        Self { transposed }
    }

    /// The transpose by rows, which holds the same arrays.
    pub(crate) fn transposed(&self) -> &SparseMatrix<T> {
        &self.transposed
    }

    /// The empty matrix of `rows × cols`, which stores nothing.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::<f64>::zeros(4, 7);
    /// assert_eq!((a.rows(), a.cols(), a.nnz()), (4, 7, 0));
    /// assert_eq!(a.col_pointers(), [0; 8]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_transposed(SparseMatrix::zeros(cols, rows))
    }

    /// The matrix of the compressed sparse column arrays: `col_ptrs` of length `cols + 1` gives
    /// where each column begins in `row_indices` and `values`. The rows of a column may come in
    /// any order, equal ones are summed in the order given, and zeros are left out, so the result
    /// has the canonical form whatever it was given, as [`SparseMatrix::from_csr`] has by rows.
    ///
    /// Arrays that are canonical already are taken as they are, after one pass that checks them.
    /// [`into_parts`](Self::into_parts) is the way back.
    ///
    /// # Panics
    ///
    /// Panics if `col_ptrs` isn't `cols + 1` nondecreasing offsets from 0 to the number of
    /// elements, if `row_indices` and `values` differ in length, or if a row is outside the
    /// matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// // Column 0 given with its rows out of order.
    /// let a = CscMatrix::from_csc(3, 2, vec![0, 2, 3], vec![2, 0, 1], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.column(0).collect::<Vec<_>>(), [(0, 2.0), (2, 1.0)]);
    /// assert_eq!(a[(1, 1)], 3.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_csc(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert_eq!(
            Some(col_ptrs.len()),
            cols.checked_add(1),
            "there isn't an offset for each column and the end"
        );
        assert_eq!(
            row_indices.len(),
            values.len(),
            "the rows and the values differ in length"
        );
        assert_eq!(col_ptrs[0], 0, "the first offset isn't 0");
        assert_eq!(
            col_ptrs[cols],
            values.len(),
            "the last offset isn't the number of elements"
        );
        assert!(
            col_ptrs.windows(2).all(|w| w[0] <= w[1]),
            "the offsets decrease"
        );
        assert!(
            row_indices.iter().all(|&i| i < rows),
            "a row lies outside the matrix"
        );
        // Checked above with the words of this form; the row form sorts and sums.
        Self::from_transposed(SparseMatrix::from_csr(
            cols,
            rows,
            col_ptrs,
            row_indices,
            values,
        ))
    }

    /// The matrix of the dense `a`, its zeros left out, in the `rows · cols` steps it takes to
    /// read `a`.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::CscMatrix;
    ///
    /// let dense = Matrix::from_rows(&[[1.0, 0.0], [3.0, 2.0]]);
    /// let a = CscMatrix::from_matrix(&dense);
    /// assert_eq!(a.row_indices(), [0, 1, 1]);
    /// assert_eq!(a.values(), [1.0, 3.0, 2.0]);
    /// assert_eq!(a.to_matrix(), dense);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_matrix(a: &Matrix<T>) -> Self {
        // The dense matrix is held by columns, as this form is: one pass.
        let (rows, cols) = a.shape();
        let mut col_ptrs = Vec::with_capacity(super::offsets(cols));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        for j in 0..cols {
            col_ptrs.push(row_indices.len());
            for (i, &x) in a.column(j).iter().enumerate() {
                if !x.is_zero() {
                    row_indices.push(i);
                    values.push(x);
                }
            }
        }
        col_ptrs.push(row_indices.len());
        Self::from_transposed(SparseMatrix::from_canonical(
            cols,
            rows,
            col_ptrs,
            row_indices,
            values,
        ))
    }

    /// The number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.transposed.cols()
    }

    /// The number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.transposed.rows()
    }

    /// `(rows, cols)`.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// The number of stored elements, which is the number of nonzero ones.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.transposed.nnz()
    }

    /// Whether the matrix stores nothing: it has no nonzero element, whatever its shape.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transposed.is_empty()
    }

    /// The offsets of the columns in the stored elements, of length `cols + 1`: column `j` is the
    /// elements from `col_pointers()[j]` up to `col_pointers()[j + 1]`.
    #[inline]
    #[must_use]
    pub fn col_pointers(&self) -> &[usize] {
        self.transposed.row_pointers()
    }

    /// The row indices of the stored elements, ascending within a column.
    #[inline]
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        self.transposed.col_indices()
    }

    /// The stored elements, in the order of the row indices.
    #[inline]
    #[must_use]
    pub fn values(&self) -> &[T] {
        self.transposed.values()
    }

    /// The matrix taken apart, without a copy: `(rows, cols, col_pointers, row_indices, values)`,
    /// which [`from_csc`](Self::from_csc) puts together again.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// let (rows, cols, col_pointers, row_indices, values) = a.clone().into_parts();
    /// assert_eq!((rows, cols), (2, 3));
    /// assert_eq!(col_pointers, [0, 1, 1, 2]);
    /// assert_eq!(CscMatrix::from_csc(rows, cols, col_pointers, row_indices, values), a);
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        let (cols, rows, col_ptrs, row_indices, values) = self.transposed.into_parts();
        (rows, cols, col_ptrs, row_indices, values)
    }

    /// The stored elements of column `j`, paired with their rows, which ascend: read in the time
    /// they take.
    ///
    /// # Panics
    ///
    /// Panics if `j` isn't a column of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![2.0, 1.0, 3.0]);
    /// assert_eq!(a.column(0).collect::<Vec<_>>(), [(0, 2.0), (2, 1.0)]);
    /// assert_eq!(a.column(1).count(), 1);
    /// ```
    #[track_caller]
    pub fn column(&self, j: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        assert!(j < self.cols(), "the column is outside the matrix");
        self.transposed.row(j)
    }

    /// The offset of the element `(i, j)` in the stored elements, or `None` where the matrix stores
    /// none, or has no such cell: a binary search of column `j`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![2.0, 1.0, 3.0]);
    /// assert_eq!(a.find(2, 0), Some(1));
    /// assert_eq!(a.values()[1], 1.0);
    /// assert_eq!(a.find(1, 0), None);
    /// ```
    #[must_use]
    pub fn find(&self, i: usize, j: usize) -> Option<usize> {
        self.transposed.find(j, i)
    }

    /// The transpose, in the same form, in time proportional to the stored elements, the rows and
    /// the columns.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// let t = a.transpose();
    /// assert_eq!(t.shape(), (3, 2));
    /// assert_eq!(t[(2, 1)], 4.0);
    /// assert_eq!(t.transpose(), a);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn transpose(&self) -> Self {
        Self::from_transposed(self.transposed.transpose())
    }

    /// The conjugate transpose, in the same form; the transpose of a real matrix.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(1, 2, vec![0, 0, 1], vec![0], vec![Complex::new(1.0, 2.0)]);
    /// assert_eq!(a.conjugate_transpose()[(1, 0)], Complex::new(1.0, -2.0));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn conjugate_transpose(&self) -> Self {
        Self::from_transposed(self.transposed.conjugate_transpose())
    }

    /// The same matrix in compressed sparse row form, in time proportional to the stored
    /// elements, the rows and the columns.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{CscMatrix, SparseMatrix};
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// assert_eq!(a.to_csr(), SparseMatrix::from_coo(2, 3, &[(0, 0, 1.5), (1, 2, 4.0)]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_csr(&self) -> SparseMatrix<T> {
        self.transposed.transpose()
    }

    /// The same matrix in coordinate form: its stored elements as triplets, in column order, in
    /// time proportional to them and to the columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// assert_eq!(a.to_coo().iter().collect::<Vec<_>>(), [(0, 0, 1.5), (1, 2, 4.0)]);
    /// ```
    #[must_use]
    pub fn to_coo(&self) -> CooMatrix<T> {
        self.transposed.to_coo().into_transpose()
    }

    /// The dense matrix of the same elements, in `rows · cols` steps and one for each stored
    /// element.
    ///
    /// # Panics
    ///
    /// Panics if `rows · cols` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 2, vec![0, 1, 2], vec![1, 0], vec![3.0, 7.0]);
    /// assert_eq!(a.to_matrix().as_slice(), &[0.0, 3.0, 7.0, 0.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_matrix(&self) -> Matrix<T> {
        let (rows, cols) = self.shape();
        let mut dense = Matrix::zeros(rows, cols);
        let (col_ptrs, row_indices, values) =
            (self.col_pointers(), self.row_indices(), self.values());
        for j in 0..cols {
            let column = dense.column_mut(j);
            for k in col_ptrs[j]..col_ptrs[j + 1] {
                column[row_indices[k]] = values[k];
            }
        }
        dense
    }

    /// `y = a·x` written into `y`, for `x` of length `cols` and `y` of length `rows`: one pass over
    /// the stored elements, each column, times its entry of `x`, added to `y`.
    ///
    /// # Panics
    ///
    /// Panics if `x` and `y` aren't of the right lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// let mut y = [0.0; 2];
    /// a.mul_vec_into(&[2.0, 1.0, 1.0], &mut y);
    /// assert_eq!(y, [3.0, 4.0]);
    /// ```
    #[track_caller]
    pub fn mul_vec_into(&self, x: &[T], y: &mut [T]) {
        self.transposed.mul_vec_transpose_into(x, y);
    }

    /// `a·x`, for `x` of length `cols`.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// assert_eq!(a.mul_vec(&[2.0, 1.0, 1.0]), [3.0, 4.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec(&self, x: &[T]) -> Vec<T> {
        self.transposed.mul_vec_transpose(x)
    }

    /// `y = aᵀ·x` written into `y`, for `x` of length `rows` and `y` of length `cols`, without the
    /// transpose being formed: each entry of `y` is a column of the matrix times `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` and `y` aren't of the right lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// let mut y = [0.0; 3];
    /// a.mul_vec_transpose_into(&[2.0, 1.0], &mut y);
    /// assert_eq!(y, [3.0, 0.0, 4.0]);
    /// ```
    #[track_caller]
    pub fn mul_vec_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.transposed.mul_vec_into(x, y);
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
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.5, 4.0]);
    /// assert_eq!(a.mul_vec_transpose(&[2.0, 1.0]), [3.0, 0.0, 4.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec_transpose(&self, x: &[T]) -> Vec<T> {
        self.transposed.mul_vec(x)
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
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(1, 2, vec![0, 0, 1], vec![0], vec![Complex::new(1.0, 2.0)]);
    /// let mut y = [Complex::ZERO; 2];
    /// a.mul_vec_adjoint_into(&[Complex::new(0.0, 1.0)], &mut y);
    /// // conj(1 + 2i)·i = 2 + i
    /// assert_eq!(y, [Complex::ZERO, Complex::new(2.0, 1.0)]);
    /// ```
    #[track_caller]
    pub fn mul_vec_adjoint_into(&self, x: &[T], y: &mut [T]) {
        self.transposed.mul_vec_conjugate_into(x, y);
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
    /// use a::math::sparse::CscMatrix;
    ///
    /// let a = CscMatrix::from_csc(1, 2, vec![0, 0, 1], vec![0], vec![Complex::new(1.0, 2.0)]);
    /// let x = [Complex::new(0.0, 1.0)];
    /// assert_eq!(a.mul_vec_adjoint(&x), a.conjugate_transpose().mul_vec(&x));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn mul_vec_adjoint(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.cols()];
        self.mul_vec_adjoint_into(x, &mut y);
        y
    }
}

impl<T: Scalar> Index<(usize, usize)> for CscMatrix<T> {
    type Output = T;

    /// The element in row `i` and column `j`, the zero of the type where the matrix stores none.
    ///
    /// # Panics
    ///
    /// Panics if the pair is outside the matrix.
    #[track_caller]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.transposed[(j, i)]
    }
}

impl<T: Scalar> Default for CscMatrix<T> {
    /// The `0 × 0` matrix, which stores nothing.
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl<T: fmt::Debug> fmt::Debug for CscMatrix<T> {
    /// Writes the shape and the stored elements in column order, each under its row and column:
    /// `2 × 3 {(1, 0): -2.0, (0, 2): 1.5}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = &self.transposed;
        write!(f, "{} × {} ", t.cols, t.rows)?;
        let mut map = f.debug_map();
        for j in 0..t.rows {
            for k in t.row_ptrs[j]..t.row_ptrs[j + 1] {
                map.entry(&(t.col_indices[k], j), &t.values[k]);
            }
        }
        map.finish()
    }
}

impl<T: Scalar> LinearOperator<T> for CscMatrix<T> {
    #[inline]
    fn rows(&self) -> usize {
        CscMatrix::rows(self)
    }

    #[inline]
    fn cols(&self) -> usize {
        CscMatrix::cols(self)
    }

    #[inline]
    #[track_caller]
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.mul_vec_into(x, y);
    }
}

impl<T: Scalar> From<&CscMatrix<T>> for SparseMatrix<T> {
    /// The same matrix by rows: [`CscMatrix::to_csr`].
    fn from(a: &CscMatrix<T>) -> Self {
        a.to_csr()
    }
}

impl<T: Scalar> From<CscMatrix<T>> for SparseMatrix<T> {
    /// The same matrix by rows: [`CscMatrix::to_csr`].
    fn from(a: CscMatrix<T>) -> Self {
        a.to_csr()
    }
}

impl<T: Scalar> From<&CscMatrix<T>> for CooMatrix<T> {
    /// The stored elements as triplets: [`CscMatrix::to_coo`].
    fn from(a: &CscMatrix<T>) -> Self {
        a.to_coo()
    }
}

impl<T: Scalar> From<CscMatrix<T>> for CooMatrix<T> {
    /// The stored elements as triplets, as [`CscMatrix::to_coo`] gives them, the rows and the
    /// values taken over without a copy.
    fn from(a: CscMatrix<T>) -> Self {
        CooMatrix::from(a.transposed).into_transpose()
    }
}
