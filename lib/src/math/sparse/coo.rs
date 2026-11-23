//! The coordinate form: a list of `(row, column, value)` triplets.
//!
//! It is the form a matrix is assembled in — a finite element code adds the contribution of one
//! element after another to whatever cells it touches — and the form a Matrix Market file is read
//! into. The triplets come in any order, a cell may have several, which are summed when the matrix
//! is converted, and nothing is sorted or searched while they are pushed.

use super::{CscMatrix, SparseMatrix};
use crate::math::matrix::{Matrix, Scalar};
use core::fmt;

/// A matrix of `rows × cols` elements of the type `T` held as triplets `(row, column, value)`, in
/// the order they were given in.
///
/// A cell may have several triplets: the matrix has their sum there, which
/// [`to_csr`](Self::to_csr), [`to_csc`](Self::to_csc) and [`to_matrix`](Self::to_matrix) form, in
/// the order the triplets are in. A triplet may hold a zero, which those conversions drop.
///
/// Two matrices in this form are equal, by `==`, when they have the same shape and the same
/// triplets in the same order; the matrices they stand for are compared by converting them.
///
/// # Examples
///
/// ```
/// use a::math::sparse::CooMatrix;
///
/// let mut a = CooMatrix::new(2, 3);
/// a.push(1, 2, 4.0);
/// a.push(0, 0, 1.0);
/// a.push(1, 2, -1.5);
/// assert_eq!(a.nnz(), 3);
/// let csr = a.to_csr();
/// assert_eq!(csr.nnz(), 2);
/// assert_eq!(csr[(1, 2)], 2.5);
/// ```
#[derive(Clone, PartialEq)]
pub struct CooMatrix<T> {
    /// The number of rows.
    rows: usize,
    /// The number of columns.
    cols: usize,
    /// The row of each triplet.
    row_indices: Vec<usize>,
    /// The column of each triplet.
    col_indices: Vec<usize>,
    /// The value of each triplet.
    values: Vec<T>,
}

impl<T: Scalar> CooMatrix<T> {
    /// The matrix of `rows × cols` with no triplets, which is the matrix of zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::<f64>::new(3, 4);
    /// assert_eq!((a.rows(), a.cols(), a.nnz()), (3, 4, 0));
    /// ```
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_capacity(rows, cols, 0)
    }

    /// The matrix of `rows × cols` with no triplets and room for `capacity` of them before its
    /// vectors grow.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let mut a = CooMatrix::with_capacity(2, 2, 4);
    /// a.push(0, 1, 1.0);
    /// assert_eq!(a.nnz(), 1);
    /// ```
    #[must_use]
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// The matrix of `rows × cols` whose triplets are `(row_indices[k], col_indices[k],
    /// values[k])`: the three vectors are taken as they are, without a copy.
    /// [`into_parts`](Self::into_parts) is the way back.
    ///
    /// # Panics
    ///
    /// Panics if the three vectors differ in length or a triplet lies outside the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(2, 2, vec![0, 1, 1], vec![1, 0, 0], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.to_matrix().as_slice(), &[0.0, 5.0, 1.0, 0.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert!(
            row_indices.len() == values.len() && col_indices.len() == values.len(),
            "the rows, the columns and the values differ in length"
        );
        assert!(
            row_indices.iter().all(|&i| i < rows) && col_indices.iter().all(|&j| j < cols),
            "an entry lies outside the matrix"
        );
        Self {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
        }
    }

    /// The matrix of triplets known to be of one length and inside the matrix, which the other
    /// forms hand over when they convert themselves.
    pub(crate) fn from_parts_unchecked(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        debug_assert!(row_indices.len() == values.len() && col_indices.len() == values.len());
        debug_assert!(row_indices.iter().all(|&i| i < rows));
        debug_assert!(col_indices.iter().all(|&j| j < cols));
        Self {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
        }
    }

    /// The triplets of the nonzero elements of the dense `a`, in row order.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::CooMatrix;
    ///
    /// let dense = Matrix::from_rows(&[[0.0, 1.0], [2.0, 0.0]]);
    /// let a = CooMatrix::from_matrix(&dense);
    /// assert_eq!(a.iter().collect::<Vec<_>>(), [(0, 1, 1.0), (1, 0, 2.0)]);
    /// assert_eq!(a.to_matrix(), dense);
    /// ```
    #[must_use]
    pub fn from_matrix(a: &Matrix<T>) -> Self {
        SparseMatrix::from_matrix(a).into()
    }

    /// Adds the triplet `(i, j, x)`: `x` is added to the element in row `i` and column `j`.
    ///
    /// # Panics
    ///
    /// Panics if the cell is outside the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let mut a = CooMatrix::new(2, 2);
    /// a.push(1, 0, 3.0);
    /// a.push(1, 0, 4.0);
    /// assert_eq!(a.to_csr()[(1, 0)], 7.0);
    /// ```
    #[track_caller]
    pub fn push(&mut self, i: usize, j: usize, x: T) {
        assert!(
            i < self.rows && j < self.cols,
            "the entry lies outside the matrix"
        );
        self.row_indices.push(i);
        self.col_indices.push(j);
        self.values.push(x);
    }

    /// The number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of triplets stored, which is more than the number of nonzero elements when a
    /// cell has several or a triplet holds a zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(1, 1, vec![0, 0], vec![0, 0], vec![1.0, 2.0]);
    /// assert_eq!(a.nnz(), 2);
    /// assert_eq!(a.to_csr().nnz(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether no triplet is stored.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The rows of the triplets.
    #[inline]
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// The columns of the triplets.
    #[inline]
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// The values of the triplets.
    #[inline]
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The matrix taken apart, without a copy: `(rows, cols, row_indices, col_indices, values)`,
    /// which [`from_triplets`](Self::from_triplets) puts together again.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let mut a = CooMatrix::new(2, 2);
    /// a.push(1, 0, 3.0);
    /// let (rows, cols, row_indices, col_indices, values) = a.into_parts();
    /// assert_eq!((rows, cols), (2, 2));
    /// assert_eq!((row_indices, col_indices, values), (vec![1], vec![0], vec![3.0]));
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.rows,
            self.cols,
            self.row_indices,
            self.col_indices,
            self.values,
        )
    }

    /// The triplets `(row, column, value)`, in the order they are stored in.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let mut a = CooMatrix::new(2, 2);
    /// a.push(1, 0, 3.0);
    /// a.push(0, 1, 4.0);
    /// assert_eq!(a.iter().collect::<Vec<_>>(), [(1, 0, 3.0), (0, 1, 4.0)]);
    /// ```
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (usize, usize, T)> + '_ {
        (0..self.values.len()).map(|k| (self.row_indices[k], self.col_indices[k], self.values[k]))
    }

    /// The transpose: the same triplets with their rows and columns exchanged, copied.
    /// [`into_transpose`](Self::into_transpose) does it without the copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(2, 3, vec![0], vec![2], vec![5.0]);
    /// let t = a.transpose();
    /// assert_eq!(t.shape(), (3, 2));
    /// assert_eq!(t.iter().collect::<Vec<_>>(), [(2, 0, 5.0)]);
    /// ```
    #[must_use]
    pub fn transpose(&self) -> Self {
        self.clone().into_transpose()
    }

    /// The transpose, at no cost: the vector of the rows and the vector of the columns change
    /// places.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(2, 3, vec![0], vec![2], vec![5.0]);
    /// assert_eq!(a.clone().into_transpose(), a.transpose());
    /// ```
    #[must_use]
    pub fn into_transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            row_indices: self.col_indices,
            col_indices: self.row_indices,
            values: self.values,
        }
    }

    /// The same matrix in compressed sparse row form: the triplets of a cell summed, in the order
    /// they are in, and no zero stored. The time and the memory are proportional to the triplets
    /// and the rows.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{CooMatrix, SparseMatrix};
    ///
    /// let a = CooMatrix::from_triplets(2, 2, vec![1, 0, 1], vec![1, 0, 1], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.to_csr(), SparseMatrix::from_diagonal(&[2.0, 4.0]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_csr(&self) -> SparseMatrix<T> {
        SparseMatrix::assemble(
            self.rows,
            self.cols,
            self.values.len(),
            |k| self.row_indices[k],
            |k| self.col_indices[k],
            |k| self.values[k],
        )
    }

    /// The same matrix in compressed sparse column form: the triplets of a cell summed, in the
    /// order they are in, and no zero stored. The time and the memory are proportional to the
    /// triplets and the columns.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(2, 2, vec![1, 0, 1], vec![1, 0, 1], vec![1.0, 2.0, 3.0]);
    /// let c = a.to_csc();
    /// assert_eq!(c.col_pointers(), [0, 1, 2]);
    /// assert_eq!(c.values(), [2.0, 4.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_csc(&self) -> CscMatrix<T> {
        // The columns of the matrix are the rows of its transpose.
        CscMatrix::from_transposed(SparseMatrix::assemble(
            self.cols,
            self.rows,
            self.values.len(),
            |k| self.col_indices[k],
            |k| self.row_indices[k],
            |k| self.values[k],
        ))
    }

    /// The dense matrix of the same elements, the triplets of a cell summed in the order they are
    /// in: `rows · cols` steps and one for each triplet.
    ///
    /// # Panics
    ///
    /// Panics if `rows · cols` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::CooMatrix;
    ///
    /// let a = CooMatrix::from_triplets(2, 2, vec![1, 0, 1], vec![1, 0, 1], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.to_matrix(), Matrix::from_diagonal(&[2.0, 4.0]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_matrix(&self) -> Matrix<T> {
        let mut dense = Matrix::zeros(self.rows, self.cols);
        let data = dense.as_mut_slice();
        for (i, j, x) in self.iter() {
            data[i + j * self.rows] += x;
        }
        dense
    }
}

impl<T: Scalar> Default for CooMatrix<T> {
    /// The `0 × 0` matrix, with no triplets.
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl<T: fmt::Debug> fmt::Debug for CooMatrix<T> {
    /// Writes the shape and the triplets in the order they are stored in:
    /// `2 × 3 [(1, 2, 4.0), (0, 0, 1.0)]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {} ", self.rows, self.cols)?;
        f.debug_list()
            .entries(
                (0..self.values.len())
                    .map(|k| (self.row_indices[k], self.col_indices[k], &self.values[k])),
            )
            .finish()
    }
}

impl<T: Scalar> From<&CooMatrix<T>> for SparseMatrix<T> {
    /// The same matrix by rows: [`CooMatrix::to_csr`].
    fn from(a: &CooMatrix<T>) -> Self {
        a.to_csr()
    }
}

impl<T: Scalar> From<CooMatrix<T>> for SparseMatrix<T> {
    /// The same matrix by rows: [`CooMatrix::to_csr`].
    fn from(a: CooMatrix<T>) -> Self {
        a.to_csr()
    }
}

impl<T: Scalar> From<&CooMatrix<T>> for CscMatrix<T> {
    /// The same matrix by columns: [`CooMatrix::to_csc`].
    fn from(a: &CooMatrix<T>) -> Self {
        a.to_csc()
    }
}

impl<T: Scalar> From<CooMatrix<T>> for CscMatrix<T> {
    /// The same matrix by columns: [`CooMatrix::to_csc`].
    fn from(a: CooMatrix<T>) -> Self {
        a.to_csc()
    }
}
