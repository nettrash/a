//! Matrices: a [`Matrix<T>`] is a rectangular array of elements of the type `T`, which is `f32`,
//! `f64`, [`Complex<f32>`](crate::math::complex::Complex),
//! [`Complex<f64>`](crate::math::complex::Complex), [`Number`](crate::math::numbers::Number) or
//! [`Rational`](crate::math::rational::Rational).

mod basic;
mod blas;
mod cholesky;
mod eigen;
mod element;
mod exact;
mod functions;
mod gemm;
mod hermitian;
mod householder;
mod lu;
mod ops;
mod properties;
mod qr;
mod rational;
mod schur;
mod svd;
mod tridiagonal;

pub use cholesky::Cholesky;
pub use eigen::{Eigen, HermitianEigen};
pub use element::{Element, Scalar};
pub use lu::Lu;
pub use properties::Norm;
pub use qr::Qr;
pub use schur::Schur;
pub use svd::Svd;

use core::fmt;
use core::ops::{Index, IndexMut};

/// A matrix of `rows × cols` elements of the type `T`, held column after column in one vector.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
/// let b = Matrix::identity(2);
/// assert_eq!(&a * &b, a);
/// assert_eq!(a[(1, 0)], 3.0);
/// assert_eq!(a.transpose(), Matrix::from_rows(&[[1.0, 3.0], [2.0, 4.0]]));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Matrix<T> {
    /// The number of rows.
    rows: usize,
    /// The number of columns.
    cols: usize,
    /// The elements, column after column.
    data: Vec<T>,
}

/// The number of elements of a `rows × cols` matrix, which a release build would otherwise let
/// wrap around and leave a matrix of nonempty dimensions with no elements.
///
/// # Panics
///
/// Panics if it overflows `usize`.
#[track_caller]
fn size(rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols)
        .expect("the matrix has more elements than a vector can hold")
}

impl<T> Matrix<T> {
    /// The matrix of `rows × cols` elements given column after column.
    ///
    /// # Panics
    ///
    /// Panics if `data` doesn't hold `rows · cols` elements.
    #[track_caller]
    pub fn from_column_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            Some(data.len()),
            rows.checked_mul(cols),
            "the data doesn't hold rows × cols elements"
        );
        Matrix { rows, cols, data }
    }

    /// The matrix of `rows × cols` elements given row after row.
    ///
    /// # Panics
    ///
    /// Panics if `data` doesn't hold `rows · cols` elements.
    #[track_caller]
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Matrix::from_column_major(cols, rows, data).into_transpose()
    }

    /// The matrix whose element in row `i` and column `j` is `f(i, j)`, computed column after
    /// column.
    ///
    /// # Panics
    ///
    /// Panics if `rows · cols` overflows `usize`.
    #[track_caller]
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(size(rows, cols));
        for j in 0..cols {
            // A column at a time, from a range whose length is known, so that the vector is not
            // checked for room at each element.
            data.extend((0..rows).map(|i| f(i, j)));
        }
        Matrix { rows, cols, data }
    }

    /// The number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Whether it has as many rows as columns.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Whether it has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The element in row `i` and column `j`, or `None` outside the matrix.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.rows && j < self.cols {
            Some(&self.data[i + j * self.rows])
        } else {
            None
        }
    }

    /// The element in row `i` and column `j`, to change, or `None` outside the matrix.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i < self.rows && j < self.cols {
            Some(&mut self.data[i + j * self.rows])
        } else {
            None
        }
    }

    /// Column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j` is not below the number of columns.
    #[inline]
    #[track_caller]
    pub fn column(&self, j: usize) -> &[T] {
        assert!(j < self.cols, "the column is outside the matrix");
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Column `j`, to change.
    ///
    /// # Panics
    ///
    /// Panics if `j` is not below the number of columns.
    #[inline]
    #[track_caller]
    pub fn column_mut(&mut self, j: usize) -> &mut [T] {
        assert!(j < self.cols, "the column is outside the matrix");
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// The elements of row `i`, from the left.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not below the number of rows.
    #[track_caller]
    pub fn row(&self, i: usize) -> impl ExactSizeIterator<Item = &T> + '_ {
        assert!(i < self.rows, "the row is outside the matrix");
        (0..self.cols).map(move |j| &self.data[i + j * self.rows])
    }

    /// The elements, column after column.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The elements, column after column, to change.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The vector of the elements, column after column.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// The matrix of `f` of each element.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Matrix<U> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect(),
        }
    }

    /// The transpose, moving the elements.
    fn into_transpose(self) -> Self {
        let (rows, cols) = (self.rows, self.cols);
        if rows <= 1 || cols <= 1 {
            return Matrix {
                rows: cols,
                cols: rows,
                data: self.data,
            };
        }
        let mut slots: Vec<Option<T>> = self.data.into_iter().map(Some).collect();
        Matrix::from_fn(cols, rows, |i, j| slots[j + i * rows].take().unwrap())
    }
}

impl<T: Clone> Matrix<T> {
    /// The matrix with the given rows.
    ///
    /// # Panics
    ///
    /// Panics if the rows are not all of the same length.
    #[track_caller]
    pub fn from_rows<R: AsRef<[T]>>(rows: &[R]) -> Self {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        assert!(
            rows.iter().all(|r| r.as_ref().len() == cols),
            "the rows are not all of the same length"
        );
        Matrix::from_fn(rows.len(), cols, |i, j| rows[i].as_ref()[j].clone())
    }

    /// The matrix with the given columns.
    ///
    /// # Panics
    ///
    /// Panics if the columns are not all of the same length.
    #[track_caller]
    pub fn from_columns<C: AsRef<[T]>>(columns: &[C]) -> Self {
        let rows = columns.first().map_or(0, |c| c.as_ref().len());
        assert!(
            columns.iter().all(|c| c.as_ref().len() == rows),
            "the columns are not all of the same length"
        );
        let data = columns
            .iter()
            .flat_map(|c| c.as_ref().iter().cloned())
            .collect();
        Matrix {
            rows,
            cols: columns.len(),
            data,
        }
    }

    /// The elements of the main diagonal, from the top left.
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.rows.min(self.cols))
            .map(|i| self.data[i + i * self.rows].clone())
            .collect()
    }
}

impl<T: Element> Matrix<T> {
    /// The `rows × cols` matrix of zeros.
    ///
    /// # Panics
    ///
    /// Panics if `rows · cols` overflows `usize`.
    #[track_caller]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        // Filled at once, rather than element by element: for floats the allocator's zeroed
        // pages, which took 1.4 µs for `256 × 256` against 69 by `from_fn`.
        Matrix {
            rows,
            cols,
            data: vec![T::zero(); size(rows, cols)],
        }
    }

    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        Matrix::from_diagonal(&vec![T::one(); n])
    }

    /// The square matrix with `diagonal` on its main diagonal and zeros elsewhere.
    pub fn from_diagonal(diagonal: &[T]) -> Self {
        let n = diagonal.len();
        let mut a = Matrix::zeros(n, n);
        for (i, d) in diagonal.iter().enumerate() {
            a.data[i + i * n] = d.clone();
        }
        a
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    /// The element in row `i` and column `j`.
    ///
    /// # Panics
    ///
    /// Panics outside the matrix.
    #[inline]
    #[track_caller]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(
            i < self.rows && j < self.cols,
            "the index is outside the matrix"
        );
        &self.data[i + j * self.rows]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    /// The element in row `i` and column `j`, to change.
    ///
    /// # Panics
    ///
    /// Panics outside the matrix.
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(
            i < self.rows && j < self.cols,
            "the index is outside the matrix"
        );
        &mut self.data[i + j * self.rows]
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    /// Writes the rows on lines of their own, each element as `Display` writes it with the
    /// formatter's precision, right-aligned in its column, the columns two spaces apart.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text: Vec<String> = self
            .data
            .iter()
            .map(|x| match f.precision() {
                Some(p) => format!("{x:.p$}"),
                None => format!("{x}"),
            })
            .collect();
        let widths: Vec<usize> = (0..self.cols)
            .map(|j| {
                let column = &text[j * self.rows..(j + 1) * self.rows];
                column.iter().map(|s| s.chars().count()).max().unwrap_or(0)
            })
            .collect();
        for i in 0..self.rows {
            if i > 0 {
                writeln!(f)?;
            }
            for (j, width) in widths.iter().enumerate() {
                let gap = if j == 0 { "" } else { "  " };
                write!(f, "{gap}{:>width$}", text[i + j * self.rows])?;
            }
        }
        Ok(())
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    /// Writes the rows as lists: `[[1, 2], [3, 4]]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.rows).map(|i| DebugRow(self, i)))
            .finish()
    }
}

/// A row of a matrix for [`fmt::Debug`].
struct DebugRow<'a, T>(&'a Matrix<T>, usize);

impl<T: fmt::Debug> fmt::Debug for DebugRow<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.row(self.1)).finish()
    }
}
