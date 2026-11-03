//! Transposes, products elementwise, row and column operations, the trace and powers.

use super::{Element, Matrix};

impl<T: Clone> Matrix<T> {
    /// The transpose `Aᵀ`, whose element in row `i` and column `j` is this one's in row `j` and
    /// column `i`.
    pub fn transpose(&self) -> Self {
        let rows = self.rows;
        Matrix::from_fn(self.cols, rows, |i, j| self.data[j + i * rows].clone())
    }
}

impl<T: Element> Matrix<T> {
    /// The conjugate transpose `A*`, the transpose with each element conjugated: the transpose
    /// itself for real elements.
    pub fn conjugate_transpose(&self) -> Self {
        let rows = self.rows;
        Matrix::from_fn(self.cols, rows, |i, j| self.data[j + i * rows].conj())
    }

    /// The product elementwise, Hadamard's `A ∘ B`.
    ///
    /// # Panics
    ///
    /// Panics if the matrices are not of the same shape.
    #[track_caller]
    pub fn hadamard(&self, other: &Matrix<T>) -> Matrix<T> {
        assert!(
            self.shape() == other.shape(),
            "the matrices are not of the same shape"
        );
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| T::mul(a, b))
                .collect(),
        }
    }

    /// Swaps rows `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if either is outside the matrix.
    #[track_caller]
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(
            a < self.rows && b < self.rows,
            "the row is outside the matrix"
        );
        if a != b {
            for j in 0..self.cols {
                self.data.swap(a + j * self.rows, b + j * self.rows);
            }
        }
    }

    /// Multiplies row `i` by `k`.
    ///
    /// # Panics
    ///
    /// Panics if the row is outside the matrix.
    #[track_caller]
    pub fn scale_row(&mut self, i: usize, k: &T) {
        assert!(i < self.rows, "the row is outside the matrix");
        for j in 0..self.cols {
            T::mul_assign(&mut self.data[i + j * self.rows], k);
        }
    }

    /// Adds `k` times row `source` to row `target`.
    ///
    /// # Panics
    ///
    /// Panics if either row is outside the matrix.
    #[track_caller]
    pub fn add_scaled_row(&mut self, target: usize, source: usize, k: &T) {
        assert!(
            target < self.rows && source < self.rows,
            "the row is outside the matrix"
        );
        for j in 0..self.cols {
            let x = T::mul(&self.data[source + j * self.rows], k);
            T::add_assign(&mut self.data[target + j * self.rows], &x);
        }
    }

    /// Swaps columns `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if either is outside the matrix.
    #[track_caller]
    pub fn swap_columns(&mut self, a: usize, b: usize) {
        assert!(
            a < self.cols && b < self.cols,
            "the column is outside the matrix"
        );
        if a != b {
            let (lo, hi) = (a.min(b), a.max(b));
            let (left, right) = self.data.split_at_mut(hi * self.rows);
            left[lo * self.rows..(lo + 1) * self.rows].swap_with_slice(&mut right[..self.rows]);
        }
    }

    /// Multiplies column `j` by `k`.
    ///
    /// # Panics
    ///
    /// Panics if the column is outside the matrix.
    #[track_caller]
    pub fn scale_column(&mut self, j: usize, k: &T) {
        for x in self.column_mut(j) {
            T::mul_assign(x, k);
        }
    }

    /// Adds `k` times column `source` to column `target`.
    ///
    /// # Panics
    ///
    /// Panics if either column is outside the matrix.
    #[track_caller]
    pub fn add_scaled_column(&mut self, target: usize, source: usize, k: &T) {
        assert!(
            target < self.cols && source < self.cols,
            "the column is outside the matrix"
        );
        let rows = self.rows;
        for i in 0..rows {
            let x = T::mul(&self.data[i + source * rows], k);
            T::add_assign(&mut self.data[i + target * rows], &x);
        }
    }

    /// The trace `tr(A)`, the sum of the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn trace(&self) -> T {
        assert!(self.is_square(), "the matrix is not square");
        let mut t = T::zero();
        for i in 0..self.rows {
            T::add_assign(&mut t, &self.data[i + i * self.rows]);
        }
        t
    }

    /// `Aⁿ`, by repeated squaring: about `2 log₂ n` products. `A⁰` is the identity.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn pow(&self, n: u32) -> Self {
        assert!(self.is_square(), "the matrix is not square");
        if n == 0 {
            return Matrix::identity(self.rows);
        }
        let mut base = self.clone();
        let mut n = n;
        while n & 1 == 0 {
            base = &base * &base;
            n >>= 1;
        }
        let mut r = base.clone();
        n >>= 1;
        while n > 0 {
            base = &base * &base;
            if n & 1 == 1 {
                r = &r * &base;
            }
            n >>= 1;
        }
        r
    }
}
