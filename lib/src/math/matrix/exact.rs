//! Exact algorithms on matrices of numbers: determinants, ranks, characteristic polynomials,
//! adjugates, inverses and null spaces, without fractions.

use super::{Matrix, rational};
use crate::math::numbers::Number;

impl Matrix<Number> {
    /// The determinant, exactly, by Bareiss's fraction-free elimination: each step's elements
    /// are minors of the matrix, `(a_ij·a_kk - a_ik·a_kj) / p` with the previous pivot `p`
    /// dividing exactly, so the numbers grow no longer than the determinant. About `n³/3`
    /// steps, each two products and an exact division, in registers while the minors fit in
    /// 63 bits; from 28 rows (22 with an element past 63 bits) by Dixon's lifting and residues
    /// modulo primes (see [`Matrix::<Rational>::det`](Matrix::<crate::math::rational::Rational>::det)).
    /// The determinant of the `0 × 0` matrix is 1, and the result is in the byte order of the
    /// top left element.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, -1, 0], [-1, 2, -1], [0, -1, 2]]).map(|&x| Number::from(x));
    /// assert_eq!(a.det(), 4);
    /// ```
    #[track_caller]
    pub fn det(&self) -> Number {
        assert!(self.is_square(), "the matrix is not square");
        rational::det_of_integers(self)
    }

    /// The rank, exactly: the number of pivots of Bareiss's fraction-free elimination.
    pub fn rank(&self) -> usize {
        rational::rank_of_integers(self)
    }

    /// The coefficients of the characteristic polynomial `det(x·I - A)`, from `xⁿ` down,
    /// exactly, by Berkowitz's method: the polynomial of each leading block from the one before
    /// by a Toeplitz matrix of `1`, `-a_kk` and `-R·A_k^j·C`, without a division. About `n⁴/2`
    /// products.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [1, 2]]).map(|&x| Number::from(x));
    /// assert_eq!(a.characteristic_polynomial(), [1, -4, 3].map(Number::from));
    /// ```
    #[track_caller]
    pub fn characteristic_polynomial(&self) -> Vec<Number> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let mut p = vec![Number::from(1u8)];
        for r in 0..n {
            // `t = (1, -a_rr, -R·C, -R·A_r·C, …)` for the row `R` and the column `C` beside the
            // leading `r × r` block `A_r`.
            let mut t = Vec::with_capacity(r + 2);
            t.push(Number::from(1u8));
            t.push(-&self[(r, r)]);
            let mut v: Vec<Number> = (0..r).map(|i| self[(i, r)].clone()).collect();
            for j in 0..r {
                let mut q = Number::default();
                for (k, x) in v.iter().enumerate() {
                    if !x.is_zero() {
                        q += &self[(r, k)] * x;
                    }
                }
                t.push(-q);
                if j + 1 < r {
                    v = (0..r)
                        .map(|i| {
                            let mut s = Number::default();
                            for (k, x) in v.iter().enumerate() {
                                if !x.is_zero() {
                                    s += &self[(i, k)] * x;
                                }
                            }
                            s
                        })
                        .collect();
                }
            }
            // `p ← T·p`, `T` lower triangular Toeplitz with the first column `t`.
            let next = (0..r + 2)
                .map(|i| {
                    let mut s = Number::default();
                    for (j, c) in p.iter().enumerate().take(i + 1) {
                        if !c.is_zero() && !t[i - j].is_zero() {
                            s += &t[i - j] * c;
                        }
                    }
                    s
                })
                .collect();
            p = next;
        }
        p
    }

    /// The adjugate `adj(A)`, the transpose of the matrix of cofactors, with
    /// `A·adj(A) = adj(A)·A = det(A)·I`, for singular matrices too: by Cayley and Hamilton,
    /// `(-1)ⁿ⁻¹·(Aⁿ⁻¹ + c₁·Aⁿ⁻² + … + cₙ₋₁·I)` for the characteristic polynomial's `c`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn adjugate(&self) -> Matrix<Number> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        if n == 0 {
            return Matrix::zeros(0, 0);
        }
        let c = self.characteristic_polynomial();
        let mut m = Matrix::<Number>::identity(n);
        for c in &c[1..n] {
            m = self * &m;
            for i in 0..n {
                m[(i, i)] += c;
            }
        }
        if n.is_multiple_of(2) { -m } else { m }
    }

    /// The inverse as the adjugate and the determinant, `A⁻¹ = adj(A) / det(A)`, exactly, or
    /// `None` when the determinant is zero: Bareiss's elimination of `[A | I]` and the
    /// fraction-free back substitution of [`solve`](Matrix::<Number>::solve), whose integers
    /// `D·A⁻¹`, `D = ±det(A)` the last pivot, are the adjugate up to the sign of the row swaps.
    /// [`inverse_rational`](Matrix::<Number>::inverse_rational) gives `A⁻¹` as rationals. The
    /// results are in the byte order of the top left element.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [5, 3]]).map(|&x| Number::from(x));
    /// let (adj, det) = a.inverse().unwrap();
    /// assert_eq!(det, 1);
    /// assert_eq!(adj, Matrix::from_rows(&[[3, -1], [-5, 2]]).map(|&x| Number::from(x)));
    /// ```
    #[track_caller]
    pub fn inverse(&self) -> Option<(Matrix<Number>, Number)> {
        assert!(self.is_square(), "the matrix is not square");
        rational::inverse_of_integers(self)
    }

    /// A basis of the null space `{x : A·x = 0}` over the rationals, as the columns of an
    /// `n × (n - rank)` matrix of integers: one vector for each column without a pivot in the
    /// fraction-free echelon form, its free coordinate positive, the other free coordinates 0
    /// and its elements without a common factor, from a fraction-free back substitution; in the
    /// byte order of the top left element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[1, 2, 3], [2, 4, 6]]).map(|&x| Number::from(x));
    /// let null = a.null_space();
    /// assert_eq!(null.shape(), (3, 2));
    /// assert!((&a * &null).as_slice().iter().all(|x| x.is_zero()));
    /// ```
    pub fn null_space(&self) -> Matrix<Number> {
        rational::null_space_of_integers(self)
    }
}
