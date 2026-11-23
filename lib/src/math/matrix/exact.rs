//! Exact algorithms on matrices of numbers: determinants, ranks, characteristic polynomials,
//! adjugates, inverses and null spaces, without fractions.

use super::{Matrix, rational};
use crate::math::numbers::Number;

impl Matrix<Number> {
    /// The determinant, exactly, by one of three methods, chosen by the order `n` and the length
    /// of the longest element where each was measured to be the fastest on one machine (the
    /// edges for elements longer than 262 144 bits are extrapolated from those measurements):
    ///
    /// - Bareiss's fraction-free elimination: each step's elements are minors of the matrix,
    ///   `(a_ij·a_kk - a_ik·a_kj) / p` with the previous pivot `p` dividing exactly, so the
    ///   numbers grow no longer than the determinant. About `n³/3` steps, each two products and an
    ///   exact division, in registers while the minors fit in 63 bits.
    /// - The residues: `det(A) mod p` for word primes `p > 2^61`, by elimination over `GF(p)`,
    ///   until the primes' product passes twice Hadamard's bound, then the Chinese remainder
    ///   theorem. For elements of `L` limbs that is about `64·n·L/61` primes, each `n³/3`
    ///   products of words for the elimination and the residues of the `n²` elements, `L`
    ///   products of words each; from 512 limbs the residues come from a remainder tree over
    ///   the primes instead, about `n·M(L)·log L` an element for all of them (`M(L)` a product of
    ///   `L` limbs). From 20 rows to 28 or 32 for elements of 25 to 192 bits, from 16 rows to 40,
    ///   56 or 80 up to 1536 bits, from 12 rows to 112 up to 3072 bits, and for longer elements
    ///   from an order that grows with their length (12 rows at 6144 bits, 16 at 16 384, 28 at
    ///   65 536, 40 at 262 144, 80 at 600 000) to `max(3·⌈bits/64⌉, 160)` rows, an edge from a
    ///   model of the two methods' costs.
    /// - Above those bands, the least common multiple of the denominators of a solution of
    ///   `A·x = b` by Dixon's lifting, a divisor of the determinant, times the quotient's residues
    ///   modulo primes (see
    ///   [`Matrix::<Rational>::det`](Matrix::<crate::math::rational::Rational>::det)); a matrix
    ///   that the lifting finds singular modulo its first primes is proved singular, by a vector
    ///   of its null space checked exactly, or lifted modulo a later prime.
    ///
    /// The result is the determinant for every matrix: the residues stop at the bound, never
    /// when they seem to settle. The determinant of the `0 × 0` matrix is 1, and the result is in
    /// the byte order of the top left element.
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
    /// [`inverse_rational`](Matrix::<Number>::inverse_rational) gives `A⁻¹` as rationals. From
    /// 24 rows (16 with an element past 63 bits) a singular matrix is proved singular first, as
    /// [`solve`](Matrix::<Number>::solve) does, rather than by the whole elimination. The results
    /// are in the byte order of the top left element.
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
