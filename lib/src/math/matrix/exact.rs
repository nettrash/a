//! Exact algorithms on matrices of numbers: determinants, ranks, characteristic polynomials,
//! adjugates, inverses and null spaces, without fractions.

use super::Matrix;
use crate::math::numbers::Number;

impl Matrix<Number> {
    /// The determinant, exactly, by Bareiss's fraction-free elimination: each step's elements
    /// are minors of the matrix, `(a_ij·a_kk - a_ik·a_kj) / p` with the previous pivot `p`
    /// dividing exactly, so the numbers grow no longer than the determinant. About `n³`
    /// products. The determinant of the `0 × 0` matrix is 1.
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
        let n = self.rows();
        let mut m = self.clone();
        let (rank, swaps) = bareiss(&mut m);
        if rank < n {
            return Number::default();
        }
        if n == 0 {
            return Number::from(1u8);
        }
        let d = m[(n - 1, n - 1)].clone();
        if swaps % 2 == 1 { -d } else { d }
    }

    /// The rank, exactly: the number of pivots of Bareiss's fraction-free elimination.
    pub fn rank(&self) -> usize {
        bareiss(&mut self.clone()).0
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
    /// `None` when the determinant is zero.
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
        let det = self.det();
        if det.is_zero() {
            return None;
        }
        Some((self.adjugate(), det))
    }

    /// A basis of the null space `{x : A·x = 0}` over the rationals, as the columns of an
    /// `n × (n - rank)` matrix of integers: one vector for each column without a pivot in the
    /// fraction-free echelon form, its free coordinate positive and its elements without a
    /// common factor.
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
        let n = self.cols();
        let mut e = self.clone();
        let (rank, _) = bareiss(&mut e);
        // The pivot columns of the echelon form.
        let mut pivots = Vec::with_capacity(rank);
        let mut row = 0;
        for c in 0..n {
            if row < rank && !e[(row, c)].is_zero() {
                pivots.push(c);
                row += 1;
            }
        }
        let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
        let mut basis = Matrix::<Number>::zeros(n, free.len());
        for (b, &f) in free.iter().enumerate() {
            // `y`, a multiple of the solution with `x_f = 1` and the other free coordinates 0.
            let mut y = vec![Number::default(); n];
            y[f] = Number::from(1u8);
            for i in (0..rank).rev() {
                let mut s = Number::default();
                for c in i..n {
                    if c != pivots[i] && !y[c].is_zero() && !e[(i, c)].is_zero() {
                        s -= &e[(i, c)] * &y[c];
                    }
                }
                let pivot = &e[(i, pivots[i])];
                let g = s.gcd(pivot);
                if !g.is_zero() {
                    let scale = pivot.div_exact(&g).abs();
                    if scale != 1 {
                        for x in &mut y {
                            *x *= &scale;
                        }
                        s *= &scale;
                    }
                }
                y[pivots[i]] = s.div_exact(pivot);
            }
            let g = y.iter().fold(Number::default(), |g, x| g.gcd(x));
            for (i, x) in y.into_iter().enumerate() {
                basis[(i, b)] = if g > 1 { x.div_exact(&g) } else { x };
            }
            if basis[(f, b)].is_negative() {
                for i in 0..n {
                    basis[(i, b)] = -&basis[(i, b)];
                }
            }
        }
        basis
    }
}

/// Bareiss's fraction-free elimination of `m` to an echelon form in place, the pivot of each
/// row the first non-zero element at or below it in the next column that has one: the rank,
/// and the number of row swaps.
fn bareiss(m: &mut Matrix<Number>) -> (usize, usize) {
    let (rows, cols) = m.shape();
    let mut previous = Number::from(1u8);
    let mut r = 0;
    let mut swaps = 0;
    for c in 0..cols {
        if r == rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !m[(i, c)].is_zero()) else {
            continue;
        };
        if p != r {
            m.swap_rows(p, r);
            swaps += 1;
        }
        let pivot = m[(r, c)].clone();
        for i in r + 1..rows {
            let below = m[(i, c)].clone();
            for j in c + 1..cols {
                let x = &(&m[(i, j)] * &pivot) - &(&below * &m[(r, j)]);
                m[(i, j)] = if previous == 1 {
                    x
                } else {
                    x.div_exact(&previous)
                };
            }
            m[(i, c)] = Number::default();
        }
        previous = pivot;
        r += 1;
    }
    (r, swaps)
}
