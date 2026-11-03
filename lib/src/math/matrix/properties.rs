//! Norms, condition numbers, ranks, null spaces and characteristic polynomials.

use super::householder::norm2;
use super::schur::hessenberg;
use super::svd::svd;
use super::{Matrix, Scalar};
use crate::math::complex::Float;

/// The power of two `k` by which the finite elements `data` are scaled, `2^k·A`, as LAPACK's
/// drivers scale, when the largest of them is so small or so large that products of elements
/// could underflow or overflow, outside `[√MIN/ε, ε/√MIN]`; 0 inside it, which the largest part
/// decides alone unless it is near an end. A complex element can have finite parts and an
/// absolute value that overflows: its largest part, within a factor √2, stands for it then.
pub(crate) fn range_exponent<T: Scalar>(data: &[T]) -> i32 {
    let zero = T::Real::ZERO;
    let small = T::Real::MIN_POSITIVE.sqrt() / T::Real::EPSILON;
    let big = data
        .iter()
        .fold(zero, |s, x| s.max(x.re().abs().max(x.im().abs())));
    if big >= small && big * T::Real::TWO.sqrt() <= T::Real::ONE / small {
        return 0;
    }
    let max = data.iter().fold(zero, |s, x| s.max(x.abs()));
    let log2 = |x: T::Real| x.ln().to_f64() / core::f64::consts::LN_2;
    let top = if max.is_finite() {
        log2(max)
    } else {
        log2(big) + 0.5
    };
    if max > zero && max < small {
        (log2(small) - top).ceil() as i32
    } else if max > T::Real::ONE / small {
        (-log2(small) - top).floor() as i32
    } else {
        0
    }
}

/// A matrix norm, for [`Matrix::norm`] and [`Matrix::cond`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Norm {
    /// The largest sum of the absolute values in a column.
    One,
    /// The spectral norm, the largest singular value.
    Two,
    /// The largest sum of the absolute values in a row.
    Infinity,
    /// Frobenius's, the square root of the sum of the squares of the absolute values.
    Frobenius,
    /// The largest absolute value of an element; not submultiplicative.
    Max,
    /// The nuclear norm, the sum of the singular values.
    Nuclear,
}

impl<T: Scalar> Matrix<T> {
    /// The norm `‖A‖` of the kind given: see [`Norm`]. The 2-norm and the nuclear norm take the
    /// singular values; the others one pass over the elements, Frobenius's scaled so that the
    /// squares neither overflow nor underflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, Norm};
    ///
    /// let a: Matrix<f64> = Matrix::from_rows(&[[1.0, -2.0], [-3.0, 4.0]]);
    /// assert_eq!(a.norm(Norm::One), 6.0);
    /// assert_eq!(a.norm(Norm::Infinity), 7.0);
    /// assert_eq!(a.norm(Norm::Max), 4.0);
    /// assert!((a.norm(Norm::Frobenius) - 30f64.sqrt()).abs() < 1e-15);
    /// ```
    pub fn norm(&self, norm: Norm) -> T::Real {
        let zero = T::Real::ZERO;
        let (m, n) = self.shape();
        match norm {
            Norm::One => (0..n)
                .map(|j| self.column(j).iter().map(|x| x.abs()).sum::<T::Real>())
                .fold(zero, T::Real::max),
            Norm::Infinity => {
                let mut rows = vec![zero; m];
                for j in 0..n {
                    for (r, x) in rows.iter_mut().zip(self.column(j)) {
                        *r += x.abs();
                    }
                }
                rows.into_iter().fold(zero, T::Real::max)
            }
            Norm::Frobenius => norm2(&self.data),
            Norm::Max => self.data.iter().map(|x| x.abs()).fold(zero, T::Real::max),
            Norm::Two => svd(self, false).0.first().copied().unwrap_or(zero),
            Norm::Nuclear => svd(self, false).0.into_iter().sum(),
        }
    }

    /// The condition number `κ(A) = ‖A‖·‖A⁻¹‖` in the norm given, which bounds how much a
    /// relative change of `A` or of `b` can change the solution of `A·x = b`: infinite for a
    /// singular matrix. In the 2-norm it is the ratio of the largest singular value to the
    /// smallest, for a matrix of any shape; in the other norms it takes the inverse.
    ///
    /// # Panics
    ///
    /// Panics if the norm is not [`Norm::Two`] and the matrix is not square.
    #[track_caller]
    pub fn cond(&self, norm: Norm) -> T::Real {
        let zero = T::Real::ZERO;
        match norm {
            Norm::Two => {
                let s = svd(self, false).0;
                match (s.first(), s.last()) {
                    (Some(&max), Some(&min)) if min > zero => max / min,
                    (Some(_), Some(_)) => T::Real::INFINITY,
                    _ => zero,
                }
            }
            Norm::Nuclear => {
                assert!(self.is_square(), "the matrix is not square");
                let s = svd(self, false).0;
                if s.last().is_some_and(|&x| x == zero) {
                    return T::Real::INFINITY;
                }
                let sum: T::Real = s.iter().copied().sum();
                let inv: T::Real = s.iter().map(|&x| T::Real::ONE / x).sum();
                sum * inv
            }
            _ => {
                assert!(self.is_square(), "the matrix is not square");
                match self.inverse() {
                    Some(inv) => self.norm(norm) * inv.norm(norm),
                    None => T::Real::INFINITY,
                }
            }
        }
    }

    /// The rank: the number of singular values above `max(m, n)·ε·σ₁`, `ε` the unit roundoff
    /// of the type and `σ₁` the largest singular value, as NumPy's `matrix_rank`.
    pub fn rank(&self) -> usize {
        let s = svd(self, false).0;
        let tol = self.rank_tolerance(&s);
        s.iter().filter(|&&x| x > tol).count()
    }

    /// An orthonormal basis of the null space `{x : A·x = 0}`, as the columns of an
    /// `n × (n - rank)` matrix: the right singular vectors of the singular values at or below
    /// the tolerance of [`rank`](Matrix::rank).
    pub fn null_space(&self) -> Matrix<T> {
        let (m, n) = self.shape();
        // With fewer rows than columns, rows of zeros make all `n` right singular vectors.
        let a = if m < n {
            Matrix::from_fn(n, n, |i, j| if i < m { self[(i, j)] } else { T::zero() })
        } else {
            self.clone()
        };
        let (s, uv) = svd(&a, true);
        let v = uv.map(|(_, v)| v).unwrap_or_default();
        let tol = self.rank_tolerance(&s);
        let rank = s.iter().filter(|&&x| x > tol).count();
        Matrix::from_fn(n, n - rank, |i, j| v[(i, rank + j)])
    }

    /// The tolerance below which a singular value counts as zero.
    fn rank_tolerance(&self, s: &[T::Real]) -> T::Real {
        let (m, n) = self.shape();
        let max = s.first().copied().unwrap_or(T::Real::ZERO);
        T::Real::from_f64(m.max(n) as f64) * T::Real::EPSILON * max
    }

    /// The coefficients of the characteristic polynomial `det(x·I - A)`, from `xⁿ` down: `1`,
    /// then `-tr(A)`, …, `(-1)ⁿ·det(A)`. From the Hessenberg form, by La Budde's recurrence on
    /// its leading blocks, about `n³` operations.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    ///
    /// let a: Matrix<f64> = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]);
    /// // (x - 1)(x - 3)
    /// assert_eq!(a.characteristic_polynomial(), [1.0, -4.0, 3.0]);
    /// ```
    #[track_caller]
    pub fn characteristic_polynomial(&self) -> Vec<T> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let mut h = self.clone();
        hessenberg(&mut h, false);
        let at = |i: usize, j: usize| h.data[i + j * n];
        // `p[s]`: the polynomial of the leading `s × s` block, from the constant term up.
        let mut p: Vec<Vec<T>> = vec![vec![T::one()]];
        for k in 0..n {
            // `(x - h_kk)·p_k`.
            let prev = &p[k];
            let mut next = vec![T::zero(); k + 2];
            for (i, &c) in prev.iter().enumerate() {
                next[i + 1] += c;
                next[i] -= at(k, k) * c;
            }
            // `- Σ h_{k-m,k}·(h_{k,k-1}·…·h_{k-m+1,k-m})·p_{k-m}`.
            let mut product = T::one();
            for m in 1..=k {
                product *= at(k - m + 1, k - m);
                let f = at(k - m, k) * product;
                if f == T::zero() {
                    continue;
                }
                for (i, &c) in p[k - m].iter().enumerate() {
                    next[i] -= f * c;
                }
            }
            p.push(next);
        }
        let mut c = p.pop().unwrap_or_default();
        c.reverse();
        c
    }
}
