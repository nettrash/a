//! Whether `R` is singular to working precision when no element of its diagonal is small: a
//! lower bound of its condition number, found by a few solves with `R` and `Rᴴ`.
//!
//! The diagonal of a triangular matrix bounds its smallest singular value from above, and no
//! more: the upper bidiagonal matrix of 1 on the diagonal and −2 above it has a condition number
//! of about `2ⁿ⁺¹` and every diagonal element 1, and a column that is an exact combination of
//! others with large coefficients leaves on the diagonal a rounding error amplified by those
//! coefficients, far above any tolerance, while `R` is as singular as the matrix. So the
//! diagonal alone cannot tell; the inverse of `R` can.
//!
//! The condition number of `A·Q_c` is that of `R`, `‖R‖₂·‖R⁻¹‖₂`, since `Q` has orthonormal
//! columns. Its lower bound here is `c·‖R⁻¹·x‖₂/‖x‖₂` for a few vectors `x`: `c`, the largest
//! 2-norm of a column of the matrix, is that of a column of `R` and no more than `‖R‖₂`, and
//! `‖R⁻¹·x‖₂/‖x‖₂` is no more than `‖R⁻¹‖₂` for any `x`. The first vector is LINPACK's (Cline,
//! Moler, Stewart and Wilkinson, "An estimate for the condition number of a matrix", SIAM J.
//! Numer. Anal. 16, 1979): `Rᴴ·y = e` is solved with each `e_k` of modulus 1 chosen when the
//! substitution reaches it, of the phase that makes `|y_k|` the largest, which follows the growth
//! that a nearly singular triangle makes. Then `y` and the vectors after it go through `R⁻¹` and
//! `R⁻ᴴ` in turn, normalised, which is the power method on `(Rᴴ·R)⁻¹`: the norms grow towards
//! `‖R⁻¹‖₂`, and a component along its leading singular vectors, however small at first, grows
//! at each solve against the others by the ratio of the next smallest singular value to the
//! smallest, which an exactly singular matrix makes enormous. Six solves in all, each costing
//! the stored elements of `R`.
//!
//! `R` is read divided by `c`, which leaves its condition number as it is and keeps the numbers
//! of the solves in a fixed range whatever the scale of the matrix: every column of `R` then
//! has a norm of at most 1, up to rounding, and every diagonal element is above `1/limit`. So an
//! element of a solution is at most `limit` times its right-hand side's element plus the norm of
//! the elements found before it (`√n` times that norm in a back substitution), and as a solve
//! stops as soon as the norm it has built passes the limit, no number in it exceeds about
//! `2·limit²·√n`: far inside the range of the type, whose `limit` is at most `1/(32·ε)`.

use crate::math::complex::Float;
use crate::math::matrix::Scalar;

/// `R` by rows, as [`super::SparseQr`] keeps it, read divided by `c`.
struct Triangle<'a, T: Scalar> {
    /// Where each row begins in `cols` and `values`, and where the last one ends.
    ptr: &'a [usize],
    /// The columns of the elements, ascending within a row, the diagonal first.
    cols: &'a [usize],
    /// The elements; those of the diagonal are real.
    values: &'a [T],
    /// The divisor.
    c: T::Real,
    /// `1/c`, which overflows when `c` is subnormal.
    unit: T::Real,
}

impl<T: Scalar> Triangle<'_, T> {
    /// The element at `p`, divided by `c`.
    fn element(&self, p: usize) -> T {
        let x = self.values[p];
        if self.unit.is_finite() {
            x.scale(self.unit)
        } else {
            T::from_parts(x.re() / self.c, x.im() / self.c)
        }
    }

    /// `x / R_kk` for the real diagonal element of row `k`, which begins at `p`.
    fn divide(&self, x: T, p: usize) -> T {
        let d = self.element(p).re();
        T::from_parts(x.re() / d, x.im() / d)
    }

    /// `y ← R⁻ᴴ·x`, or `y ← R⁻ᴴ·e` for LINPACK's `e` when `x` is `None`, by the rows of `R`, the
    /// columns of `Rᴴ`: `y_j` gathers `Σ conj(R_kj)·y_k` from the rows `k` above `j` before its
    /// turn comes. The square of `‖y‖₂`, or `None` once that passes `bound`.
    fn solve_adjoint(&self, x: Option<&[T]>, y: &mut [T], bound: T::Real) -> Option<T::Real> {
        y.fill(T::zero());
        let mut norm = T::Real::ZERO;
        for k in 0..y.len() {
            let (begin, end) = (self.ptr[k], self.ptr[k + 1]);
            let sum = y[k];
            let rhs = match x {
                Some(x) => x[k] - sum,
                None => {
                    // `e_k = −sum/|sum|` makes `|e_k − sum| = 1 + |sum|`, the largest of modulus 1.
                    let size = sum.abs();
                    let e = if size == T::Real::ZERO {
                        T::one()
                    } else {
                        T::from_parts(-sum.re() / size, -sum.im() / size)
                    };
                    e - sum
                }
            };
            let element = self.divide(rhs, begin);
            y[k] = element;
            norm += element.abs2();
            if passes(norm, bound) {
                return None;
            }
            for p in begin + 1..end {
                let j = self.cols[p];
                y[j] += self.element(p).conj() * element;
            }
        }
        Some(norm)
    }

    /// `z ← R⁻¹·x` by back substitution. The square of `‖z‖₂`, or `None` once that passes
    /// `bound`.
    fn solve(&self, x: &[T], z: &mut [T], bound: T::Real) -> Option<T::Real> {
        let mut norm = T::Real::ZERO;
        for k in (0..z.len()).rev() {
            let (begin, end) = (self.ptr[k], self.ptr[k + 1]);
            let mut sum = x[k];
            for p in begin + 1..end {
                sum -= self.element(p) * z[self.cols[p]];
            }
            let element = self.divide(sum, begin);
            z[k] = element;
            norm += element.abs2();
            if passes(norm, bound) {
                return None;
            }
        }
        Some(norm)
    }
}

/// Whether the square of a norm passes `bound`. A NaN, which the bounds on the numbers of the
/// solves rule out, would count as passing it: when in doubt the triangle is refused.
fn passes<R: Float>(norm: R, bound: R) -> bool {
    norm > bound || norm.is_nan()
}

/// The number of solves with `R` or `Rᴴ`: LINPACK's vector, and five steps of the power method.
const SOLVES: usize = 6;

/// Whether the lower bound of the condition number of `R` passes `limit`: `R` given by rows as
/// `ptr`, `cols` and `values`, its diagonal first in each row, real and above `c/limit`, and `c`
/// the largest 2-norm of a column of the matrix, positive and finite.
pub(super) fn singular_to_working_precision<T: Scalar>(
    ptr: &[usize],
    cols: &[usize],
    values: &[T],
    c: T::Real,
    limit: T::Real,
) -> bool {
    let n = ptr.len() - 1;
    if n == 0 {
        return false;
    }
    let r = Triangle {
        ptr,
        cols,
        values,
        c,
        unit: T::Real::ONE / c,
    };
    let bound = limit * limit;
    let mut x = vec![T::zero(); n];
    let mut y = vec![T::zero(); n];
    // `‖R⁻ᴴ·e‖₂/‖e‖₂` passes the limit when `‖y‖₂²` passes `limit²·n`, `‖e‖₂²` being `n`.
    let Some(mut norm) = r.solve_adjoint(None, &mut y, bound * T::Real::from_f64(n as f64)) else {
        return true;
    };
    for step in 1..SOLVES {
        // The squares of the norms stay above `1/n`: `‖R⁻¹·x‖₂ ≥ ‖x‖₂/‖R‖₂`, and `‖R‖₂` is at
        // most `‖R‖_F ≤ √n` divided by `c`; LINPACK's `y` is larger still.
        let unit = T::Real::ONE / norm.sqrt();
        for (x, &y) in x.iter_mut().zip(&y) {
            *x = y.scale(unit);
        }
        let next = if step % 2 == 1 {
            r.solve(&x, &mut y, bound)
        } else {
            r.solve_adjoint(Some(&x), &mut y, bound)
        };
        match next {
            Some(found) => norm = found,
            None => return true,
        }
    }
    false
}
