//! The singular value decomposition: a reduction to a real bidiagonal matrix by Householder
//! reflections in panels (after one to triangular by QR for a tall matrix), then divide and
//! conquer on the bidiagonal matrix, whose singular vectors the reflections turn into those of the
//! matrix; the singular values alone by the dqds algorithm. Small matrices are reduced without
//! panels, their vectors by the implicit QR algorithm.

mod bidiagonal;
mod dc;
mod dqds;
mod qr;
mod secular;
mod small;

use super::properties::range_exponent;
use super::{Matrix, Scalar};
use crate::math::complex::Float;
use bidiagonal::{apply_adjoint_right, reduce, triangularize};

/// The singular value decomposition `A = U·Σ·V*` of an `m × n` matrix, `k = min(m, n)`: `U` of
/// `m × k` and `V` of `n × k` with orthonormal columns, `Σ` the diagonal of the singular values,
/// real, not negative, in descending order. From [`Matrix::svd`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a: Matrix<f64> = Matrix::from_rows(&[[3.0, 0.0], [4.0, 5.0]]);
/// let svd = a.svd();
/// let s = svd.singular_values();
/// assert!((s[0] - 45f64.sqrt()).abs() < 1e-14 && (s[1] - 5f64.sqrt()).abs() < 1e-14);
/// let sigma = Matrix::from_diagonal(s);
/// let back = &(&svd.u() * &sigma) * &svd.v().transpose();
/// assert!((&back - &a).as_slice().iter().all(|x| x.abs() < 1e-14));
/// ```
#[derive(Clone, Debug)]
pub struct Svd<T: Scalar> {
    /// `U`, `m × k`.
    u: Matrix<T>,
    /// The singular values, descending.
    s: Vec<T::Real>,
    /// `V`, `n × k`.
    v: Matrix<T>,
}

impl<T: Scalar> Svd<T> {
    /// `U`, `m × min(m, n)`, the left singular vectors as orthonormal columns.
    pub fn u(&self) -> Matrix<T> {
        self.u.clone()
    }

    /// The singular values, real, not negative, in descending order.
    pub fn singular_values(&self) -> &[T::Real] {
        &self.s
    }

    /// `V`, `n × min(m, n)`, the right singular vectors as orthonormal columns.
    pub fn v(&self) -> Matrix<T> {
        self.v.clone()
    }
}

impl<T: Scalar> Matrix<T> {
    /// The singular value decomposition `A = U·Σ·V*`: see [`Svd`]. A reduction to a real
    /// bidiagonal matrix by Householder reflections from both sides, in panels whose updates of
    /// the rest of the matrix are products of blocks, then divide and conquer on the bidiagonal
    /// matrix, Gu and Eisenstat's, and the reflections applied to its singular vectors in blocks:
    /// about `9n³` operations for a square matrix, most of them in products of blocks. Up to
    /// about a hundred columns (88 to 112 by the type), where that takes longer, the reduction
    /// is made one column and row at a time, `U` and `V` are multiplied out from the reflections,
    /// and the implicit QR algorithm's rotations are applied to them. A matrix with at least
    /// `2.5` times as many rows as columns is first reduced to triangular by QR. A matrix with a
    /// NaN or an infinity gives NaNs.
    pub fn svd(&self) -> Svd<T> {
        let (s, uv) = svd(self, true);
        let (u, v) = uv.unwrap_or_default();
        Svd { u, s, v }
    }

    /// The singular values, real, not negative, in descending order, without `U` and `V`:
    /// much less work than [`svd`](Matrix::svd). Those of the bidiagonal matrix come from the
    /// dqds algorithm, to high relative accuracy, so that a bidiagonal matrix has even its
    /// smallest singular values to nearly full precision; QR goes first from `1.75` times as
    /// many rows as columns.
    pub fn singular_values(&self) -> Vec<T::Real> {
        svd(self, false).0
    }
}

/// The singular values of `a`, and `U` and `V` when `want_uv` is set.
#[allow(clippy::type_complexity)]
pub(crate) fn svd<T: Scalar>(
    a: &Matrix<T>,
    want_uv: bool,
) -> (Vec<T::Real>, Option<(Matrix<T>, Matrix<T>)>) {
    let (m, n) = a.shape();
    // A wide matrix by its conjugate transpose, `A* = V·Σ·U*`.
    let wide = m < n;
    let (rows, cols) = if wide { (n, m) } else { (m, n) };
    let swap = |uv: Option<(Matrix<T>, Matrix<T>)>| {
        if wide { uv.map(|(u, v)| (v, u)) } else { uv }
    };
    if a.data.iter().any(|x| !x.is_finite()) {
        let nan = T::from_parts(T::Real::NAN, T::Real::NAN);
        let uv = want_uv.then(|| {
            (
                Matrix::from_fn(rows, cols, |_, _| nan),
                Matrix::from_fn(cols, cols, |_, _| nan),
            )
        });
        return (vec![T::Real::NAN; cols], swap(uv));
    }
    // Scaled by a power of two when the largest element is so small or so large that products
    // of elements could underflow or overflow, as LAPACK's drivers do.
    let k = range_exponent(&a.data);
    let mut x = if wide {
        adjoint(&a.data, m, n).data
    } else {
        a.data.clone()
    };
    if k == 0 {
        let (s, uv) = decompose(x, rows, cols, want_uv);
        return (s, swap(uv));
    }
    let up = T::Real::TWO.powi(k);
    for x in &mut x {
        *x = x.scale(up);
    }
    let (mut s, uv) = decompose(x, rows, cols, want_uv);
    let down = T::Real::TWO.powi(-k);
    for x in &mut s {
        *x *= down;
    }
    (s, swap(uv))
}

/// [`svd`] of the `m × n` column-major `a`, `m ≥ n`, its elements finite and of a moderate size.
#[allow(clippy::type_complexity)]
fn decompose<T: Scalar>(
    mut a: Vec<T>,
    m: usize,
    n: usize,
    want_uv: bool,
) -> (Vec<T::Real>, Option<(Matrix<T>, Matrix<T>)>) {
    // QR first from `m ≥ 2.5n` with the vectors, from `m ≥ 1.75n` without, where it starts to
    // pay.
    if n > 0
        && if want_uv {
            2 * m >= 5 * n
        } else {
            4 * m >= 7 * n
        }
    {
        // `A = Q·R`, and `R = U·Σ·V*`: fewer operations than the reduction of `A` itself.
        let taus = triangularize(&mut a, m, n);
        let mut r = vec![T::zero(); n * n];
        for j in 0..n {
            r[j * n..j * n + j + 1].copy_from_slice(&a[j * m..j * m + j + 1]);
        }
        let (s, uv) = decompose(r, n, n, want_uv);
        let uv = uv.map(|(ur, v)| {
            // `U* = [U_R*, 0]·Q*`.
            let mut x = vec![T::zero(); n * m];
            for j in 0..n {
                for (i, x) in x[j * n..(j + 1) * n].iter_mut().enumerate() {
                    *x = ur.data[j + i * n].conj();
                }
            }
            apply_adjoint_right(&a, m, m, &taus, &mut x, n);
            (adjoint(&x, n, m), v)
        });
        return (s, uv);
    }
    if n <= small::largest::<T>(want_uv) {
        return small::decompose(a, m, n, want_uv);
    }
    let (mut d, mut e, reduced) = reduce(a, m, n);
    if !want_uv {
        if !dqds::singular_values(&mut d, &e) {
            e.push(T::Real::ZERO);
            qr::bidiagonal_qr::<_, qr::Columns<_>, qr::Columns<_>>(&mut d, &mut e, None, None);
        }
        return (d, None);
    }
    let mut ub = vec![T::Real::ZERO; n * n];
    let mut vb = vec![T::Real::ZERO; n * n];
    dc::bidiagonal_svd(&mut d, &mut e, &mut ub, &mut vb, T::gemm_real);
    // `U* = [U_B*, 0]·Q*` and `V* = V_B*·P*`, the reflections applied from the right.
    let mut x = vec![T::zero(); n * m];
    let mut y = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            x[i + j * n] = T::from_real(ub[j + i * n]);
            y[i + j * n] = T::from_real(vb[j + i * n]);
        }
    }
    reduced.apply_q_right(&mut x, n);
    reduced.apply_p_right(&mut y, n);
    (d, Some((adjoint(&x, n, m), adjoint(&y, n, n))))
}

/// The conjugate transpose of the `rows × cols` column-major `x`, in tiles.
fn adjoint<T: Scalar>(x: &[T], rows: usize, cols: usize) -> Matrix<T> {
    const TILE: usize = 16;
    let mut a = Matrix::zeros(cols, rows);
    for j0 in (0..rows).step_by(TILE) {
        for i0 in (0..cols).step_by(TILE) {
            for j in j0..(j0 + TILE).min(rows) {
                for i in i0..(i0 + TILE).min(cols) {
                    a.data[i + j * cols] = x[j + i * rows].conj();
                }
            }
        }
    }
    a
}
