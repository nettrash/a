//! The Cholesky decomposition of a Hermitian positive definite matrix.
//!
//! Up to 7 × 7, or 4 × 4 for complex elements, the decomposition is unrolled ([`tiny`]); from
//! 8 to 9, or 10 for `f64`, it is the dot products of [`dot`]. Up to 32 × 32, or 16 × 16 for
//! complex elements, a leaf factors it column by column ([`narrow`]): for `f64` and `f32` the
//! NEON leaf (`aarch64`) holds each column in vector registers while the columns before it are
//! subtracted, its pivot summed apart in scalars so that the square root need not wait for it to
//! be taken out of a vector; complex elements take [`dot`]. Larger matrices are halved: `L11`, then
//! `L21 = A21·L11⁻*` by a triangular solve on the right, reading a copy of `L11` in working memory
//! taken once for the whole recursion, then `A22 - L21·L21*` on its lower triangle, in blocks of
//! [`UPDATE`] columns. The upper triangle of `L` is zeroed as its parts are finished.
//!
//! # Speed
//!
//! Measured on an Apple M4 Max, one core, `f64`, in µs, the median of 21 interleaved batches,
//! against faer 0.24 (`llt`, `Par::Seq`) and nalgebra 0.35 (`cholesky`):
//!
//! | n | ours | faer | nalgebra |
//! | --- | --- | --- | --- |
//! | 2 | 0.018 | 0.071 | 0.019 |
//! | 4 | 0.026 | 0.114 | 0.039 |
//! | 8 | 0.104 | 0.211 | 0.124 |
//! | 16 | 0.256 | 0.436 | 0.399 |
//! | 32 | 1.09 | 1.42 | 1.99 |
//! | 64 | 5.37 | 7.86 | 10.9 |
//! | 128 | 31.7 | 33.1 | 57.1 |
//! | 256 | 170 | 187 | 432 |
//! | 512 | 1 197 | 1 403 | 3 735 |
//! | 1024 | 7 778 | 8 753 | 29 377 |
//!
//! Apple's Accelerate, on the matrix unit (SME), ran 2 to 6 times faster from 32 × 32 up
//! (`dpotrf` 220 µs at 512).

use super::blas::{Op, Side, Uplo, gemm, trsm, with_scratch};
use super::{Matrix, Scalar};
use crate::math::complex::Float;

// The leaves of `f64` and `f32` in NEON intrinsics on little-endian aarch64; the portable leaf
// serves the other types and targets, and Miri.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod aarch64;

/// The Cholesky decomposition `A = L·L*` of a Hermitian (for real elements, symmetric) positive
/// definite matrix, `L` lower triangular with a positive real diagonal. From
/// [`Matrix::cholesky`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a = Matrix::from_rows(&[[4.0, 2.0], [2.0, 5.0]]);
/// let c = a.cholesky().unwrap();
/// assert_eq!(c.l(), Matrix::from_rows(&[[2.0, 0.0], [1.0, 2.0]]));
/// assert_eq!(c.det(), 16.0);
/// assert!(Matrix::from_rows(&[[1.0, 2.0], [2.0, 1.0]]).cholesky().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct Cholesky<T> {
    /// `L`, zeros above the diagonal.
    l: Matrix<T>,
}

impl<T: Scalar> Matrix<T> {
    /// The Cholesky decomposition `A = L·L*`, or `None` when the matrix is not positive
    /// definite: a pivot is not positive, or not a number. Only the lower triangle is read, the
    /// upper one taken to be its conjugate. About `n³/3` operations, nearly all of them in
    /// products of blocks.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn cholesky(&self) -> Option<Cholesky<T>> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let mut l = self.clone();
        match n {
            0 => {}
            1 => tiny::<T, 1>(&mut l.data)?,
            2 => tiny::<T, 2>(&mut l.data)?,
            3 => tiny::<T, 3>(&mut l.data)?,
            4 => tiny::<T, 4>(&mut l.data)?,
            // Unrolled past 4, complex elements ran slower than by the loops; real ones ran
            // fastest unrolled up to 7, by the dot products of [`dot`] at 8 and 9 (and 10 for
            // `f64`), and by the NEON leaves past that.
            5 if !T::IS_COMPLEX => tiny::<T, 5>(&mut l.data)?,
            6 if !T::IS_COMPLEX => tiny::<T, 6>(&mut l.data)?,
            7 if !T::IS_COMPLEX => tiny::<T, 7>(&mut l.data)?,
            8 | 9 => dot(&mut l.data, n, n)?,
            10 if size_of::<T>() == 8 => dot(&mut l.data, n, n)?,
            _ if n <= narrow_order::<T>() => narrow(&mut l.data, n, n)?,
            _ => {
                // The working memory of the recursion: a copy of the largest `L11`.
                let n1 = split::<T>(n);
                with_scratch(n1 * n1, |work| factor(&mut l.data, n, n, work))?;
            }
        }
        Some(Cholesky { l })
    }
}

impl<T: Scalar> Cholesky<T> {
    /// `L`, lower triangular with a positive real diagonal.
    pub fn l(&self) -> Matrix<T> {
        self.l.clone()
    }

    /// The determinant of the matrix, the square of the product of the diagonal of `L`: real
    /// and positive.
    pub fn det(&self) -> T::Real {
        let mut d = T::Real::ONE;
        for x in self.l.diagonal() {
            let r = x.re();
            d *= r * r;
        }
        d
    }

    /// The solution `X` of `A·X = B`.
    ///
    /// # Panics
    ///
    /// Panics if `B` has not as many rows as the matrix.
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Matrix<T> {
        let n = self.l.rows();
        assert_eq!(b.rows(), n, "the right-hand side has not as many rows");
        let mut x = b.clone();
        let cols = x.cols();
        trsm(
            Side::Left,
            Uplo::Lower,
            Op::N,
            false,
            n,
            cols,
            &self.l.data,
            n,
            &mut x.data,
            n,
        );
        trsm(
            Side::Left,
            Uplo::Lower,
            Op::C,
            false,
            n,
            cols,
            &self.l.data,
            n,
            &mut x.data,
            n,
        );
        x
    }

    /// The inverse of the matrix.
    pub fn inverse(&self) -> Matrix<T> {
        self.solve(&Matrix::identity(self.l.rows()))
    }
}

/// Blocks up to this order are factored without recursion: 32 for real elements, whose leaves
/// keep their columns in vector registers, and 16 for complex ones.
const fn narrow_order<T: Scalar>() -> usize {
    if T::IS_COMPLEX { 16 } else { 32 }
}

/// Where the recursion splits `n` columns: the first part a multiple of [`narrow_order`].
fn split<T: Scalar>(n: usize) -> usize {
    (n / 2).next_multiple_of(narrow_order::<T>()).min(n - 1)
}

/// Factors the `n × n` block at the start of `a` in place, its lower triangle: `L11`, then
/// `L21 = A21·L11⁻*`, then `A22 - L21·L21*` the same way, and zeroes the upper triangle.
/// `L11` and `L21` share their columns, so the solve reads a copy of `L11` in `work`, of at
/// least `split(n)²` elements.
fn factor<T: Scalar>(a: &mut [T], n: usize, ld: usize, work: &mut [T]) -> Option<()> {
    if n <= narrow_order::<T>() {
        return narrow(a, n, ld);
    }
    let n1 = split::<T>(n);
    let n2 = n - n1;
    factor(a, n1, ld, work)?;
    let (left, right) = a.split_at_mut(n1 * ld);
    let l11 = &mut work[..n1 * n1];
    for j in 0..n1 {
        l11[j * n1 + j..(j + 1) * n1].copy_from_slice(&left[j * ld + j..j * ld + n1]);
    }
    trsm(
        Side::Right,
        Uplo::Lower,
        Op::C,
        false,
        n2,
        n1,
        l11,
        n1,
        &mut left[n1..],
        ld,
    );
    // `A22 -= L21·L21*` on the lower triangle only, a block of columns at a time from its
    // diagonal down: about half the products of the whole block.
    let l21 = &left[n1..];
    let a22 = &mut right[n1..];
    let mut j0 = 0;
    while j0 < n2 {
        let w = UPDATE.min(n2 - j0);
        gemm(
            n2 - j0,
            w,
            n1,
            -T::one(),
            &l21[j0..],
            ld,
            Op::N,
            &l21[j0..],
            ld,
            Op::C,
            T::one(),
            &mut a22[j0 + j0 * ld..],
            ld,
        );
        j0 += w;
    }
    factor(a22, n2, ld, work)?;
    // The upper triangle of `L` is zeros: `A12`, and the triangles of the leaves.
    for j in 0..n2 {
        right[j * ld..j * ld + n1].fill(T::zero());
    }
    Some(())
}

/// The width of the column blocks of the update of the trailing block.
const UPDATE: usize = 64;

/// Factors a small block column by column, then zeroes the upper triangle: by the NEON leaves
/// for `f64` and `f32`, and by [`dot`] for the other types and targets, and Miri.
fn narrow<T: Scalar>(a: &mut [T], n: usize, ld: usize) -> Option<()> {
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    if let Some(done) = narrow_neon(a, n, ld) {
        return done.then_some(());
    }
    dot(a, n, ld)
}

/// [`narrow`] without the NEON leaves: each element of column `j` is a dot product of its row
/// with row `j`, the products not fused, so that the sums' chains wait on additions alone.
fn dot<T: Scalar>(a: &mut [T], n: usize, ld: usize) -> Option<()> {
    for j in 0..n {
        let mut d = a[j + j * ld].re();
        for p in 0..j {
            d -= a[j + p * ld].abs2();
        }
        if d.is_nan() || d <= T::Real::ZERO {
            return None;
        }
        let d = d.sqrt();
        a[j + j * ld] = T::from_real(d);
        let r = T::Real::ONE / d;
        for i in j + 1..n {
            let mut s = a[i + j * ld];
            for p in 0..j {
                s -= a[i + p * ld] * a[j + p * ld].conj();
            }
            a[i + j * ld] = s.scale(r);
        }
    }
    for j in 1..n {
        a[j * ld..j * ld + j].fill(T::zero());
    }
    Some(())
}

/// [`narrow`] by the NEON leaves, for `f64` and `f32`: whether the block was positive
/// definite, or `None` for the other types.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
fn narrow_neon<T: Scalar>(a: &mut [T], n: usize, ld: usize) -> Option<bool> {
    use core::any::TypeId;
    let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
    if !(is_f64 || TypeId::of::<T>() == TypeId::of::<f32>()) {
        return None;
    }
    assert!((1..=aarch64::ROWS).contains(&n) && (n - 1) * ld + n <= a.len());
    let len = a.len();
    let a = a.as_mut_ptr();
    // SAFETY: `T` is the type each branch casts to, and the block lies in the slice, as
    // checked.
    Some(unsafe {
        if is_f64 {
            let a = a.cast::<f64>();
            match n {
                1..=2 => aarch64::leaf_f64::<1, 2>(a, n, ld, len),
                3..=4 => aarch64::leaf_f64::<2, 4>(a, n, ld, len),
                5..=6 => aarch64::leaf_f64::<3, 6>(a, n, ld, len),
                7..=8 => aarch64::leaf_f64::<4, 8>(a, n, ld, len),
                9..=12 => aarch64::leaf_f64::<6, 12>(a, n, ld, len),
                13..=16 => aarch64::leaf_f64::<8, 16>(a, n, ld, len),
                17..=24 => aarch64::leaf_f64::<12, 24>(a, n, ld, len),
                _ => aarch64::leaf_f64::<16, 32>(a, n, ld, len),
            }
        } else {
            let a = a.cast::<f32>();
            match n {
                1..=4 => aarch64::leaf_f32::<1, 4>(a, n, ld, len),
                5..=8 => aarch64::leaf_f32::<2, 8>(a, n, ld, len),
                9..=12 => aarch64::leaf_f32::<3, 12>(a, n, ld, len),
                13..=16 => aarch64::leaf_f32::<4, 16>(a, n, ld, len),
                17..=24 => aarch64::leaf_f32::<6, 24>(a, n, ld, len),
                _ => aarch64::leaf_f32::<8, 32>(a, n, ld, len),
            }
        }
    })
}

/// [`narrow`] of an `N × N` matrix, `N` known, so that its loops unroll.
#[inline(always)]
fn tiny<T: Scalar, const N: usize>(a: &mut [T]) -> Option<()> {
    let a = &mut a[..N * N];
    for j in 0..N {
        for p in 0..j {
            let s = a[j + p * N].conj();
            for i in j..N {
                a[i + j * N] = a[i + j * N].mul_sub(a[i + p * N], s);
            }
        }
        let d = pivot(&mut a[j + j * N..(j + 1) * N])?;
        let r = T::Real::ONE / d;
        for i in j + 1..N {
            a[i + j * N] = a[i + j * N].scale(r);
        }
    }
    for j in 1..N {
        for i in 0..j {
            a[i + j * N] = T::zero();
        }
    }
    Some(())
}

/// The pivot of a column, its first element, updated: its square root, written in its place,
/// or `None` when it is not positive, or not a number. Its imaginary part, the rounding of
/// products of conjugates, is dropped.
#[inline(always)]
fn pivot<T: Scalar>(col: &mut [T]) -> Option<T::Real> {
    let d = col[0].re();
    if d.is_nan() || d <= T::Real::ZERO {
        return None;
    }
    let d = d.sqrt();
    col[0] = T::from_real(d);
    Some(d)
}
