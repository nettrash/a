//! The LU decomposition with partial pivoting, and what it gives: determinants, solutions of
//! linear systems and inverses.
//!
//! # The decomposition
//!
//! Square matrices up to 16 × 16 are factored by [`tiny`]: a copy on the stack, its order known
//! so that its loops unroll, the pivots chosen and the rows swapped by selects rather than
//! branches, which keeps the whole matrix in registers. Up to 24 columns, [`unblocked`]
//! eliminates a column at a time by rank-one updates of whole columns. Larger matrices go
//! Toledo's way ([`factor`]): the left half, then the right half updated by it, down to panels
//! of [`NARROW`] = 16 columns. The row swaps of each half are applied to the other a column at a
//! time, each column read once, rather than a row at a time across all of them, which took a
//! third of the time at 512 × 512; and `U12 = L11⁻¹·A12` comes from the triangular solve with a
//! dense copy of it that the product updating `A22` reads, in working memory taken once.
//!
//! Every path chooses the same pivots, the first element of the largest `|re| + |im|` in its
//! column, as LAPACK does, and divides by a pivot so small that its reciprocal overflows rather
//! than multiplying by the reciprocal, as LAPACK's `getf2` does.
//!
//! # The inverse
//!
//! Up to 12 × 12, [`tiny_inverse`] inverts on the stack, in unrolled loops, and allocates only
//! the inverse. Larger matrices are factored in working memory, then inverted as LAPACK's
//! `getri` does: `U⁻¹` by [`upper_inverse`], then `X·L = U⁻¹` by one triangular solve on the
//! right, then the columns swapped back. That is `2n³` operations in all, against `8n³/3` for
//! the solve against the identity that it replaced, nearly all of them in products of blocks.
//!
//! # Speed
//!
//! Measured on an Apple M4 Max, one core, `f64`, in µs, the median of 15 to 21 interleaved
//! batches, against faer 0.24 (`partial_piv_lu`, then its `inverse`, `Par::Seq`) and nalgebra
//! 0.35 (`lu`, `try_inverse`):
//!
//! | n | LU | faer | nalgebra | inverse | faer | nalgebra |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 2 | 0.034 | 0.122 | 0.036 | 0.014 | 0.233 | 0.015 |
//! | 4 | 0.061 | 0.223 | 0.071 | 0.054 | 0.490 | 0.042 |
//! | 8 | 0.140 | 0.381 | 0.173 | 0.209 | 0.970 | 0.582 |
//! | 16 | 0.525 | 0.933 | 0.541 | 1.21 | 2.57 | 2.43 |
//! | 32 | 2.03 | 3.19 | 2.59 | 5.10 | 9.24 | 13.8 |
//! | 64 | 9.45 | 14.2 | 14.7 | 24.0 | 38.1 | 70.8 |
//! | 128 | 52.5 | 71.2 | 99.8 | 146 | 183 | 439 |
//! | 256 | 302 | 400 | 1 109 | 903 | 1 061 | 4 071 |
//! | 512 | 2 024 | 2 697 | 8 417 | 5 769 | 7 675 | 31 030 |
//! | 1024 | 14 424 | 18 348 | 63 443 | 43 363 | 53 809 | 231 502 |
//!
//! nalgebra inverts matrices up to 4 × 4 by their cofactors, a formula with half the operations
//! and none of the pivoting, at 3 × 3 and 4 × 4 a little faster than the decomposition. Apple's
//! Accelerate, on the matrix unit (SME) rather than NEON, ran 2 to 5 times faster from 128 × 128
//! up (`dgetrf` 836 µs and `dgetrf` with `dgetri` 1 785 µs at 512).

use super::blas::{Op, Side, Uplo, gemm, trsm, trsm_into, with_scratch};
use super::{Matrix, Scalar};
use crate::math::complex::Float;
use core::hint::select_unpredictable;

/// The LU decomposition `P·A = L·U` of an `m × n` matrix, by Gaussian elimination with partial
/// pivoting: `P` a permutation, `L` of `m × min(m, n)` lower triangular with ones on its
/// diagonal, `U` of `min(m, n) × n` upper triangular. From [`Matrix::lu`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a = Matrix::from_rows(&[[2.0, 1.0], [4.0, 3.0]]);
/// let lu = a.lu();
/// // The larger first column's element is the pivot: the rows are swapped.
/// assert_eq!(lu.permutation(), [1, 0]);
/// assert_eq!(lu.l(), Matrix::from_rows(&[[1.0, 0.0], [0.5, 1.0]]));
/// assert_eq!(lu.u(), Matrix::from_rows(&[[4.0, 3.0], [0.0, -0.5]]));
/// assert_eq!(&lu.p() * &a, &lu.l() * &lu.u());
/// assert_eq!(lu.det(), 2.0);
/// let x = lu.solve(&Matrix::from_rows(&[[3.0], [7.0]])).unwrap();
/// assert_eq!(x, Matrix::from_rows(&[[1.0], [1.0]]));
/// ```
#[derive(Clone, Debug)]
pub struct Lu<T> {
    /// `L` below the diagonal and `U` on and above it.
    lu: Matrix<T>,
    /// Row `i` of `P·A` is row `perm[i]` of `A`.
    perm: Vec<usize>,
    /// Whether `P` is an odd permutation.
    odd: bool,
}

impl<T: Scalar> Matrix<T> {
    /// The LU decomposition with partial pivoting, `P·A = L·U`: see [`Lu`]. It takes about
    /// `2n³/3` operations for an `n × n` matrix, nearly all of them in products of blocks.
    pub fn lu(&self) -> Lu<T> {
        let (m, n) = self.shape();
        let mut lu = self.clone();
        let mut perm: Vec<usize> = (0..m).collect();
        let mut odd = false;
        if m == n && n <= TINY {
            // Tiny squares unrolled, their loops of known lengths.
            match n {
                0 => {}
                1 => odd = tiny::<T, 1>(&mut lu.data, &mut perm),
                2 => odd = tiny::<T, 2>(&mut lu.data, &mut perm),
                3 => odd = tiny::<T, 3>(&mut lu.data, &mut perm),
                4 => odd = tiny::<T, 4>(&mut lu.data, &mut perm),
                5 => odd = tiny::<T, 5>(&mut lu.data, &mut perm),
                6 => odd = tiny::<T, 6>(&mut lu.data, &mut perm),
                7 => odd = tiny::<T, 7>(&mut lu.data, &mut perm),
                8 => odd = tiny::<T, 8>(&mut lu.data, &mut perm),
                9 => odd = tiny::<T, 9>(&mut lu.data, &mut perm),
                10 => odd = tiny::<T, 10>(&mut lu.data, &mut perm),
                11 => odd = tiny::<T, 11>(&mut lu.data, &mut perm),
                12 => odd = tiny::<T, 12>(&mut lu.data, &mut perm),
                13 => odd = tiny::<T, 13>(&mut lu.data, &mut perm),
                14 => odd = tiny::<T, 14>(&mut lu.data, &mut perm),
                15 => odd = tiny::<T, 15>(&mut lu.data, &mut perm),
                _ => odd = tiny::<T, 16>(&mut lu.data, &mut perm),
            }
            return Lu { lu, perm, odd };
        }
        with_pivots(m.min(n), |pivots| {
            factor_all(&mut lu.data, m, n, pivots);
            for (i, &p) in pivots.iter().enumerate() {
                if p != i {
                    perm.swap(i, p);
                    odd = !odd;
                }
            }
        });
        Lu { lu, perm, odd }
    }

    /// The determinant `det(A)`, from the LU decomposition: the product of the pivots, with the
    /// sign of the permutation. The determinant of the `0 × 0` matrix is 1.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn det(&self) -> T {
        assert!(self.is_square(), "the matrix is not square");
        self.lu().det()
    }

    /// The inverse `A⁻¹`, from the LU decomposition, or `None` when a pivot is zero: the matrix
    /// is singular. A matrix that is singular only up to rounding gives an inverse with huge
    /// elements; its [condition number](Matrix::cond) tells how far to trust it.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn inverse(&self) -> Option<Self> {
        assert!(self.is_square(), "the matrix is not square");
        match self.rows() {
            0 => Some(self.clone()),
            1 => tiny_inverse_inline::<T, 1>(self),
            2 => tiny_inverse_inline::<T, 2>(self),
            3 => tiny_inverse_inline::<T, 3>(self),
            4 => tiny_inverse_inline::<T, 4>(self),
            5 => tiny_inverse::<T, 5>(self),
            6 => tiny_inverse::<T, 6>(self),
            7 => tiny_inverse::<T, 7>(self),
            8 => tiny_inverse::<T, 8>(self),
            9 => tiny_inverse::<T, 9>(self),
            10 => tiny_inverse::<T, 10>(self),
            11 => tiny_inverse::<T, 11>(self),
            12 => tiny_inverse::<T, 12>(self),
            n => {
                // The factors in working memory, their row swaps kept as they are, so that the
                // inverse is the only allocation.
                with_scratch(n * n, |lu| {
                    lu.copy_from_slice(&self.data);
                    with_pivots(n, |pivots| {
                        factor_all(lu, n, n, pivots);
                        if (0..n).any(|i| lu[i + i * n] == T::zero()) {
                            return None;
                        }
                        let mut x = Matrix::zeros(n, n);
                        inverse_from(lu, n, &mut x.data);
                        // Column `j` of `X·P` is column `pivots[j]` of `X` and the other way
                        // round, from the last swap to the first.
                        for (j, &p) in pivots.iter().enumerate().rev() {
                            if p != j {
                                let (a, b) = x.data.split_at_mut(p * n);
                                a[j * n..j * n + n].swap_with_slice(&mut b[..n]);
                            }
                        }
                        Some(x)
                    })
                })
            }
        }
    }

    /// The solution `X` of `A·X = B`, from the LU decomposition, or `None` when the matrix is
    /// singular.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, or `B` has not as many rows.
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Option<Self> {
        assert!(self.is_square(), "the matrix is not square");
        self.lu().solve(b)
    }

    /// `Aⁿ` for any integer `n`: [`pow`](Matrix::pow) of the inverse for a negative `n`, `None`
    /// when that doesn't exist.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn powi(&self, n: i32) -> Option<Self> {
        if n >= 0 {
            Some(self.pow(n.unsigned_abs()))
        } else {
            Some(self.inverse()?.pow(n.unsigned_abs()))
        }
    }
}

impl<T: Scalar> Lu<T> {
    /// `L`, `m × min(m, n)`, lower triangular with ones on the diagonal.
    pub fn l(&self) -> Matrix<T> {
        let (m, n) = self.lu.shape();
        Matrix::from_fn(m, m.min(n), |i, j| match i.cmp(&j) {
            core::cmp::Ordering::Greater => self.lu[(i, j)],
            core::cmp::Ordering::Equal => T::one(),
            core::cmp::Ordering::Less => T::zero(),
        })
    }

    /// `U`, `min(m, n) × n`, upper triangular.
    pub fn u(&self) -> Matrix<T> {
        let (m, n) = self.lu.shape();
        Matrix::from_fn(m.min(n), n, |i, j| {
            if i <= j { self.lu[(i, j)] } else { T::zero() }
        })
    }

    /// `P`, the `m × m` permutation matrix with `P·A = L·U`.
    pub fn p(&self) -> Matrix<T> {
        let m = self.perm.len();
        Matrix::from_fn(m, m, |i, j| {
            if self.perm[i] == j {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// The permutation: row `i` of `P·A` is row `permutation()[i]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Whether a pivot is zero, which for a square matrix makes it singular.
    pub fn is_singular(&self) -> bool {
        let (m, n) = self.lu.shape();
        (0..m.min(n)).any(|i| self.lu[(i, i)] == T::zero())
    }

    /// The determinant of the square matrix: the product of the pivots, negated for an odd
    /// permutation.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn det(&self) -> T {
        assert!(self.lu.is_square(), "the matrix is not square");
        let mut d = T::one();
        for i in 0..self.lu.rows() {
            d *= self.lu[(i, i)];
        }
        if self.odd { -d } else { d }
    }

    /// The solution `X` of `A·X = B` for the square matrix, or `None` when it is singular.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, or `B` has not as many rows.
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        assert!(self.lu.is_square(), "the matrix is not square");
        let n = self.lu.rows();
        assert_eq!(b.rows(), n, "the right-hand side has not as many rows");
        if self.is_singular() {
            return None;
        }
        let mut x = Matrix::from_fn(n, b.cols(), |i, j| b[(self.perm[i], j)]);
        let cols = x.cols();
        trsm(
            Side::Left,
            Uplo::Lower,
            Op::N,
            true,
            n,
            cols,
            &self.lu.data,
            n,
            &mut x.data,
            n,
        );
        trsm(
            Side::Left,
            Uplo::Upper,
            Op::N,
            false,
            n,
            cols,
            &self.lu.data,
            n,
            &mut x.data,
            n,
        );
        Some(x)
    }

    /// The inverse of the square matrix, or `None` when it is singular.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn inverse(&self) -> Option<Matrix<T>> {
        assert!(self.lu.is_square(), "the matrix is not square");
        if self.is_singular() {
            return None;
        }
        // `A⁻¹ = U⁻¹·L⁻¹·P`, as LAPACK's `getri`: `U⁻¹`, then `X·L = U⁻¹` for `X`, then the
        // columns of `X` permuted; about `4n³/3` operations, nearly all of them in products of
        // blocks.
        let n = self.lu.rows();
        let mut x = Matrix::zeros(n, n);
        inverse_from(&self.lu.data, n, &mut x.data);
        // Column `perm[j]` of `X·P` is column `j` of `X`: the cycles of the permutation, a swap
        // of whole columns at a time.
        let mut to = self.perm.clone();
        for j in 0..n {
            while to[j] != j {
                let k = to[j];
                let (a, b) = x.data.split_at_mut(j.max(k) * n);
                a[j.min(k) * n..j.min(k) * n + n].swap_with_slice(&mut b[..n]);
                to.swap(j, k);
            }
        }
        Some(x)
    }
}

/// Writes `X = U⁻¹·L⁻¹` for the factors `lu` of order `n` to `x`, which holds zeros: `U⁻¹`,
/// then `X·L = U⁻¹` for `X`.
fn inverse_from<T: Scalar>(lu: &[T], n: usize, x: &mut [T]) {
    upper_inverse(lu, n, x, n, n);
    trsm(Side::Right, Uplo::Lower, Op::N, true, n, n, lu, n, x, n);
}

/// Triangles up to this order are inverted by a solve against the identity.
const INVERSE_LEAF: usize = 16;

/// Writes the inverse of the upper triangle of order `n` at the start of `u` to the upper
/// triangle of `w`, recursively: `W11 = U11⁻¹`, `W22 = U22⁻¹` and `W12 = -U11⁻¹·U12·U22⁻¹`, the
/// last by two triangular solves from the triangles of `U` itself, so that no block is read
/// where another is written. The lower triangle of `w` is left as it is.
fn upper_inverse<T: Scalar>(u: &[T], ldu: usize, w: &mut [T], ldw: usize, n: usize) {
    if n <= INVERSE_LEAF {
        for j in 0..n {
            for i in 0..=j {
                w[i + j * ldw] = if i == j { T::one() } else { T::zero() };
            }
        }
        trsm(Side::Left, Uplo::Upper, Op::N, false, n, n, u, ldu, w, ldw);
        return;
    }
    let n1 = split(n);
    let n2 = n - n1;
    let (w1, w2) = w.split_at_mut(n1 * ldw);
    for j in 0..n2 {
        let (src, dst) = (&u[(n1 + j) * ldu..], &mut w2[j * ldw..]);
        for (d, s) in dst[..n1].iter_mut().zip(&src[..n1]) {
            *d = -*s;
        }
    }
    let u22 = &u[n1 + n1 * ldu..];
    trsm(
        Side::Left,
        Uplo::Upper,
        Op::N,
        false,
        n1,
        n2,
        u,
        ldu,
        w2,
        ldw,
    );
    trsm(
        Side::Right,
        Uplo::Upper,
        Op::N,
        false,
        n1,
        n2,
        u22,
        ldu,
        w2,
        ldw,
    );
    upper_inverse(u, ldu, w1, ldw, n1);
    upper_inverse(u22, ldu, &mut w2[n1..], ldw, n2);
}

/// [`Matrix::inverse`] of an `N × N` matrix, `N` known, so that its loops unroll: the LU
/// decomposition of a copy on the stack, as [`tiny`] finds it, then `U⁻¹`, `X·L = U⁻¹` and the
/// permutation of the columns, all on the stack, then the only allocation, the inverse.
///
/// The decomposition stops at the first zero pivot, and does not skip the updates by zeros: these
/// change nothing that is finite, and the branches cost more than they save here. The pivots are
/// chosen and the rows swapped by selects, not branches, which keeps the whole matrix in
/// registers. Orders from 5 are functions of their own, as one function holding them all compiled
/// worse; up to 4, the call itself cost a fifth of the time, so they are inlined.
#[inline(never)]
fn tiny_inverse<T: Scalar, const N: usize>(a: &Matrix<T>) -> Option<Matrix<T>> {
    tiny_inverse_inline::<T, N>(a)
}

/// [`tiny_inverse`], inlined. Its loops index rows and columns as the elimination is written: every
/// index is known once they unroll.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn tiny_inverse_inline<T: Scalar, const N: usize>(a: &Matrix<T>) -> Option<Matrix<T>> {
    let mut lu = [[T::zero(); N]; N];
    lu.as_flattened_mut().copy_from_slice(&a.data);
    let mut perm: [usize; N] = core::array::from_fn(|i| i);
    let mut r = [T::zero(); N];
    for j in 0..N {
        // The first element of the largest `|re| + |im|`, the first itself when it is NaN.
        let mut p = j;
        let mut best = lu[j][j].abs1();
        for i in j + 1..N {
            let v = lu[j][i].abs1();
            let greater = v > best;
            best = select_unpredictable(greater, v, best);
            p = select_unpredictable(greater, i, p);
        }
        // The swap by selects, every index known: the whole matrix stays in registers.
        for i in j + 1..N {
            let swap = p == i;
            for col in &mut lu {
                let (x, y) = (col[j], col[i]);
                col[j] = select_unpredictable(swap, y, x);
                col[i] = select_unpredictable(swap, x, y);
            }
            let (x, y) = (perm[j], perm[i]);
            perm[j] = select_unpredictable(swap, y, x);
            perm[i] = select_unpredictable(swap, x, y);
        }
        let d = lu[j][j];
        if d == T::zero() {
            return None;
        }
        r[j] = T::one() / d;
        if r[j].is_finite() {
            for i in j + 1..N {
                lu[j][i] *= r[j];
            }
        } else {
            // A reciprocal that overflows, of a pivot near the bottom of the range: divide.
            for i in j + 1..N {
                lu[j][i] /= d;
            }
        }
        for c in j + 1..N {
            let u = lu[c][j];
            for i in j + 1..N {
                lu[c][i] = lu[c][i].mul_sub(lu[j][i], u);
            }
        }
    }
    // `W = U⁻¹` by columns, each by back substitution with the reciprocals of the diagonal.
    let mut w = [[T::zero(); N]; N];
    for j in 0..N {
        w[j][j] = r[j];
        for i in (0..j).rev() {
            let mut s = T::zero();
            for k in i + 1..=j {
                s = s.mul_sub(lu[k][i], w[j][k]);
            }
            w[j][i] = s * r[i];
        }
    }
    // `X·L = W` from the last column: `X[:, j] = W[:, j] - Σ_{k > j} X[:, k]·L[k, j]`.
    let mut x = [[T::zero(); N]; N];
    for j in (0..N).rev() {
        let mut col = w[j];
        for k in j + 1..N {
            let l = lu[j][k];
            for (c, x) in col.iter_mut().zip(&x[k]) {
                *c = c.mul_sub(*x, l);
            }
        }
        x[j] = col;
    }
    // Column `perm[j]` of `X·P` is column `j` of `X`.
    let mut inverse = vec![T::zero(); N * N];
    for (j, &p) in perm.iter().enumerate() {
        inverse[p * N..p * N + N].copy_from_slice(&x[j]);
    }
    Some(Matrix::from_column_major(N, N, inverse))
}

/// Pivots up to this many are kept on the stack.
const STACK_PIVOTS: usize = 32;

/// Runs `f` on `k` zeros, for the row swaps of a factorization.
fn with_pivots<X>(k: usize, f: impl FnOnce(&mut [usize]) -> X) -> X {
    if k <= STACK_PIVOTS {
        f(&mut [0; STACK_PIVOTS][..k])
    } else {
        f(&mut vec![0; k])
    }
}

/// Factors the `m × n` matrix `a` in place, `P·A = L·U`: `pivots[i]` is the row swapped with
/// row `i`, in turn, for `i < min(m, n)`.
fn factor_all<T: Scalar>(a: &mut [T], m: usize, n: usize, pivots: &mut [usize]) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    if k <= UNBLOCKED {
        unblocked(a, m, k, m, pivots);
    } else {
        with_scratch(work_size(k), |work| factor(a, m, k, m, pivots, work));
    }
    if n > k {
        // The columns right of the square part: the same row swaps, then `L⁻¹`.
        let (left, right) = a.split_at_mut(k * m);
        swap_rows(right, m, n - k, pivots, 0);
        trsm(
            Side::Left,
            Uplo::Lower,
            Op::N,
            true,
            k,
            n - k,
            left,
            m,
            right,
            m,
        );
    }
}

/// Columns narrower than this are factored without recursion.
const NARROW: usize = 16;

/// Matrices of up to this many columns are factored without recursion at all.
const UNBLOCKED: usize = 24;

/// Where the recursion splits `n` columns: the first part a multiple of [`NARROW`], so that the
/// unblocked panels are full.
fn split(n: usize) -> usize {
    (n / 2).next_multiple_of(NARROW).min(n - 1)
}

/// The working memory of [`factor`] of `n` columns: the dense copy of the largest `U12` of the
/// recursion, which an uneven split may leave below its top.
fn work_size(n: usize) -> usize {
    if n <= NARROW {
        return 0;
    }
    let n1 = split(n);
    (n1 * (n - n1)).max(work_size(n1)).max(work_size(n - n1))
}

/// Factors the `m × n` block at the start of `a`, `n ≤ m`, in place, Toledo's way: the left
/// half, then the right half updated by it, each half the same way down to narrow panels.
/// `pivots[i]` is the row swapped with row `i`, within the block. `work` holds the dense copy of
/// `U12` for the product that updates `A22`: at least [`work_size`] elements.
fn factor<T: Scalar>(
    a: &mut [T],
    m: usize,
    n: usize,
    ld: usize,
    pivots: &mut [usize],
    work: &mut [T],
) {
    if n <= NARROW {
        unblocked(a, m, n, ld, pivots);
        return;
    }
    let n1 = split(n);
    let n2 = n - n1;
    factor(a, m, n1, ld, &mut pivots[..n1], work);
    let (left, right) = a.split_at_mut(n1 * ld);
    swap_rows(right, ld, n2, &pivots[..n1], 0);
    // `U12 = L11⁻¹·A12`, then `A22 -= L21·U12`, `U12` read from its copy.
    let u12 = &mut work[..n1 * n2];
    trsm_into(
        Uplo::Lower,
        Op::N,
        true,
        n1,
        n2,
        left,
        ld,
        right,
        ld,
        u12,
        n1,
    );
    gemm(
        m - n1,
        n2,
        n1,
        -T::one(),
        &left[n1..],
        ld,
        Op::N,
        u12,
        n1,
        Op::N,
        T::one(),
        &mut right[n1..],
        ld,
    );
    factor(&mut right[n1..], m - n1, n2, ld, &mut pivots[n1..], work);
    for p in &mut pivots[n1..] {
        *p += n1;
    }
    swap_rows(left, ld, n1, &pivots[n1..], n1);
}

/// Factors a narrow block by rank-one updates, choosing as pivot the element of the largest
/// `|re| + |im|` in each column, as LAPACK does. A zero column is left as it is.
fn unblocked<T: Scalar>(a: &mut [T], m: usize, n: usize, ld: usize, pivots: &mut [usize]) {
    for j in 0..n {
        let p = j + pivot(&a[j * ld + j..j * ld + m]);
        pivots[j] = p;
        if p != j {
            for c in 0..n {
                a.swap(j + c * ld, p + c * ld);
            }
        }
        let (left, right) = a.split_at_mut(((j + 1) * ld).min(a.len()));
        let col = &mut left[j * ld + j..j * ld + m];
        let (d, l) = col.split_first_mut().unwrap();
        if *d == T::zero() {
            continue;
        }
        let r = T::one() / *d;
        if r.is_finite() {
            for x in l.iter_mut() {
                *x *= r;
            }
        } else {
            // A reciprocal that overflows, of a pivot near the bottom of the range: divide, as
            // LAPACK's `getf2` does.
            for x in l.iter_mut() {
                *x /= *d;
            }
        }
        for c in 0..n - j - 1 {
            let col = &mut right[c * ld + j..c * ld + m];
            let (u, x) = col.split_first_mut().unwrap();
            let u = *u;
            if u == T::zero() {
                continue;
            }
            for (x, l) in x.iter_mut().zip(l.iter()) {
                *x = x.mul_sub(*l, u);
            }
        }
    }
}

/// The order up to which square matrices are factored by [`tiny`]: past it, [`unblocked`] ran
/// as fast.
const TINY: usize = 16;

/// [`unblocked`] of an `N × N` matrix, `N` known, so that its loops unroll; the row swaps are
/// applied to `perm`, and the result is whether they make an odd permutation. The matrix is
/// copied to the stack, its pivots chosen and its rows swapped by selects, not branches, every
/// index known, so that it stays in registers: a 5 × 5 decomposition so took 67 ns against 88.
/// Each order is a function of its own. Its loops index rows and columns as the elimination is
/// written: every index is known once they unroll.
#[inline(never)]
#[allow(clippy::needless_range_loop)]
fn tiny<T: Scalar, const N: usize>(data: &mut [T], perm: &mut [usize]) -> bool {
    let mut a = [[T::zero(); N]; N];
    a.as_flattened_mut().copy_from_slice(&data[..N * N]);
    let perm: &mut [usize; N] = (&mut perm[..N]).try_into().unwrap();
    let mut odd = false;
    for j in 0..N {
        // The first element of the largest `|re| + |im|`, the first itself when it is NaN.
        let mut p = j;
        let mut best = a[j][j].abs1();
        for i in j + 1..N {
            let v = a[j][i].abs1();
            let greater = v > best;
            best = select_unpredictable(greater, v, best);
            p = select_unpredictable(greater, i, p);
        }
        odd ^= p != j;
        for i in j + 1..N {
            let swap = p == i;
            for col in &mut a {
                let (x, y) = (col[j], col[i]);
                col[j] = select_unpredictable(swap, y, x);
                col[i] = select_unpredictable(swap, x, y);
            }
            let (x, y) = (perm[j], perm[i]);
            perm[j] = select_unpredictable(swap, y, x);
            perm[i] = select_unpredictable(swap, x, y);
        }
        let d = a[j][j];
        if d == T::zero() {
            continue;
        }
        let r = T::one() / d;
        if r.is_finite() {
            for i in j + 1..N {
                a[j][i] *= r;
            }
        } else {
            // A reciprocal that overflows, of a pivot near the bottom of the range: divide.
            for i in j + 1..N {
                a[j][i] /= d;
            }
        }
        for c in j + 1..N {
            let u = a[c][j];
            if u == T::zero() {
                continue;
            }
            for i in j + 1..N {
                a[c][i] = a[c][i].mul_sub(a[j][i], u);
            }
        }
    }
    data[..N * N].copy_from_slice(a.as_flattened());
    odd
}

/// The index of the pivot of a column: the first element of the largest `|re| + |im|`, the
/// first element itself when it is NaN, as the scan from it with `>` finds.
fn pivot<T: Scalar>(col: &[T]) -> usize {
    let first = col[0].abs1();
    if first.is_nan() {
        return 0;
    }
    // The largest by `max`, which passes over NaNs, then where it first is.
    let best = col.iter().fold(first, |b, x| b.max(x.abs1()));
    col.iter().position(|x| x.abs1() == best).unwrap_or(0)
}

/// Swaps rows `first + i` and `pivots[i]` (an absolute row) of the `n` columns of `a`, in order,
/// a column at a time.
fn swap_rows<T>(a: &mut [T], ld: usize, n: usize, pivots: &[usize], first: usize) {
    for c in 0..n {
        let col = &mut a[c * ld..];
        for (i, &p) in pivots.iter().enumerate() {
            let i = first + i;
            if p != i {
                col.swap(i, p);
            }
        }
    }
}
