//! The products and triangular solves the algorithms are built on, on column-major blocks: a
//! block of `rows × cols` starts at the beginning of its slice, and column `j` of it at `j · ld`.

use super::Scalar;
use core::mem::MaybeUninit;

mod trsm;

/// How a block takes part in a product.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// As it is.
    N,
    /// Transposed.
    T,
    /// Conjugate transposed.
    C,
}

impl Op {
    /// Element `(i, j)` of `op(a)`, for `a` with the leading dimension `ld`.
    #[inline(always)]
    fn at<T: Scalar>(self, a: &[T], ld: usize, i: usize, j: usize) -> T {
        match self {
            Op::N => a[i + j * ld],
            Op::T => a[j + i * ld],
            Op::C => a[j + i * ld].conj(),
        }
    }
}

/// `c = β·c + α·op(a)·op(b)`, with `op(a)` of `m × k`, `op(b)` of `k × n` and `c` of `m × n`.
/// When `β` is zero, `c` is not read, so it may hold anything, NaNs included; when `α` is zero or
/// `k` is, neither `a` nor `b` is. Computed by [`super::gemm`]: small products directly, larger
/// ones by packed blocks.
///
/// # Panics
///
/// Panics if the leading dimension of a nonempty block is less than its rows, or the block reaches
/// past its slice.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub(crate) fn gemm<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    opa: Op,
    b: &[T],
    ldb: usize,
    opb: Op,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    T::gemm(m, n, k, alpha, a, lda, opa, b, ldb, opb, beta, c, ldc);
}

/// Which side of `b` a triangular block multiplies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    /// `op(a)⁻¹ · b`.
    Left,
    /// `b · op(a)⁻¹`.
    Right,
}

/// Which triangle of a square block holds it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Uplo {
    /// On and above the diagonal.
    Upper,
    /// On and below the diagonal.
    Lower,
}

/// `b = op(a)⁻¹ · b` or `b = b · op(a)⁻¹`, for the triangle `uplo` of the square block `a`,
/// whose diagonal is taken as ones when `unit` is set; `b` is `m × n`. The triangle is halved
/// until it is small, so that most of the work is the product of the off-diagonal block; see
/// [`trsm`](mod@trsm) for the leaves.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trsm<T: Scalar>(
    side: Side,
    uplo: Uplo,
    op: Op,
    unit: bool,
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    trsm::solve(side, uplo, op, unit, m, n, a, lda, b, ldb);
}

/// [`trsm`](fn@trsm) on the left, `b = op(a)⁻¹ · b`, the solution also written to the `m × n`
/// block `x` of the leading dimension `ldx`: a dense copy of it for products that update the rest
/// of the columns `b` lies in, which they could not read where it is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trsm_into<T: Scalar>(
    uplo: Uplo,
    op: Op,
    unit: bool,
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    x: &mut [T],
    ldx: usize,
) {
    trsm::solve_into(uplo, op, unit, m, n, a, lda, b, ldb, x, ldx);
}

/// Elements of working memory that [`with_scratch`] takes from the stack rather than allocates.
const SCRATCH: usize = 1024;

/// Runs `f` on `len` zeros of working memory: on the stack when they are few, so that small
/// problems allocate nothing, and allocated otherwise.
pub(crate) fn with_scratch<T: Scalar, X>(len: usize, f: impl FnOnce(&mut [T]) -> X) -> X {
    if len <= SCRATCH {
        let mut s = [const { MaybeUninit::<T>::uninit() }; SCRATCH];
        for v in &mut s[..len] {
            v.write(T::zero());
        }
        // SAFETY: the first `len` elements were just initialized.
        f(unsafe { &mut *(&mut s[..len] as *mut [MaybeUninit<T>] as *mut [T]) })
    } else {
        f(&mut vec![T::zero(); len])
    }
}
