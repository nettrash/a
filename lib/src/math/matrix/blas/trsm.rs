//! The triangular solves of [`trsm`](fn@super::trsm).
//!
//! The triangle is halved until it is at most [`LEAF`] = 16 rows, the first part a multiple of
//! 16, so that most of the work is the products of the off-diagonal blocks. A leaf copies its
//! triangle to the stack first ([`Packed`]), its elements scaled by the reciprocals of the
//! diagonal, so that the substitution is fused multiply-adds with nothing but them on the path
//! from one unknown to the next; the scaling by the diagonal is applied once at the end. For
//! `f64` and `f32` the leaves are NEON kernels (`aarch64`): on the left, two columns of `b` in
//! vector registers at a time, each solved row updating them by a lane of its own vector; on
//! the right, 16 rows at a time, each group of columns summed from those solved before it. The
//! portable leaves, for complex elements, other targets and Miri, work in groups of four rows.
//!
//! A triangle with a zero, a subnormal or a non-finite element on its diagonal, whose reciprocal
//! would not be safe to use, and columns whose results are not all finite, are solved by plain
//! substitution with divisions instead ([`substitute`]), so that infinities and NaNs reach only
//! the elements they would. So are the tiniest triangles, up to [`TINY`], when that divides
//! little: packing them would cost more than it saves.
//!
//! On the left, `op(a)⁻¹·b`, the rows solved first update the rows below them; as they share
//! their columns with these, a product cannot read them where they are, so the leaves also write
//! the solution to a dense buffer that the products read, one for the whole solve. On the right,
//! `b·op(a)⁻¹`, the halves are columns, which lie apart.
//!
//! # Speed
//!
//! Measured on an Apple M4 Max, one core, `f64`, in GFLOP/s (`m²n` on the left, `mn²` on the
//! right), against faer 0.24's triangular solves (`Par::Seq`) and the original leaves, scalar
//! substitution with the solved rows copied at every level:
//!
//! | solve | ours | faer | before |
//! | --- | --- | --- | --- |
//! | `L⁻¹·B`, unit, 16 × 16 by 16 | 18.5 | 14.7 | 4.8 |
//! | `L⁻¹·B`, unit, 64 × 64 by 64 | 35.6 | 31.8 | 11.8 |
//! | `L⁻¹·B`, unit, 512 × 512 by 512 | 51.6 | 39.8 | 35.5 |
//! | `U⁻¹·B`, 256 × 256 by 256 | 47.4 | 39.8 | 20.5 |
//! | `U⁻¹·b`, 64 × 64 by 1 | 6.3 | 4.5 | 4.8 |
//! | `B·L⁻*`, 32 by 16 × 16 | 17.2 | 16.6 | 8.9 |
//! | `B·L⁻*`, 64 by 32 × 32 | 26.5 | 24.8 | 19.0 |
//! | `B·L⁻*`, 256 by 128 × 128 | 43.6 | 37.4 | 40.6 |
//! | `B·L⁻¹`, unit, 64 by 64 × 64 | 33.8 | 28.7 | 28.8 |
//! | `B·L⁻¹`, unit, 512 by 512 × 512 | 50.5 | 46.4 | 50.3 |
//!
//! The right side gained least: its old leaves already ran down the columns of `b`, which the
//! compiler vectorized. Accelerate, on the matrix unit, ran 3 to 7 times faster from 64 up.

use super::{Op, Side, Uplo, gemm, with_scratch};
use crate::math::matrix::Scalar;
use core::mem::MaybeUninit;

// The leaves of `f64` and `f32` in NEON intrinsics on little-endian aarch64, as the products'
// kernels; the portable leaves serve the other types and targets, and Miri.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod aarch64;

/// The most rows (on the left) or columns (on the right) of a triangle solved by substitution.
pub(super) const LEAF: usize = 16;

/// Triangles up to this order are solved by plain substitution, which costs less than packing
/// them when it divides little: on the left, for one or two columns of `b`, or up to 8 when the
/// diagonal is taken as ones; on the right, when the diagonal is taken as ones.
const TINY: usize = 4;

/// The rows of `b` a leaf on the right solves at a time.
const ROWS: usize = 16;

/// `b = op(a)⁻¹·b` or `b = b·op(a)⁻¹`: see [`trsm`](fn@super::trsm).
#[allow(clippy::too_many_arguments)]
pub(super) fn solve<T: Scalar>(
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
    if m == 0 || n == 0 {
        return;
    }
    let t = Tri::new(a, lda, uplo, op, unit);
    match side {
        Side::Left if m <= LEAF => left_leaf(&t, m, n, b, ldb, None),
        Side::Left => with_scratch(m * n, |x| left(&t, m, n, b, ldb, x, m)),
        Side::Right => right(&t, m, n, b, ldb),
    }
}

/// `b = op(a)⁻¹·b`, the solution also written to `x`: see [`trsm_into`](super::trsm_into).
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_into<T: Scalar>(
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
    if m == 0 || n == 0 {
        return;
    }
    let t = Tri::new(a, lda, uplo, op, unit);
    left(&t, m, n, b, ldb, x, ldx);
}

/// A triangular block `op(a)` of a [`trsm`](fn@super::trsm).
struct Tri<'a, T> {
    /// The block `a` holding it.
    a: &'a [T],
    /// The leading dimension of `a`.
    lda: usize,
    /// How `a` is taken.
    op: Op,
    /// Whether `op(a)` is lower triangular.
    lower: bool,
    /// Whether the diagonal is taken as ones.
    unit: bool,
}

impl<'a, T: Scalar> Tri<'a, T> {
    /// The triangle `uplo` of `op(a)`.
    fn new(a: &'a [T], lda: usize, uplo: Uplo, op: Op, unit: bool) -> Self {
        Tri {
            a,
            lda,
            op,
            // `op(a)` is lower triangular when exactly one of these holds.
            lower: (uplo == Uplo::Lower) == (op == Op::N),
            unit,
        }
    }

    /// Element `(i, j)` of `op(a)`.
    #[inline(always)]
    fn at(&self, i: usize, j: usize) -> T {
        self.op.at(self.a, self.lda, i, j)
    }

    /// The block of `op(a)` from row `i` and column `j`, where it is held: in `a` at `(i, j)`,
    /// or at `(j, i)` when transposed.
    fn block(&self, i: usize, j: usize) -> &'a [T] {
        if self.op == Op::N {
            &self.a[i + j * self.lda..]
        } else {
            &self.a[j + i * self.lda..]
        }
    }

    /// The diagonal block of `op(a)` from `(k, k)`.
    fn sub(&self, k: usize) -> Self {
        Tri {
            a: &self.a[k + k * self.lda..],
            ..*self
        }
    }
}

impl<T> Clone for Tri<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Tri<'_, T> {}

/// Where the recursion splits `k` rows or columns: the first part a multiple of [`LEAF`], so
/// that the leaves are full.
fn split(k: usize) -> usize {
    (k / 2).next_multiple_of(LEAF).min(k - 1)
}

/// `b = op(a)⁻¹·b` for `m > 0` rows, the solution also written to `x`, of the leading
/// dimension `ldx`, which the products read.
fn left<T: Scalar>(
    t: &Tri<T>,
    m: usize,
    n: usize,
    b: &mut [T],
    ldb: usize,
    x: &mut [T],
    ldx: usize,
) {
    if m <= LEAF {
        left_leaf(t, m, n, b, ldb, Some((x, ldx)));
        return;
    }
    let k1 = split(m);
    let k2 = m - k1;
    let (a11, a22) = (t, &t.sub(k1));
    let minus_one = -T::one();
    if t.lower {
        left(a11, k1, n, b, ldb, x, ldx);
        gemm(
            k2,
            n,
            k1,
            minus_one,
            t.block(k1, 0),
            t.lda,
            t.op,
            x,
            ldx,
            Op::N,
            T::one(),
            &mut b[k1..],
            ldb,
        );
        left(a22, k2, n, &mut b[k1..], ldb, &mut x[k1..], ldx);
    } else {
        left(a22, k2, n, &mut b[k1..], ldb, &mut x[k1..], ldx);
        gemm(
            k1,
            n,
            k2,
            minus_one,
            t.block(0, k1),
            t.lda,
            t.op,
            &x[k1..],
            ldx,
            Op::N,
            T::one(),
            b,
            ldb,
        );
        left(a11, k1, n, b, ldb, x, ldx);
    }
}

/// `b = b·op(a)⁻¹` for `n > 0` columns.
fn right<T: Scalar>(t: &Tri<T>, m: usize, n: usize, b: &mut [T], ldb: usize) {
    if n <= LEAF {
        right_leaf(t, m, n, b, ldb);
        return;
    }
    let k1 = split(n);
    let k2 = n - k1;
    let (a11, a22) = (t, &t.sub(k1));
    let minus_one = -T::one();
    let (b1, b2) = b.split_at_mut(k1 * ldb);
    if t.lower {
        // `x·op(a) = b` from the last columns, which the lower triangle leaves alone.
        right(a22, m, k2, b2, ldb);
        gemm(
            m,
            k1,
            k2,
            minus_one,
            b2,
            ldb,
            Op::N,
            t.block(k1, 0),
            t.lda,
            t.op,
            T::one(),
            b1,
            ldb,
        );
        right(a11, m, k1, b1, ldb);
    } else {
        right(a11, m, k1, b1, ldb);
        gemm(
            m,
            k2,
            k1,
            minus_one,
            b1,
            ldb,
            Op::N,
            t.block(0, k1),
            t.lda,
            t.op,
            T::one(),
            b2,
            ldb,
        );
        right(a22, m, k2, b2, ldb);
    }
}

/// A triangle of order `k <= LEAF` ready for substitution: `l[i + p · LEAF]` is element `(i, p)`
/// of the triangle times `r[p]`, the reciprocal of its diagonal element (1 when the diagonal is
/// taken as ones), and the triangle is lower when `forward` is set. The unknowns `y` of the
/// substitution `y_i = b_i - Σ l_ip·y_p` are then the solution times the diagonal.
///
/// The scaled elements may overflow where the substitution with divisions would not: the
/// substitutions check that their results are finite, and run with divisions when they are not.
struct Packed<T> {
    /// The scaled triangle, by columns, zeros outside it: the first [`columns`](Self::columns)
    /// columns are written, as many as the substitutions read.
    l: [MaybeUninit<T>; LEAF * LEAF],
    /// The columns of `l` written.
    columns: usize,
    /// The reciprocals of the diagonal.
    r: [T; LEAF],
    /// Whether the triangle is lower, solved from the first row down.
    forward: bool,
}

impl<T: Scalar> Packed<T> {
    /// An empty triangle, to [`fill`](Self::fill) where it lies, as it is too large to move.
    fn new() -> Self {
        Packed {
            l: [const { MaybeUninit::uninit() }; LEAF * LEAF],
            columns: 0,
            r: [T::one(); LEAF],
            forward: false,
        }
    }

    /// Packs `op(a)` (`transpose` unset) or its transpose, of order `k`, or `false` when the
    /// substitution by reciprocals would not be safe: a diagonal element is zero, not finite or
    /// so small that its reciprocal overflows. The columns past `k` that the substitutions
    /// read, up to the next power of two from 4, are zeros: only these are written, so that a
    /// small triangle costs little.
    fn fill(&mut self, t: &Tri<T>, k: usize, transpose: bool) -> bool {
        self.columns = k.next_power_of_two().clamp(4, LEAF);
        for v in &mut self.l[..self.columns * LEAF] {
            v.write(T::zero());
        }
        self.forward = t.lower != transpose;
        let (a, lda, op) = (t.a, t.lda, t.op);
        let conj = op == Op::C;
        // Column `p` of the packed triangle is a column of `a` when `op(a)` is `a` and it is not
        // transposed, or `op(a)` is transposed and it is; a row of `a` otherwise.
        let column = (op == Op::N) != transpose;
        for p in 0..k {
            if !t.unit {
                let d = a[p + p * lda];
                let d = if conj { d.conj() } else { d };
                let q = T::one() / d;
                // A reciprocal that is finite and nonzero, of a finite nonzero element: then the
                // products by it round as the divisions would, up to an ulp.
                if !(d.is_finite() && q.is_finite()) || q == T::zero() {
                    return false;
                }
                self.r[p] = q;
            }
            let rp = self.r[p];
            let rows = if self.forward { p + 1..k } else { 0..p };
            let dst = &mut self.l[p * LEAF..p * LEAF + LEAF];
            if column {
                let src = &a[p * lda..];
                for i in rows {
                    let v = if conj { src[i].conj() } else { src[i] };
                    dst[i].write(v * rp);
                }
            } else {
                for i in rows {
                    let v = a[p + i * lda];
                    let v = if conj { v.conj() } else { v };
                    dst[i].write(v * rp);
                }
            }
        }
        true
    }

    /// The written columns of the scaled triangle.
    fn l(&self) -> &[T] {
        let l = &self.l[..self.columns * LEAF];
        // SAFETY: the first `columns` columns are written by `fill`.
        unsafe { &*(l as *const [MaybeUninit<T>] as *const [T]) }
    }
}

/// `b = op(a)⁻¹·b` for `m <= LEAF` rows, by substitution, also written to `x` when given.
fn left_leaf<T: Scalar>(
    t: &Tri<T>,
    m: usize,
    n: usize,
    b: &mut [T],
    ldb: usize,
    mut x: Option<(&mut [T], usize)>,
) {
    if m <= TINY && (n <= 2 || (t.unit && n <= 8)) {
        substitute(Side::Left, t, m, n, b, ldb);
        if let Some((x, ldx)) = x {
            for j in 0..n {
                x[j * ldx..j * ldx + m].copy_from_slice(&b[j * ldb..j * ldb + m]);
            }
        }
        return;
    }
    let mut packed = Packed::new();
    let p = packed.fill(t, m, false).then_some(&packed);
    let mut j = 0;
    while j < n {
        let c = if n - j >= 2 { 2 } else { 1 };
        let done = p.as_ref().is_some_and(|p| {
            let x = x.as_mut().map(|(x, l)| (&mut **x, *l));
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            let x = match left_neon(p, t.unit, m, j, c, b, ldb, x) {
                Ok(done) => return done,
                Err(x) => x,
            };
            match (m.div_ceil(4), c) {
                (1, 2) => left_columns::<T, 1, 2>(p, t.unit, m, j, b, ldb, x),
                (1, _) => left_columns::<T, 1, 1>(p, t.unit, m, j, b, ldb, x),
                (2, 2) => left_columns::<T, 2, 2>(p, t.unit, m, j, b, ldb, x),
                (2, _) => left_columns::<T, 2, 1>(p, t.unit, m, j, b, ldb, x),
                (3, 2) => left_columns::<T, 3, 2>(p, t.unit, m, j, b, ldb, x),
                (3, _) => left_columns::<T, 3, 1>(p, t.unit, m, j, b, ldb, x),
                (_, 2) => left_columns::<T, 4, 2>(p, t.unit, m, j, b, ldb, x),
                (_, _) => left_columns::<T, 4, 1>(p, t.unit, m, j, b, ldb, x),
            }
        });
        if !done {
            let b = &mut b[j * ldb..];
            substitute(Side::Left, t, m, c, b, ldb);
            if let Some((x, ldx)) = x.as_mut() {
                for q in 0..c {
                    let x = &mut x[(j + q) * *ldx..];
                    x[..m].copy_from_slice(&b[q * ldb..q * ldb + m]);
                }
            }
        }
        j += c;
    }
}

/// Columns `j..j + c` of [`left_leaf`] by the NEON kernels, for `f64` and `f32`: whether they
/// were solved, as [`left_columns`] tells, or `x` back for the other types.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[allow(clippy::too_many_arguments)]
fn left_neon<'x, T: Scalar>(
    p: &Packed<T>,
    unit: bool,
    m: usize,
    j: usize,
    c: usize,
    b: &mut [T],
    ldb: usize,
    x: Option<(&'x mut [T], usize)>,
) -> Result<bool, Option<(&'x mut [T], usize)>> {
    use core::any::TypeId;
    let (is_f64, is_f32) = (
        TypeId::of::<T>() == TypeId::of::<f64>(),
        TypeId::of::<T>() == TypeId::of::<f32>(),
    );
    if !(is_f64 || is_f32) {
        return Err(x);
    }
    assert!((1..=LEAF).contains(&m) && (1..=2).contains(&c));
    assert!((j + c - 1) * ldb + m <= b.len());
    // The kernels read the columns up to the vectors' rows, at most the next power of two.
    assert!(p.l().len() >= LEAF * m.next_power_of_two().clamp(4, LEAF));
    let bp = b[j * ldb..].as_mut_ptr();
    let (xp, ldx) = match x {
        Some((x, ldx)) => {
            assert!((j + c - 1) * ldx + m <= x.len());
            (x[j * ldx..].as_mut_ptr(), ldx)
        }
        None => (core::ptr::null_mut(), 0),
    };
    let (l, r, f) = (p.l(), &p.r, p.forward);
    // SAFETY: `T` is the type each branch casts to, and the columns lie in the slices, as
    // checked.
    unsafe {
        if is_f64 {
            let (l, r) = (
                &*(l as *const [T] as *const [f64]),
                &*(r as *const [T; LEAF]).cast(),
            );
            let (b, x) = (bp.cast::<f64>(), xp.cast::<f64>());
            Ok(match (m, c) {
                (..=2, 1) => aarch64::left_f64::<1, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=2, _) => aarch64::left_f64::<1, 2>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=4, 1) => aarch64::left_f64::<2, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=4, _) => aarch64::left_f64::<2, 2>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=8, 1) => aarch64::left_f64::<4, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=8, _) => aarch64::left_f64::<4, 2>(l, r, f, unit, m, b, ldb, x, ldx),
                (_, 1) => aarch64::left_f64::<8, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (_, _) => aarch64::left_f64::<8, 2>(l, r, f, unit, m, b, ldb, x, ldx),
            })
        } else {
            let (l, r) = (
                &*(l as *const [T] as *const [f32]),
                &*(r as *const [T; LEAF]).cast(),
            );
            let (b, x) = (bp.cast::<f32>(), xp.cast::<f32>());
            Ok(match (m, c) {
                (..=4, 1) => aarch64::left_f32::<1, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=4, _) => aarch64::left_f32::<1, 2>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=8, 1) => aarch64::left_f32::<2, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (..=8, _) => aarch64::left_f32::<2, 2>(l, r, f, unit, m, b, ldb, x, ldx),
                (_, 1) => aarch64::left_f32::<4, 1>(l, r, f, unit, m, b, ldb, x, ldx),
                (_, _) => aarch64::left_f32::<4, 2>(l, r, f, unit, m, b, ldb, x, ldx),
            })
        }
    }
}

/// Columns `j..j + C` of [`left_leaf`] for `m <= 4·G` rows, solved together so that their
/// substitutions overlap, or `false`, with `b` left as it was, when a value is not finite: the
/// substitution then runs with divisions, so that infinities and NaNs reach only the elements
/// they would.
///
/// The rows go in `G` groups of four, held in registers: each group is solved, then updates the
/// groups after it, a block of four by four. The rows past `m` are zeros, which the zeros of the
/// packed triangle there keep.
#[inline(always)]
fn left_columns<T: Scalar, const G: usize, const C: usize>(
    p: &Packed<T>,
    unit: bool,
    m: usize,
    j: usize,
    b: &mut [T],
    ldb: usize,
    x: Option<(&mut [T], usize)>,
) -> bool {
    assert!(m <= 4 * G && G * 4 <= LEAF);
    // `y[c][g][k]`: row `4g + k` of column `j + c`.
    let mut y = [[[T::zero(); 4]; G]; C];
    for (c, y) in y.iter_mut().enumerate() {
        let col = &b[(j + c) * ldb..][..m];
        for (i, v) in col.iter().enumerate() {
            y[i / 4][i % 4] = *v;
        }
    }
    // Element `(i, q)` of the scaled triangle.
    let pl = p.l();
    let l = |i: usize, q: usize| pl[i + q * LEAF];
    if p.forward {
        for g in 0..G {
            // The triangle of the group, then its products into the groups below.
            for q in 0..4 {
                for k in q + 1..4 {
                    let lk = l(4 * g + k, 4 * g + q);
                    for y in &mut y {
                        y[g][k] = y[g][k].mul_sub(lk, y[g][q]);
                    }
                }
            }
            for h in g + 1..G {
                for q in 0..4 {
                    for k in 0..4 {
                        let lk = l(4 * h + k, 4 * g + q);
                        for y in &mut y {
                            y[h][k] = y[h][k].mul_sub(lk, y[g][q]);
                        }
                    }
                }
            }
        }
    } else {
        for g in (0..G).rev() {
            for q in (0..4).rev() {
                for k in 0..q {
                    let lk = l(4 * g + k, 4 * g + q);
                    for y in &mut y {
                        y[g][k] = y[g][k].mul_sub(lk, y[g][q]);
                    }
                }
            }
            for h in 0..g {
                for q in 0..4 {
                    for k in 0..4 {
                        let lk = l(4 * h + k, 4 * g + q);
                        for y in &mut y {
                            y[h][k] = y[h][k].mul_sub(lk, y[g][q]);
                        }
                    }
                }
            }
        }
    }
    if !y.iter().flatten().flatten().all(|v| v.is_finite()) {
        return false;
    }
    if !unit {
        for y in &mut y {
            for (g, y) in y.iter_mut().enumerate() {
                for (k, y) in y.iter_mut().enumerate() {
                    *y *= p.r[4 * g + k];
                }
            }
        }
    }
    for (c, y) in y.iter().enumerate() {
        let col = &mut b[(j + c) * ldb..][..m];
        for (i, v) in col.iter_mut().enumerate() {
            *v = y[i / 4][i % 4];
        }
    }
    if let Some((x, ldx)) = x {
        for (c, y) in y.iter().enumerate() {
            let col = &mut x[(j + c) * ldx..][..m];
            for (i, v) in col.iter_mut().enumerate() {
                *v = y[i / 4][i % 4];
            }
        }
    }
    true
}

/// `b = b·op(a)⁻¹` for `n <= LEAF` columns, by substitution, [`ROWS`] rows at a time.
fn right_leaf<T: Scalar>(t: &Tri<T>, m: usize, n: usize, b: &mut [T], ldb: usize) {
    if n <= TINY && t.unit {
        substitute(Side::Right, t, m, n, b, ldb);
        return;
    }
    // `x·op(a) = b` is `op(a)ᵀ·xᵀ = bᵀ`: the substitution runs on the transpose.
    let mut p = Packed::new();
    if !p.fill(t, n, true) {
        substitute(Side::Right, t, m, n, b, ldb);
        return;
    }
    let mut i = 0;
    while i < m {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if let Some((rows, done)) = right_neon(&p, t.unit, m - i, n, &mut b[i..], ldb) {
            if !done {
                substitute(Side::Right, t, rows, n, &mut b[i..], ldb);
            }
            i += rows;
            continue;
        }
        let (rows, done) = if m - i >= ROWS {
            (ROWS, right_rows::<T, ROWS>(&p, t.unit, n, i, b, ldb))
        } else if m - i >= 4 {
            (4, right_rows::<T, 4>(&p, t.unit, n, i, b, ldb))
        } else {
            (1, right_rows::<T, 1>(&p, t.unit, n, i, b, ldb))
        };
        if !done {
            substitute(Side::Right, t, rows, n, &mut b[i..], ldb);
        }
        i += rows;
    }
}

/// The first rows of [`right_leaf`] by the NEON kernels, for `f64` and `f32`: how many rows
/// they took and whether they solved them, or `None` for the other types and for fewer rows
/// than a vector holds.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
fn right_neon<T: Scalar>(
    p: &Packed<T>,
    unit: bool,
    m: usize,
    n: usize,
    b: &mut [T],
    ldb: usize,
) -> Option<(usize, bool)> {
    use core::any::TypeId;
    let is_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
    let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
    let lanes = if is_f64 { 2 } else { 4 };
    if !(is_f64 || is_f32) || m < lanes {
        return None;
    }
    let rows = if m >= 16 { 16 } else { lanes };
    assert!((1..=LEAF).contains(&n) && (n - 1) * ldb + rows <= b.len());
    assert!(p.l().len() >= LEAF * n.next_power_of_two().clamp(4, LEAF));
    let (l, r, f) = (p.l(), &p.r, p.forward);
    let bp = b.as_mut_ptr();
    // SAFETY: `T` is the type each branch casts to, and the rows lie in the slice, as checked.
    let done = unsafe {
        let r = r as *const [T; LEAF];
        if is_f64 {
            let (l, r, b) = (
                &*(l as *const [T] as *const [f64]),
                &*r.cast(),
                bp.cast::<f64>(),
            );
            if rows == 16 {
                aarch64::right_f64::<8>(l, r, f, unit, n, b, ldb)
            } else {
                aarch64::right_f64::<1>(l, r, f, unit, n, b, ldb)
            }
        } else {
            let (l, r, b) = (
                &*(l as *const [T] as *const [f32]),
                &*r.cast(),
                bp.cast::<f32>(),
            );
            if rows == 16 {
                aarch64::right_f32::<4>(l, r, f, unit, n, b, ldb)
            } else {
                aarch64::right_f32::<1>(l, r, f, unit, n, b, ldb)
            }
        }
    };
    Some((rows, done))
}

/// Rows `i..i + R` of [`right_leaf`]: each unknown column is summed in registers from those
/// solved before it. `false`, with `b` left as it was, when a value is not finite.
#[inline(always)]
fn right_rows<T: Scalar, const R: usize>(
    p: &Packed<T>,
    unit: bool,
    n: usize,
    i: usize,
    b: &mut [T],
    ldb: usize,
) -> bool {
    assert!(n <= LEAF);
    let pl = p.l();
    let mut y = [[T::zero(); R]; LEAF];
    let solve = |j: usize, y: &mut [[T; R]; LEAF], before: core::ops::Range<usize>| {
        let s = i + j * ldb;
        let mut acc: [T; R] = b[s..s + R].try_into().unwrap();
        for q in before {
            let l = pl[j + q * LEAF];
            for (acc, y) in acc.iter_mut().zip(&y[q]) {
                *acc = acc.mul_sub(*y, l);
            }
        }
        y[j] = acc;
    };
    if p.forward {
        for j in 0..n {
            solve(j, &mut y, 0..j);
        }
    } else {
        for j in (0..n).rev() {
            solve(j, &mut y, j + 1..n);
        }
    }
    if !y[..n].iter().flatten().all(|v| v.is_finite()) {
        return false;
    }
    for (j, y) in y.iter().enumerate().take(n) {
        let s = i + j * ldb;
        let col = &mut b[s..s + R];
        if unit {
            col.copy_from_slice(y);
        } else {
            for (b, y) in col.iter_mut().zip(y) {
                *b = *y * p.r[j];
            }
        }
    }
    true
}

/// [`trsm`](fn@super::trsm) of a small triangle by substitution with divisions, element by
/// element: the path for triangles whose reciprocals of the diagonal are not safe to use.
fn substitute<T: Scalar>(side: Side, t: &Tri<T>, m: usize, n: usize, b: &mut [T], ldb: usize) {
    let (lower, unit) = (t.lower, t.unit);
    let at = |i: usize, j: usize| t.at(i, j);
    match side {
        Side::Left => {
            for j in 0..n {
                let col = &mut b[j * ldb..j * ldb + m];
                if lower {
                    for i in 0..m {
                        let mut s = col[i];
                        for (p, x) in col[..i].iter().enumerate() {
                            s -= at(i, p) * *x;
                        }
                        col[i] = if unit { s } else { s / at(i, i) };
                    }
                } else {
                    for i in (0..m).rev() {
                        let mut s = col[i];
                        for (p, x) in col.iter().enumerate().skip(i + 1) {
                            s -= at(i, p) * *x;
                        }
                        col[i] = if unit { s } else { s / at(i, i) };
                    }
                }
            }
        }
        Side::Right => {
            // Column `j` of `x = b · op(a)⁻¹` solves `x · op(a) = b` from the side `op(a)` is
            // triangular towards.
            let order: Vec<usize> = if lower {
                (0..n).rev().collect()
            } else {
                (0..n).collect()
            };
            for &j in &order {
                for &p in &order {
                    if p == j {
                        break;
                    }
                    let x = at(p, j);
                    for i in 0..m {
                        let v = b[i + p * ldb];
                        b[i + j * ldb] -= v * x;
                    }
                }
                if !unit {
                    let d = at(j, j);
                    for i in 0..m {
                        b[i + j * ldb] /= d;
                    }
                }
            }
        }
    }
}
