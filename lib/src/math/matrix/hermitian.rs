//! The Hermitian eigenproblem: a blocked reduction to a real symmetric tridiagonal matrix, its
//! eigenvalues and eigenvectors by divide and conquer, and the reflections applied back to them
//! in blocks; small matrices without the blocks, in [`small`].

use super::blas::{Op, gemm};
use super::element::sealed::Scalar as _;
use super::householder::{Block, reflector};
use super::tridiagonal;
use super::{Matrix, Scalar};
use crate::math::complex::Float;

pub(super) mod small;

/// The width of the panels of the tridiagonal reduction.
const NB: usize = 32;

/// Trailing blocks up to this order are reduced without panels: the panels' products of blocks
/// overtake the rank-two updates only past it, as measured, sooner for single precision complex
/// elements.
fn unblocked_largest<T: Scalar>() -> usize {
    if T::IS_COMPLEX && size_of::<T::Real>() == 4 {
        128
    } else {
        192
    }
}

/// The width of the blocks of reflections applied back to the eigenvectors.
const BACK: usize = 32;

/// The eigenvalues of the Hermitian `a`, ascending, and its eigenvectors when `want_v` is set.
pub(crate) fn hermitian<T: Scalar>(
    a: &Matrix<T>,
    want_v: bool,
) -> (Vec<T::Real>, Option<Matrix<T>>) {
    let n = a.rows();
    if n <= small::largest::<T>(want_v) {
        return small::hermitian(a, want_v);
    }
    let nan = || {
        let nan = T::from_parts(T::Real::NAN, T::Real::NAN);
        let v = want_v.then(|| Matrix::from_fn(n, n, |_, _| nan));
        (vec![T::Real::NAN; n], v)
    };
    if a.data.iter().any(|x| !x.is_finite()) {
        return nan();
    }
    // The lower triangle, scaled into a range where the reduction neither overflows nor
    // underflows, as LAPACK's `heevd`.
    let mut h = vec![T::zero(); n * n];
    let mut norm = T::Real::ZERO;
    for j in 0..n {
        for i in j..n {
            let x = a.data[i + j * n];
            h[i + j * n] = x;
            norm = norm.max(x.abs());
        }
    }
    let eps = T::Real::EPSILON * T::Real::HALF;
    let rmin = (T::Real::MIN_POSITIVE / eps).sqrt();
    let rmax = (T::Real::ONE / (T::Real::MIN_POSITIVE / eps)).sqrt();
    let scale = if norm > T::Real::ZERO && norm < rmin {
        rmin / norm
    } else if norm > rmax {
        rmax / norm
    } else {
        T::Real::ONE
    };
    if scale != T::Real::ONE {
        for j in 0..n {
            for x in &mut h[j * n + j..(j + 1) * n] {
                *x = x.scale(scale);
            }
        }
    }
    let (mut d, mut e, taus) = tridiagonalize(&mut h, n);
    let unscale = |d: &mut Vec<T::Real>| {
        if scale != T::Real::ONE {
            let r = T::Real::ONE / scale;
            for x in d.iter_mut() {
                *x *= r;
            }
        }
    };
    if !want_v {
        if !tridiagonal::values(&mut d, &mut e) {
            return nan();
        }
        unscale(&mut d);
        return (d, None);
    }
    let Some(z) = tridiagonal::eigen::<T::Re>(&mut d, &mut e) else {
        return nan();
    };
    let mut v: Vec<T> = z.iter().map(|x| T::from_real(x.re())).collect();
    back_transform(&h, n, &taus, &mut v);
    unscale(&mut d);
    (d, Some(Matrix::from_column_major(n, n, v)))
}

/// Reduces the Hermitian `n × n` matrix `h`, from its lower triangle, to the real symmetric
/// tridiagonal `T = Q*·A·Q`, `Q = H₁·H₂·…·H_{n-1}`, as LAPACK's `hetrd`: returns the diagonal,
/// the subdiagonal and the factors `τ` of the reflections, whose vectors are left below the
/// subdiagonal of `h`. Panels of [`NB`] columns are reduced with the trailing matrix untouched
/// but for the matrix-vector products, which accumulate the update `A -= V·W* + W·V*` as LAPACK's
/// `latrd`; it is then made by products of blocks.
fn tridiagonalize<T: Scalar>(h: &mut [T], n: usize) -> (Vec<T::Real>, Vec<T::Real>, Vec<T>) {
    let mut d = vec![T::Real::ZERO; n];
    let mut e = vec![T::Real::ZERO; n.saturating_sub(1)];
    let mut taus = vec![T::zero(); n.saturating_sub(1)];
    let mut p0 = 0;
    let (mut w, mut x, mut y) = (Vec::new(), Vec::new(), Vec::new());
    let unblocked_from = unblocked_largest::<T>();
    while n - p0 > unblocked_from {
        let m = n - p0;
        w.clear();
        w.resize(m * NB, T::zero());
        panel(h, n, p0, &mut w, &mut d, &mut e, &mut taus);
        // `A₂₂ -= V·W* + W·V*` on the lower triangle, as `[V W]·[W V]*`.
        let m2 = m - NB;
        x.resize(m2 * 2 * NB, T::zero());
        y.resize(m2 * 2 * NB, T::zero());
        for j in 0..NB {
            let v = &h[(p0 + j) * n + p0 + NB..(p0 + j + 1) * n];
            let wj = &w[j * m + NB..(j + 1) * m];
            x[j * m2..(j + 1) * m2].copy_from_slice(v);
            x[(NB + j) * m2..(NB + j + 1) * m2].copy_from_slice(wj);
            y[j * m2..(j + 1) * m2].copy_from_slice(wj);
            y[(NB + j) * m2..(NB + j + 1) * m2].copy_from_slice(v);
        }
        let off = (p0 + NB) * (n + 1);
        her2k_lower(m2, 2 * NB, &x, &y, &mut h[off..], n);
        p0 += NB;
    }
    unblocked(h, n, p0, &mut d, &mut e, &mut taus, &mut vec![T::zero(); n]);
    (d, e, taus)
}

/// `c -= x·y*` on the lower triangle of the `m × m` block `c`, `x` and `y` of `m × k` with the
/// leading dimension `m`: block columns, each from the diagonal down, by [`gemm`]. The diagonal
/// blocks are updated whole, above their diagonals too, where nothing reads.
fn her2k_lower<T: Scalar>(m: usize, k: usize, x: &[T], y: &[T], c: &mut [T], ldc: usize) {
    const COLUMNS: usize = 64;
    let mut j0 = 0;
    while j0 < m {
        let w = COLUMNS.min(m - j0);
        let (one, minus) = (T::one(), -T::one());
        gemm(
            m - j0,
            w,
            k,
            minus,
            &x[j0..],
            m,
            Op::N,
            &y[j0..],
            m,
            Op::C,
            one,
            &mut c[j0 * (ldc + 1)..],
            ldc,
        );
        j0 += w;
    }
}

/// Reduces the [`NB`] columns from `p0` of `h`, as LAPACK's `latrd`: the columns are updated by
/// the reflections before them in the panel as they come, and `w` (`m × NB`, `m = n - p0`) gets
/// the `W` of the update `A -= V·W* + W·V*` of the trailing matrix. The reflection vectors are
/// left in `h` with their leading ones.
fn panel<T: Scalar>(
    h: &mut [T],
    n: usize,
    p0: usize,
    w: &mut [T],
    d: &mut [T::Real],
    e: &mut [T::Real],
    taus: &mut [T],
) {
    let m = n - p0;
    let mut t = [T::zero(); 2 * NB];
    for i in 0..NB {
        let ci = p0 + i;
        {
            // Column `ci` from row `ci`, less what the panel's reflections before it change:
            // `a -= V·conj(W[ci, :]) + W·conj(V[ci, :])`.
            let (left, right) = h.split_at_mut(ci * n);
            let col = &mut right[ci..n];
            let mut cols: Vec<&[T]> = Vec::with_capacity(2 * i);
            let mut coef = Vec::with_capacity(2 * i);
            for j in 0..i {
                let v = &left[(p0 + j) * n + ci..(p0 + j + 1) * n];
                let wj = &w[j * m + i..(j + 1) * m];
                cols.extend([v, wj]);
                coef.extend([-wj[0].conj(), -v[0].conj()]);
            }
            add_columns(col, &cols, &coef);
            col[0] = T::from_real(col[0].re());
            d[ci] = col[0].re();
        }
        if ci + 1 == n {
            break;
        }
        let (tau, beta) = {
            let col = &mut h[ci * n + ci + 1..(ci + 1) * n];
            let (alpha, x) = col.split_first_mut().expect("a subdiagonal");
            let r = reflect(*alpha, x);
            *alpha = T::one();
            r
        };
        e[ci] = beta.re();
        taus[ci] = tau;
        // `W[:, i] = τ·(A·v - V·(W*·v) - W·(V*·v))`, then `-= (τ/2)·(W[:, i]*·v)·v`.
        let (wprev, wcur) = w.split_at_mut(i * m);
        let wi = &mut wcur[i + 1..m];
        let v = &h[ci * n + ci + 1..(ci + 1) * n];
        hemv(m - i - 1, &h[(ci + 1) * (n + 1)..], n, v, wi);
        if i > 0 {
            let mut cols: Vec<&[T]> = Vec::with_capacity(2 * i);
            for j in 0..i {
                cols.push(&wprev[j * m + i + 1..(j + 1) * m]);
                cols.push(&h[(p0 + j) * n + ci + 1..(p0 + j + 1) * n]);
            }
            // `t = [W V]*·v`, then `W[:, i] -= [V W]·t`.
            let t = &mut t[..2 * i];
            dot_columns(&cols, v, t);
            for pair in cols.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
            for x in t.iter_mut() {
                *x = -*x;
            }
            add_columns(wi, &cols, t);
        }
        for x in wi.iter_mut() {
            *x *= tau;
        }
        let alpha = -(tau * dotc(wi, v)).scale(T::Real::HALF);
        for (x, v) in wi.iter_mut().zip(v) {
            *x += alpha * *v;
        }
    }
}

/// Reduces the columns from `p0` on one at a time, as LAPACK's `hetd2`: each reflection is
/// applied to the trailing matrix by the rank-two update `A -= v·w* + w·v*`, `w` of length `n`
/// the room for its `w`.
fn unblocked<T: Scalar>(
    h: &mut [T],
    n: usize,
    p0: usize,
    d: &mut [T::Real],
    e: &mut [T::Real],
    taus: &mut [T],
    w: &mut [T],
) {
    for j in p0..n {
        let diag = h[j * (n + 1)].re();
        d[j] = diag;
        if j + 1 == n {
            break;
        }
        let (tau, beta) = {
            let col = &mut h[j * n + j + 1..(j + 1) * n];
            let (alpha, x) = col.split_first_mut().expect("a subdiagonal");
            let r = reflect(*alpha, x);
            *alpha = T::one();
            r
        };
        e[j] = beta.re();
        taus[j] = tau;
        if tau == T::zero() {
            continue;
        }
        let m = n - j - 1;
        let (left, right) = h.split_at_mut((j + 1) * n);
        let v = &left[j * n + j + 1..];
        let a22 = &mut right[j + 1..];
        let w = &mut w[..m];
        // `w = τ·A·v - (τ/2)·(τ·v*·A·v)·v`.
        hemv(m, a22, n, v, w);
        for x in w.iter_mut() {
            *x *= tau;
        }
        let alpha = -(tau * dotc(w, v)).scale(T::Real::HALF);
        for (x, v) in w.iter_mut().zip(v) {
            *x += alpha * *v;
        }
        for c in 0..m {
            let (vc, wc) = (-v[c].conj(), -w[c].conj());
            let col = &mut a22[c * n + c..c * n + m];
            for ((x, &v), &w) in col.iter_mut().zip(&v[c..]).zip(&w[c..]) {
                *x = fma(w, vc, fma(v, wc, *x));
            }
        }
    }
}

/// [`reflector`] with the norm of `x` from one pass of sums of squares in lanes when that sum is
/// in the range where no square can have overflowed or underflowed to matter, as is usual; from
/// the scaled norm otherwise.
fn reflect<T: Scalar>(alpha: T, x: &mut [T]) -> (T, T) {
    let mut s = [T::Real::ZERO; 4];
    let (quads, rest) = x.as_chunks::<4>();
    for c in quads {
        for l in 0..4 {
            s[l] += c[l].abs2();
        }
    }
    for v in rest {
        s[0] += v.abs2();
    }
    let ssq = (s[0] + s[2]) + (s[1] + s[3]);
    let safe = ssq >= T::Real::MIN_POSITIVE / T::Real::EPSILON && ssq <= T::Real::MAX;
    if !safe {
        return reflector(alpha, x);
    }
    // LAPACK's `larfg` from here, as `reflector`.
    let norm = alpha.abs().hypot(ssq.sqrt());
    let beta = if alpha.re() >= T::Real::ZERO {
        -norm
    } else {
        norm
    };
    let tau = T::from_parts((beta - alpha.re()) / beta, -alpha.im() / beta);
    let r = T::one() / (alpha - T::from_real(beta));
    for v in x.iter_mut() {
        *v *= r;
    }
    (tau, T::from_real(beta))
}

/// Whether the target fuses multiply-adds in hardware; elsewhere `mul_add` is a library call.
const FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `c + a·b`, rounded once where the target fuses multiply-adds.
#[inline(always)]
pub(super) fn fma<T: Scalar>(a: T, b: T, c: T) -> T {
    if !FMA {
        c + a * b
    } else if T::IS_COMPLEX {
        let (ar, ai, br, bi) = (a.re(), a.im(), b.re(), b.im());
        let re = (-ai).mul_add(bi, ar.mul_add(br, c.re()));
        let im = ai.mul_add(br, ar.mul_add(bi, c.im()));
        T::from_parts(re, im)
    } else {
        T::from_real(a.re().mul_add(b.re(), c.re()))
    }
}

/// `c + conj(a)·b`, rounded once where the target fuses multiply-adds.
#[inline(always)]
pub(super) fn fma_conj<T: Scalar>(a: T, b: T, c: T) -> T {
    if !FMA {
        c + a.conj() * b
    } else if T::IS_COMPLEX {
        let (ar, ai, br, bi) = (a.re(), a.im(), b.re(), b.im());
        let re = ai.mul_add(bi, ar.mul_add(br, c.re()));
        let im = (-ai).mul_add(br, ar.mul_add(bi, c.im()));
        T::from_parts(re, im)
    } else {
        T::from_real(a.re().mul_add(b.re(), c.re()))
    }
}

/// The sum of the lanes of an accumulator.
#[inline(always)]
pub(super) fn lanes<T: Scalar, const L: usize>(s: [T; L]) -> T {
    let mut t = T::zero();
    for x in s {
        t += x;
    }
    t
}

/// `x*·y`.
pub(super) fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut t = [T::zero()];
    dot_columns(&[x], y, &mut t);
    t[0]
}

/// `t_j = c_j*·v` for the columns `c_j`, each at least as long as `v`: four at a time, in lanes.
fn dot_columns<T: Scalar>(cols: &[&[T]], v: &[T], t: &mut [T]) {
    if T::IS_COMPLEX {
        dot_columns_by::<T, 2>(cols, v, t);
    } else {
        dot_columns_by::<T, 4>(cols, v, t);
    }
}

/// [`dot_columns`] with `L` lanes.
fn dot_columns_by<T: Scalar, const L: usize>(cols: &[&[T]], v: &[T], t: &mut [T]) {
    let len = v.len();
    let (vb, vt) = v.as_chunks::<L>();
    let body = len - vt.len();
    let (quads, rest) = cols.as_chunks::<4>();
    let (tq, tr) = t.as_chunks_mut::<4>();
    for (cols, t) in quads.iter().zip(tq) {
        let c = cols.map(|c| &c[..len]);
        let (a0, a1, a2, a3) = (
            lanes_of::<T, L>(c[0]),
            lanes_of::<T, L>(c[1]),
            lanes_of::<T, L>(c[2]),
            lanes_of::<T, L>(c[3]),
        );
        let mut s = [[T::zero(); L]; 4];
        let it = vb.iter().zip(a0).zip(a1).zip(a2.iter().zip(a3));
        for (((x, a0), a1), (a2, a3)) in it {
            for l in 0..L {
                s[0][l] = fma_conj(a0[l], x[l], s[0][l]);
                s[1][l] = fma_conj(a1[l], x[l], s[1][l]);
                s[2][l] = fma_conj(a2[l], x[l], s[2][l]);
                s[3][l] = fma_conj(a3[l], x[l], s[3][l]);
            }
        }
        for (k, (s, c)) in s.iter_mut().zip(c).enumerate() {
            for (&x, &a) in vt.iter().zip(&c[body..]) {
                s[0] = fma_conj(a, x, s[0]);
            }
            t[k] = lanes(*s);
        }
    }
    for (c, t) in rest.iter().zip(tr) {
        let c = &c[..len];
        let mut s = [T::zero(); L];
        for (x, a) in vb.iter().zip(lanes_of::<T, L>(c)) {
            for l in 0..L {
                s[l] = fma_conj(a[l], x[l], s[l]);
            }
        }
        for (&x, &a) in vt.iter().zip(&c[body..]) {
            s[0] = fma_conj(a, x, s[0]);
        }
        *t = lanes(s);
    }
}

/// The whole chunks of `L` of `x`.
#[inline(always)]
pub(super) fn lanes_of<T, const L: usize>(x: &[T]) -> &[[T; L]] {
    x.as_chunks::<L>().0
}

/// `y += Σ_j c_j·a_j` for the columns `a_j`, each at least as long as `y`: four at a time.
fn add_columns<T: Scalar>(y: &mut [T], cols: &[&[T]], c: &[T]) {
    let len = y.len();
    let (quads, rest) = cols.as_chunks::<4>();
    let (cq, cr) = c.as_chunks::<4>();
    for (cols, c) in quads.iter().zip(cq) {
        let [a0, a1, a2, a3] = cols.map(|c| &c[..len]);
        let it = y.iter_mut().zip(a0).zip(a1).zip(a2.iter().zip(a3));
        for (((y, &a0), &a1), (&a2, &a3)) in it {
            *y = fma(a3, c[3], fma(a2, c[2], fma(a1, c[1], fma(a0, c[0], *y))));
        }
    }
    for (a, &c) in rest.iter().zip(cr) {
        for (y, &a) in y.iter_mut().zip(&a[..len]) {
            *y = fma(a, c, *y);
        }
    }
}

/// `y = A·x` for the Hermitian `m × m` block `a` held in its lower triangle, the imaginary parts
/// of its diagonal taken as zeros: each element below the diagonal read once, for both the
/// column and the row it is in, four columns at a time.
fn hemv<T: Scalar>(m: usize, a: &[T], lda: usize, x: &[T], y: &mut [T]) {
    if T::IS_COMPLEX {
        hemv_by::<T, 2>(m, a, lda, x, y);
    } else {
        hemv_by::<T, 4>(m, a, lda, x, y);
    }
}

/// [`hemv`] with `L` rows a step, the rows' sums for the columns in `L` lanes.
fn hemv_by<T: Scalar, const L: usize>(m: usize, a: &[T], lda: usize, x: &[T], y: &mut [T]) {
    let y = &mut y[..m];
    y.fill(T::zero());
    let mut j = 0;
    while j + 4 <= m {
        // The 4 × 4 block on the diagonal.
        let mut t = [T::zero(); 4];
        for k in 0..4 {
            let col = &a[(j + k) * lda..];
            t[k] += x[j + k].scale(col[j + k].re());
            for l in k + 1..4 {
                let v = col[j + l];
                y[j + l] = fma(v, x[j + k], y[j + l]);
                t[k] = fma_conj(v, x[j + l], t[k]);
            }
        }
        // Below it.
        let b = j + 4;
        let col = |k: usize| &a[(j + k) * lda + b..(j + k) * lda + m];
        let (c0, c1, c2, c3) = (col(0), col(1), col(2), col(3));
        let (x0, x1, x2, x3) = (x[j], x[j + 1], x[j + 2], x[j + 3]);
        let (ys, xr) = (&mut y[b..m], &x[b..m]);
        let (head, tail) = ys.as_chunks_mut::<L>();
        let body = head.len() * L;
        let mut s = [[T::zero(); L]; 4];
        let it = head
            .iter_mut()
            .zip(lanes_of::<T, L>(xr))
            .zip(lanes_of::<T, L>(c0));
        let it = it
            .zip(lanes_of::<T, L>(c1))
            .zip(lanes_of::<T, L>(c2).iter().zip(lanes_of::<T, L>(c3)));
        for ((((yv, xv), a0), a1), (a2, a3)) in it {
            for l in 0..L {
                let y = fma(a1[l], x1, fma(a0[l], x0, yv[l]));
                yv[l] = fma(a3[l], x3, fma(a2[l], x2, y));
                s[0][l] = fma_conj(a0[l], xv[l], s[0][l]);
                s[1][l] = fma_conj(a1[l], xv[l], s[1][l]);
                s[2][l] = fma_conj(a2[l], xv[l], s[2][l]);
                s[3][l] = fma_conj(a3[l], xv[l], s[3][l]);
            }
        }
        for (i, y) in tail.iter_mut().enumerate() {
            let r = body + i;
            *y = fma(
                c3[r],
                x3,
                fma(c2[r], x2, fma(c1[r], x1, fma(c0[r], x0, *y))),
            );
            s[0][0] = fma_conj(c0[r], xr[r], s[0][0]);
            s[1][0] = fma_conj(c1[r], xr[r], s[1][0]);
            s[2][0] = fma_conj(c2[r], xr[r], s[2][0]);
            s[3][0] = fma_conj(c3[r], xr[r], s[3][0]);
        }
        for k in 0..4 {
            y[j + k] += t[k] + lanes(s[k]);
        }
        j += 4;
    }
    while j < m {
        let col = &a[j * lda..j * lda + m];
        let xj = x[j];
        let mut t = xj.scale(col[j].re());
        for r in j + 1..m {
            y[r] = fma(col[r], xj, y[r]);
            t = fma_conj(col[r], x[r], t);
        }
        y[j] += t;
        j += 1;
    }
}

/// `V = Q·V` for the `n × n` eigenvectors `v` of the tridiagonal matrix, `Q` from the reflections
/// [`tridiagonalize`] left in `h`, as LAPACK's `ormtr`: blocks of [`BACK`] of them, from the
/// last, each `I - Y·T·Y*` applied by three products of blocks, `Y*` copied out so that every
/// product takes its blocks as they are.
fn back_transform<T: Scalar>(h: &[T], n: usize, taus: &[T], v: &mut [T]) {
    if n < 2 {
        return;
    }
    let (one, zero) = (T::one(), T::zero());
    let mut yh = Vec::new();
    let mut w = Vec::new();
    let mut tw = Vec::new();
    // The reflections act on the rows from 1 on, held as a QR factorization's are.
    let mut end = n - 1;
    while end > 0 {
        let start = (end - 1) / BACK * BACK;
        let block = Block::new(
            &h[1 + start * (n + 1)..],
            n,
            n - 1 - start,
            &taus[start..end],
        );
        let (m, k) = (block.m, block.k);
        yh.clear();
        yh.resize(k * m, zero);
        for (j, col) in block.v.chunks_exact(m).enumerate() {
            for (i, &y) in col.iter().enumerate() {
                yh[j + i * k] = y.conj();
            }
        }
        w.resize(k * n, zero);
        tw.resize(k * n, zero);
        let c = &mut v[1 + start..];
        gemm(k, n, m, one, &yh, k, Op::N, c, n, Op::N, zero, &mut w, k);
        gemm(
            k,
            n,
            k,
            one,
            &block.t,
            k,
            Op::N,
            &w,
            k,
            Op::N,
            zero,
            &mut tw,
            k,
        );
        gemm(m, n, k, -one, &block.v, m, Op::N, &tw, k, Op::N, one, c, n);
        end = start;
    }
}
