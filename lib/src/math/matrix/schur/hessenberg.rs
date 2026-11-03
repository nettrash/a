//! The reduction to upper Hessenberg form by Householder reflections, blocked as LAPACK's
//! `gehrd`: a panel of reflections at a time, built with `A·V·T` as LAPACK's `lahr2`, so that
//! most of the work on the rest of the matrix is products of blocks.

use super::super::blas::{Op, gemm};
use super::super::householder::{form_q, reflect_left, reflect_right, reflector};
use super::super::{Matrix, Scalar};

/// The columns of a panel.
const PANEL: usize = 32;

/// The last columns, reduced one at a time: there the panels' products are too small to pay.
const UNBLOCKED: usize = 32;

/// Reduces `a` to upper Hessenberg form `H = Q*·A·Q` in place, and gives `Q` when `want_q` is
/// set.
pub(crate) fn hessenberg<T: Scalar>(a: &mut Matrix<T>, want_q: bool) -> Option<Matrix<T>> {
    let n = a.rows();
    let mut taus = vec![T::zero(); n.saturating_sub(2)];
    reduce(&mut a.data, n, n, &mut taus);
    let q = want_q.then(|| {
        let mut q = Matrix::identity(n);
        if n > 2 {
            let sub = form_q(&a.data[1..], n, n - 1, &taus, n - 1);
            for j in 0..n - 1 {
                q.data[(j + 1) * n + 1..(j + 2) * n].copy_from_slice(sub.column(j));
            }
        }
        q
    });
    for j in 0..n.saturating_sub(2) {
        a.data[j * n + j + 2..(j + 1) * n].fill(T::zero());
    }
    q
}

/// Reduces the leading `ihi × ihi` block of the `n × n` matrix `a` to upper Hessenberg form, the
/// rows below it taken to be zero in its columns: `A = Q·H·Q*` with `Q = H₁·H₂·…·H_{ihi-2}`,
/// each reflection's factor in `taus` and its vector below the subdiagonal of its column, the
/// columns after `ihi` multiplied by `Q*` from the left.
pub(crate) fn reduce<T: Scalar>(a: &mut [T], n: usize, ihi: usize, taus: &mut [T]) {
    let mut p = 0;
    let mut t = vec![T::zero(); PANEL * PANEL];
    let mut y = Vec::new();
    while p + UNBLOCKED + 2 <= ihi {
        let ib = PANEL.min(ihi - p - 1);
        y.resize(ihi * ib, T::zero());
        panel(a, n, ihi, p, ib, &mut taus[p..p + ib], &mut t, &mut y);
        // `A·(I - V·T·V*) = A - Y·V*`: the columns after the panel in one product, the vectors'
        // last unit element put in for it.
        let e = a[p + ib + (p + ib - 1) * n];
        a[p + ib + (p + ib - 1) * n] = T::one();
        let (v, rest) = a.split_at_mut((p + ib) * n);
        let v = &v[p + ib + p * n..];
        gemm(
            ihi,
            ihi - p - ib,
            ib,
            -T::one(),
            &y,
            ihi,
            Op::N,
            v,
            n,
            Op::C,
            T::one(),
            rest,
            n,
        );
        a[p + ib + (p + ib - 1) * n] = e;
        // The panel's own columns above the reflections' rows: `Y·V₁*`, `V₁` unit lower
        // triangular.
        for j in 0..ib - 1 {
            for l in 0..=j {
                let f = if l == j {
                    T::one()
                } else {
                    a[p + 1 + j + (p + l) * n].conj()
                };
                let column = &mut a[(p + 1 + j) * n..(p + 1 + j) * n + p + 1];
                for (x, y) in column.iter_mut().zip(&y[l * ihi..l * ihi + p + 1]) {
                    *x -= *y * f;
                }
            }
        }
        // `(I - V·T·V*)*` from the left on the rows of the reflections, the columns after.
        let at = p + 1 + p * n;
        let (v, c) = a.split_at_mut((p + ib) * n);
        block_left(
            &v[at..],
            n,
            ihi - p - 1,
            ib,
            &t,
            PANEL,
            &mut c[p + 1..],
            n,
            n - p - ib,
        );
        p += ib;
    }
    // The rest one reflection at a time, as LAPACK's `gehd2`.
    for j in p..ihi.saturating_sub(2) {
        let (tau, beta) = {
            let alpha = a[j + 1 + j * n];
            reflector(alpha, &mut a[j * n + j + 2..j * n + ihi])
        };
        a[j + 1 + j * n] = T::one();
        taus[j] = tau;
        let (left, right) = a.split_at_mut((j + 1) * n);
        let v = &left[j * n + j + 1..j * n + ihi];
        // `A·H` on the rows of the block, then `H*·A` on the rows of the reflection.
        reflect_right(tau, v, ihi, ihi - j - 1, right, n);
        reflect_left(tau, true, v, ihi - j - 1, n - j - 1, &mut right[j + 1..], n);
        a[j + 1 + j * n] = beta;
    }
}

/// `C = (I - V·T·V*)*·C = C - V·T*·V*·C` for the `m × cols` block `c`: `V` of `m × k` unit
/// lower trapezoidal, its vectors below the diagonal of `v`, and `T` of `k × k` upper
/// triangular. The conjugate transposes are made explicitly, so that each product takes its
/// blocks as they are.
#[allow(clippy::too_many_arguments)]
pub(super) fn block_left<T: Scalar>(
    v: &[T],
    ldv: usize,
    m: usize,
    k: usize,
    t: &[T],
    ldt: usize,
    c: &mut [T],
    ldc: usize,
    cols: usize,
) {
    if cols == 0 || k == 0 {
        return;
    }
    let mut vh = vec![T::zero(); k * m];
    let mut full = vec![T::zero(); m * k];
    for j in 0..k {
        vh[j + j * k] = T::one();
        full[j + j * m] = T::one();
        for i in j + 1..m {
            let x = v[i + j * ldv];
            vh[j + i * k] = x.conj();
            full[i + j * m] = x;
        }
    }
    let mut th = vec![T::zero(); k * k];
    for j in 0..k {
        for l in 0..=j {
            th[j + l * k] = t[l + j * ldt].conj();
        }
    }
    let mut w = vec![T::zero(); k * cols];
    gemm(
        k,
        cols,
        m,
        T::one(),
        &vh,
        k,
        Op::N,
        c,
        ldc,
        Op::N,
        T::zero(),
        &mut w,
        k,
    );
    let mut tw = vec![T::zero(); k * cols];
    gemm(
        k,
        cols,
        k,
        T::one(),
        &th,
        k,
        Op::N,
        &w,
        k,
        Op::N,
        T::zero(),
        &mut tw,
        k,
    );
    gemm(
        m,
        cols,
        k,
        -T::one(),
        &full,
        m,
        Op::N,
        &tw,
        k,
        Op::N,
        T::one(),
        c,
        ldc,
    );
}

/// Reduces the `nb` columns of `a` from column `p`, rows below `ihi` zero, as LAPACK's `lahr2`:
/// the reflections `I - V·T·V*` that zero each column below its subdiagonal, applied to the
/// panel only, with `T` (`nb × nb`, upper triangular, columns `PANEL` apart) and
/// `Y = A·V·T` (`ihi × nb`) for the rest. The vectors' last unit element is left as `β`.
#[allow(clippy::too_many_arguments)]
fn panel<T: Scalar>(
    a: &mut [T],
    n: usize,
    ihi: usize,
    p: usize,
    nb: usize,
    taus: &mut [T],
    t: &mut [T],
    y: &mut [T],
) {
    // The reflections' rows start at `k`; `Y` and `T` are `ihi` and `PANEL` apart.
    let k = p + 1;
    let ld = ihi;
    let at = |i: usize, j: usize| i + j * n;
    let mut e = T::zero();
    let mut w = vec![T::zero(); nb];
    for i in 0..nb {
        let col = p + i;
        if i > 0 {
            // The column of `A - Y·V*`, rows `k..`.
            let w = &mut w[..i];
            for (j, w) in w.iter_mut().enumerate() {
                *w = -a[at(k + i - 1, p + j)].conj();
            }
            gemv(&mut a[at(k, col)..at(ihi, col)], &y[k..], ld, w);
            // Then `(I - V·T·V*)*` from the left: `w = T*·V*·b`, `b -= V·w`, with `V₁` the
            // unit lower triangle of rows `k..k + i` and `V₂` the rows below.
            w.copy_from_slice(&a[at(k, col)..at(k + i, col)]);
            for j in 0..i {
                let mut s = w[j];
                for r in j + 1..i {
                    s += a[at(k + r, p + j)].conj() * w[r];
                }
                w[j] = s;
            }
            gemv_c(w, &a[at(k + i, p)..], n, &a[at(k + i, col)..at(ihi, col)]);
            for j in (0..i).rev() {
                let mut s = T::zero();
                for l in 0..=j {
                    s += t[l + j * PANEL].conj() * w[l];
                }
                w[j] = -s;
            }
            let (left, right) = a.split_at_mut(at(0, col));
            gemv(&mut right[k + i..ihi], &left[at(k + i, p)..], n, w);
            for r in (0..i).rev() {
                let mut s = w[r];
                for j in 0..r {
                    s += left[at(k + r, p + j)] * w[j];
                }
                w[r] = s;
            }
            for (b, w) in right[k..k + i].iter_mut().zip(w.iter()) {
                *b += *w;
            }
            a[at(k + i - 1, p + i - 1)] = e;
        }
        let (tau, beta) = {
            let alpha = a[at(k + i, col)];
            reflector(alpha, &mut a[at(k + i + 1, col)..at(ihi, col)])
        };
        taus[i] = tau;
        e = beta;
        a[at(k + i, col)] = T::one();
        // `Y[k.., i] = A[k.., col + 1..]·v`, the vector's element `c` paired with column `c`.
        let (left, right) = a.split_at_mut(at(0, col + 1));
        let v = &left[at(k + i, col)..at(ihi, col)];
        let yi = &mut y[k + i * ld..ihi + i * ld];
        yi.fill(T::zero());
        gemv(yi, &right[k..], n, v);
        // `T[0..i, i] = V₂*·v`, then `Y[k.., i] = τ·(Y[k.., i] - Y[k.., 0..i]·T[0..i, i])`.
        let ti = &mut t[i * PANEL..i * PANEL + i];
        ti.fill(T::zero());
        gemv_c(ti, &left[at(k + i, p)..], n, v);
        let w = &mut w[..i];
        for (w, t) in w.iter_mut().zip(ti.iter()) {
            *w = -*t;
        }
        let (done, yi) = y.split_at_mut(i * ld);
        let yi = &mut yi[k..ihi];
        if i > 0 {
            gemv(yi, &done[k..], ld, w);
        }
        for y in yi.iter_mut() {
            *y *= tau;
        }
        // `T[0..i, i] = -τ·T[0..i, 0..i]·T[0..i, i]`.
        for j in 0..i {
            let mut s = T::zero();
            for l in j..i {
                s += t[j + l * PANEL] * t[l + i * PANEL];
            }
            t[j + i * PANEL] = -(tau * s);
        }
        t[i + i * PANEL] = tau;
    }
    a[at(k + nb - 1, p + nb - 1)] = e;
    // `Y[0..k, ..] = A[0..k, k..]·V·T`: `V₁` from the panel's rows `k..k + nb`, `V₂` below.
    for j in 0..nb {
        y[j * ld..j * ld + k].copy_from_slice(&a[at(0, k + j)..at(k, k + j)]);
    }
    for j in 0..nb {
        for r in j + 1..nb {
            let f = a[at(k + r, p + j)];
            let (left, right) = y.split_at_mut(r * ld);
            for (x, y) in left[j * ld..j * ld + k].iter_mut().zip(&right[..k]) {
                *x += *y * f;
            }
        }
    }
    if ihi > k + nb {
        gemm(
            k,
            nb,
            ihi - k - nb,
            T::one(),
            &a[at(0, k + nb)..],
            n,
            Op::N,
            &a[at(k + nb, p)..],
            n,
            Op::N,
            T::one(),
            y,
            ld,
        );
    }
    for j in (0..nb).rev() {
        let f = t[j + j * PANEL];
        for x in &mut y[j * ld..j * ld + k] {
            *x *= f;
        }
        for l in 0..j {
            let f = t[l + j * PANEL];
            let (left, right) = y.split_at_mut(j * ld);
            for (x, y) in right[..k].iter_mut().zip(&left[l * ld..l * ld + k]) {
                *x += *y * f;
            }
        }
    }
}

/// `y += A·x`, for `A` of `y.len() × x.len()` with the leading dimension `lda`, four columns at
/// a time so that each pass over `y` does four multiply-adds.
pub(super) fn gemv<T: Scalar>(y: &mut [T], a: &[T], lda: usize, x: &[T]) {
    let m = y.len();
    let mut c = 0;
    while c + 4 <= x.len() {
        let (x0, x1, x2, x3) = (x[c], x[c + 1], x[c + 2], x[c + 3]);
        let a0 = &a[c * lda..c * lda + m];
        let a1 = &a[(c + 1) * lda..(c + 1) * lda + m];
        let a2 = &a[(c + 2) * lda..(c + 2) * lda + m];
        let a3 = &a[(c + 3) * lda..(c + 3) * lda + m];
        for i in 0..m {
            y[i] += a0[i] * x0 + a1[i] * x1 + a2[i] * x2 + a3[i] * x3;
        }
        c += 4;
    }
    while c < x.len() {
        let x0 = x[c];
        for (y, a) in y.iter_mut().zip(&a[c * lda..c * lda + m]) {
            *y += *a * x0;
        }
        c += 1;
    }
}

/// `w += A*·x`, for `A` of `x.len() × w.len()` with the leading dimension `lda`: dot products
/// four columns at a time, each in four partial sums so that they don't wait on each other.
pub(super) fn gemv_c<T: Scalar>(w: &mut [T], a: &[T], lda: usize, x: &[T]) {
    let m = x.len();
    let body = m - m % 4;
    let mut j = 0;
    while j + 4 <= w.len() {
        let cols: [&[T]; 4] = core::array::from_fn(|c| &a[(j + c) * lda..(j + c) * lda + m]);
        let mut s = [[T::zero(); 4]; 4];
        for r in (0..body).step_by(4) {
            for (s, col) in s.iter_mut().zip(&cols) {
                for l in 0..4 {
                    s[l] += col[r + l].conj() * x[r + l];
                }
            }
        }
        for (c, (s, col)) in s.iter().zip(&cols).enumerate() {
            let mut t = (s[0] + s[1]) + (s[2] + s[3]);
            for r in body..m {
                t += col[r].conj() * x[r];
            }
            w[j + c] += t;
        }
        j += 4;
    }
    while j < w.len() {
        let col = &a[j * lda..j * lda + m];
        let mut s = [T::zero(); 4];
        for r in (0..body).step_by(4) {
            for l in 0..4 {
                s[l] += col[r + l].conj() * x[r + l];
            }
        }
        let mut t = (s[0] + s[1]) + (s[2] + s[3]);
        for r in body..m {
            t += col[r].conj() * x[r];
        }
        w[j] += t;
        j += 1;
    }
}
