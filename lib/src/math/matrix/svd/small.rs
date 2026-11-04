//! The singular value decomposition of small matrices, where the blocked path's fixed costs
//! outweigh its arithmetic: the reduction to bidiagonal one column and row at a time, `U` and `V`
//! formed explicitly from its reflections, `U` in their place, and the implicit QR algorithm's
//! rotations applied straight to them; the singular values alone by dqds. No allocation but the
//! results': the scratch is on the stack.

use super::super::hermitian::small::{make_column, reflect_columns, reflect_columns2};
use super::super::hermitian::{dotc, fma, fma_conj, lanes, lanes_of};
use super::super::{Matrix, Scalar};
use super::bidiagonal::reflector;
use super::{dqds, qr};
use crate::math::complex::Float;

/// The most columns any type has solved here.
const LARGEST: usize = 192;

/// The numbers of columns whose scratch is the smallest of the sizes of it, then the middle one,
/// so that the smaller problems don't clear room for the largest.
const TIERS: [usize; 2] = [16, 48];

/// The most columns of the elements `T` solved here, as measured on an M4: past them, divide and
/// conquer and the reflections applied in blocks take less time. The QR algorithm's rotations
/// of real single precision vectors go in the most lanes, which moves their crossover furthest;
/// without the vectors, the reduction here keeps ahead of the blocked one longest for complex
/// elements, and least long for real single precision ones.
pub(super) fn largest<T: Scalar>(want_uv: bool) -> usize {
    match (want_uv, T::IS_COMPLEX, size_of::<T::Real>()) {
        (true, false, 4) => 112,
        (true, true, 4) => 88,
        (true, _, _) => 96,
        (false, false, 4) => 96,
        (false, false, _) => 112,
        (false, true, _) => LARGEST,
    }
}

/// [`decompose`](super::decompose) of the `m × n` column-major `a`, `m ≥ n`, `n` up to
/// [`largest`] and `m` below `2.5n` (QR goes first from there), its elements finite and of a
/// moderate size.
#[allow(clippy::type_complexity)]
pub(super) fn decompose<T: Scalar>(
    a: Vec<T>,
    m: usize,
    n: usize,
    want_uv: bool,
) -> (Vec<T::Real>, Option<(Matrix<T>, Matrix<T>)>) {
    // Room for up to `2.5` times as many rows as columns.
    if n <= TIERS[0] {
        solve::<T, { TIERS[0] }, { TIERS[0] * 5 / 2 }>(a, m, n, want_uv)
    } else if n <= TIERS[1] {
        solve::<T, { TIERS[1] }, { TIERS[1] * 5 / 2 }>(a, m, n, want_uv)
    } else {
        solve::<T, LARGEST, { LARGEST * 5 / 2 }>(a, m, n, want_uv)
    }
}

/// [`decompose`] with room for `N` columns and `M` rows.
#[allow(clippy::type_complexity)]
fn solve<T: Scalar, const N: usize, const M: usize>(
    mut a: Vec<T>,
    m: usize,
    n: usize,
    want_uv: bool,
) -> (Vec<T::Real>, Option<(Matrix<T>, Matrix<T>)>) {
    debug_assert!(m >= n && n <= N && (n == 0 || m < M));
    let zero = T::Real::ZERO;
    let mut d = vec![zero; n];
    let (mut e, mut tauq, mut taup) = ([zero; N], [T::zero(); N], [T::zero(); N]);
    bidiagonalize::<T, N, M>(&mut a, m, n, &mut d, &mut e, &mut tauq, &mut taup);
    let k = n.saturating_sub(1);
    if !want_uv {
        if !dqds::singular_values(&mut d, &e[..k]) {
            qr::bidiagonal_qr::<_, qr::Columns<_>, qr::Columns<_>>(&mut d, &mut e[..n], None, None);
        }
        return (d, None);
    }
    let mut v = form_v::<T, N>(&a, m, n, &taup[..k]);
    form_u(&mut a, m, n, &tauq[..n]);
    let mut cu = Cols {
        data: &mut a,
        rows: m,
        ld: m,
    };
    let mut cv = Cols {
        data: &mut v,
        rows: n,
        ld: n,
    };
    qr::bidiagonal_qr(&mut d, &mut e[..n], Some(&mut cu), Some(&mut cv));
    (
        d,
        Some((
            Matrix::from_column_major(m, n, a),
            Matrix::from_column_major(n, n, v),
        )),
    )
}

/// Reduces the `m × n` column-major `a`, `m ≥ n`, to the real upper bidiagonal `B = Q*·A·P`,
/// one column and row at a time, the reflections of LAPACK's `gebd2`: `Q = H₀·H₁·…·H_{n-1}`, the
/// vector of `H_j`
/// below the diagonal of column `j` and its factor in `tauq`; `P = G₀·G₁·…·G_{n-2}`, `G_j`
/// acting on the elements from `j + 1`, the rest of its vector after the leading one in row `j`
/// from column `j + 2` and its factor in `taup`. `G_j*` turns the conjugate of the row into
/// `(β, 0, …)`, so that the row times `G_j` is `(β, 0, …)`.
///
/// Each right reflection is applied with the next left one, in one pass over the rest of the
/// matrix that also forms the product for the right reflection after: the rest is read twice a
/// step and written once, not four times and twice. With `x_k` the columns as held, `t` and
/// `f_k = τ_p·conj(w_k)` from the right reflection not yet applied, and `v` the left one's
/// vector, the products `v*·(x_k - f_k·t) = v*·x_k - f_k·(v*·t)` come from the columns as held,
/// and the row the next right reflection is made of from them alone.
fn bidiagonalize<T: Scalar, const N: usize, const M: usize>(
    a: &mut [T],
    m: usize,
    n: usize,
    d: &mut [T::Real],
    e: &mut [T::Real],
    tauq: &mut [T],
    taup: &mut [T],
) {
    // The right reflection not yet applied: its factor and vector, `w[0]` for the column `i`,
    // and `t`, the rows of `C·w` from `i`.
    let (mut tp, mut w) = (T::zero(), [T::zero(); N]);
    let mut room = ([T::zero(); M], [T::zero(); M]);
    let (mut t, mut next) = (&mut room.0, &mut room.1);
    let (mut f, mut g) = ([T::zero(); N], [T::zero(); N]);
    for i in 0..n {
        let (left, right) = a.split_at_mut((i + 1) * m);
        let col = &mut left[i * m..];
        if tp != T::zero() {
            for (x, &t) in col[i..].iter_mut().zip(&t[i..m]) {
                *x = fma(t, -tp, *x);
            }
        }
        let (tq, beta) = {
            let (head, tail) = col[i..].split_at_mut(1);
            reflector(head[0], tail)
        };
        d[i] = beta.re();
        tauq[i] = tq;
        if i + 1 == n {
            break;
        }
        let k = n - i - 1;
        let v = &col[i + 1..];
        let (f, g) = (&mut f[..k], &mut g[..k]);
        for (f, &w) in f.iter_mut().zip(&w[1..=k]) {
            *f = tp * w.conj();
        }
        // `g_j = conj(τ_q)·v*·(x_j - f_j·t)`, the row `i` of the columns after both
        // reflections, and from it the next right reflection.
        let vt = if tp != T::zero() {
            t[i] + dotc(v, &t[i + 1..m])
        } else {
            T::zero()
        };
        products(v, &right[i..], m, k, g);
        let ct = tq.conj();
        let z = &mut w[..k];
        for (j, ((z, g), &f)) in z.iter_mut().zip(g.iter_mut()).zip(f.iter()).enumerate() {
            let x = right[i + j * m];
            *g = ct * (*g - f * vt);
            *z = (x - f * t[i] - *g).conj();
        }
        let (tp1, beta) = {
            let (head, tail) = z.split_at_mut(1);
            reflector(head[0], tail)
        };
        e[i] = beta.re();
        taup[i] = tp1;
        z[0] = T::one();
        for (j, &w) in z.iter().enumerate().skip(1) {
            right[i + j * m] = w;
        }
        // The rows after `i`: `x_j -= f_j·t + g_j·v`, and `C·w` of the result.
        let rows = i + 1..m;
        update(
            v,
            (&t[rows.clone()], &f[..]),
            &g[..],
            &mut right[i + 1..],
            m,
            (&*z, &mut next[rows.clone()]),
            tp1 != T::zero(),
        );
        tp = tp1;
        core::mem::swap(&mut t, &mut next);
    }
}

/// `p_j = (1, v)*·x_j` for the `k` columns `x_j` of `c` (leading dimension `ld`), each of
/// `v.len() + 1` elements: four columns a pass, in lanes of two vector registers each, so that
/// the products' additions don't wait on one another.
fn products<T: Scalar>(v: &[T], c: &[T], ld: usize, k: usize, p: &mut [T]) {
    match size_of::<T>() {
        4 => products_by::<T, 8>(v, c, ld, k, p),
        8 => products_by::<T, 4>(v, c, ld, k, p),
        _ => products_by::<T, 2>(v, c, ld, k, p),
    }
}

/// [`products`] with `L` lanes.
fn products_by<T: Scalar, const L: usize>(v: &[T], c: &[T], ld: usize, k: usize, p: &mut [T]) {
    let len = v.len() + 1;
    let col = |j: usize| &c[j * ld..j * ld + len];
    let (vb, vt) = v.as_chunks::<L>();
    let body = len - 1 - vt.len();
    let mut j = 0;
    while j + 4 <= k {
        let x = [col(j), col(j + 1), col(j + 2), col(j + 3)];
        let mut s = [[T::zero(); L]; 4];
        let [x0, x1, x2, x3] = x.map(|x| lanes_of::<T, L>(&x[1..]));
        for ((v, (y0, y1)), (y2, y3)) in vb.iter().zip(x0.iter().zip(x1)).zip(x2.iter().zip(x3)) {
            for l in 0..L {
                s[0][l] = fma_conj(v[l], y0[l], s[0][l]);
                s[1][l] = fma_conj(v[l], y1[l], s[1][l]);
                s[2][l] = fma_conj(v[l], y2[l], s[2][l]);
                s[3][l] = fma_conj(v[l], y3[l], s[3][l]);
            }
        }
        for q in 0..4 {
            let mut t = x[q][0] + lanes(s[q]);
            for (&v, &y) in vt.iter().zip(&x[q][1 + body..]) {
                t = fma_conj(v, y, t);
            }
            p[j + q] = t;
        }
        j += 4;
    }
    while j < k {
        let x = col(j);
        p[j] = x[0] + dotc(v, &x[1..]);
        j += 1;
    }
}

/// `x_j -= f_j·t + g_j·v` for the `k = g.len()` columns `x_j` of `c` (leading dimension `ld`)
/// of `v.len()` elements, and with `accumulate`, `y = Σ w_j·x_j` of the results: four columns a
/// pass, each element of `t`, `v` and `y` loaded once for the four.
fn update<T: Scalar>(
    v: &[T],
    (t, f): (&[T], &[T]),
    g: &[T],
    c: &mut [T],
    ld: usize,
    (w, y): (&[T], &mut [T]),
    accumulate: bool,
) {
    // Separate loops with and without the sums, which the compiler vectorizes only apart.
    if accumulate {
        update_by::<T, true>(v, (t, f), g, c, ld, (w, y));
    } else {
        update_by::<T, false>(v, (t, f), g, c, ld, (w, y));
    }
}

/// [`update`], the sums with `ACCUMULATE`.
fn update_by<T: Scalar, const ACCUMULATE: bool>(
    v: &[T],
    (t, f): (&[T], &[T]),
    g: &[T],
    c: &mut [T],
    ld: usize,
    (w, y): (&[T], &mut [T]),
) {
    let len = v.len();
    let k = g.len();
    let (t, y) = (&t[..len], &mut y[..len]);
    y.fill(T::zero());
    let mut j = 0;
    while j + 4 <= k {
        let block = &mut c[j * ld..];
        let (c0, block) = block.split_at_mut(ld);
        let (c1, block) = block.split_at_mut(ld);
        let (c2, c3) = block.split_at_mut(ld);
        let m = |a: &[T]| [-a[j], -a[j + 1], -a[j + 2], -a[j + 3]];
        let wj = [w[j], w[j + 1], w[j + 2], w[j + 3]];
        let (x0, x1, x2, x3) = (
            &mut c0[..len],
            &mut c1[..len],
            &mut c2[..len],
            &mut c3[..len],
        );
        columns4::<T, ACCUMULATE>(x0, x1, x2, x3, (v, t, &mut *y), (m(f), m(g), wj));
        j += 4;
    }
    while j < k {
        let x = &mut c[j * ld..j * ld + len];
        let (mf, mg, wj) = (-f[j], -g[j], w[j]);
        for ((x, (&v, &t)), y) in x.iter_mut().zip(v.iter().zip(t)).zip(y.iter_mut()) {
            let u = fma(v, mg, fma(t, mf, *x));
            *x = u;
            if ACCUMULATE {
                *y = fma(u, wj, *y);
            }
        }
        j += 1;
    }
}

/// The four columns of a pass of [`update_by`], each its own argument, so that the compiler
/// knows that they don't overlap and vectorizes the loop without checking at run time.
#[inline(always)]
fn columns4<T: Scalar, const ACCUMULATE: bool>(
    x0: &mut [T],
    x1: &mut [T],
    x2: &mut [T],
    x3: &mut [T],
    (v, t, y): (&[T], &[T], &mut [T]),
    (mf, mg, w): ([T; 4], [T; 4], [T; 4]),
) {
    let it = x0
        .iter_mut()
        .zip(x1.iter_mut())
        .zip(x2.iter_mut().zip(x3.iter_mut()))
        .zip(v.iter().zip(t).zip(y.iter_mut()));
    for (((x0, x1), (x2, x3)), ((&v, &t), y)) in it {
        let u0 = fma(v, mg[0], fma(t, mf[0], *x0));
        let u1 = fma(v, mg[1], fma(t, mf[1], *x1));
        let u2 = fma(v, mg[2], fma(t, mf[2], *x2));
        let u3 = fma(v, mg[3], fma(t, mf[3], *x3));
        (*x0, *x1, *x2, *x3) = (u0, u1, u2, u3);
        if ACCUMULATE {
            *y = fma(u3, w[3], fma(u2, w[2], fma(u1, w[1], fma(u0, w[0], *y))));
        }
    }
}

/// `V = P = G₀·G₁·…·G_{n-2}`, `n × n`, from the right reflections [`bidiagonalize`] left in the
/// rows of `a`, as LAPACK's `ungbr`: from the last, two at a time, each pair applied to the
/// columns it changes, then turned into its own two columns.
fn form_v<T: Scalar, const N: usize>(a: &[T], m: usize, n: usize, taus: &[T]) -> Vec<T> {
    let mut v = vec![T::zero(); n * n];
    if n == 0 {
        return v;
    }
    // `G_r` acts on the rows from `r + 1`, its vector after the leading one in row `r` of `a`
    // from column `r + 2`, gathered into `w`, and makes column `r + 1` of `V`.
    let gather = |w: &mut [T], r: usize| {
        for (k, w) in w.iter_mut().enumerate() {
            *w = a[r + (r + 2 + k) * m];
        }
    };
    let (mut wa, mut wb) = ([T::zero(); N], [T::zero(); N]);
    let mut r = n - 1;
    while r >= 2 {
        // `G_{r-2}·G_{r-1}`, on the rows from `r - 1`.
        let (b, c) = (r - 1, r - 2);
        let (va, vb) = (&mut wa[..n - r], &mut wb[..n - r - 1]);
        gather(va, c);
        gather(vb, b);
        let (left, right) = v.split_at_mut(r * n);
        if r + 1 < n {
            let pair = ((taus[c], &*va), (taus[b], &*vb));
            reflect_columns2(pair.0, pair.1, &mut right[n + b..], n, n - r - 1);
        }
        let col = &mut right[..n];
        make_column(col, r, taus[b], vb);
        if taus[c] != T::zero() {
            reflect_columns(taus[c], va, &mut col[b..], n, 1);
        }
        make_column(&mut left[b * n..], b, taus[c], va);
        r -= 2;
    }
    if r == 1 {
        let w = &mut wa[..n - 2];
        gather(w, 0);
        let right = &mut v[n..];
        if taus[0] != T::zero() && n > 2 {
            reflect_columns(taus[0], w, &mut right[n + 1..], n, n - 2);
        }
        make_column(&mut right[..n], 1, taus[0], w);
    }
    v[0] = T::one();
    v
}

/// `U = Q·[I; 0] = H₀·H₁·…·H_{n-1}·[I; 0]`, `m × n`, in place of the reflections whose vectors
/// [`bidiagonalize`] left below the diagonal of `a`, as LAPACK's `ung2r`: from the last, two at
/// a time, each pair applied to the columns of `U` it changes, then turned into its own two
/// columns.
fn form_u<T: Scalar>(a: &mut [T], m: usize, n: usize, taus: &[T]) {
    // `H_j` acts on the rows from `j`, its vector after the leading one in column `j` below the
    // diagonal, and makes column `j`.
    let mut j = n;
    while j >= 2 {
        // `H_{j-2}·H_{j-1}`, on the rows from `j - 2`.
        let (b, c) = (j - 1, j - 2);
        let (left, right) = a.split_at_mut(j * m);
        let (left, colb) = left.split_at_mut(b * m);
        let colc = &mut left[c * m..];
        if j < n {
            let pair = ((taus[c], &colc[c + 1..]), (taus[b], &colb[b + 1..]));
            reflect_columns2(pair.0, pair.1, &mut right[c..], m, n - j);
        }
        in_place_column(colb, b, taus[b]);
        if taus[c] != T::zero() {
            reflect_columns(taus[c], &colc[c + 1..], &mut colb[c..], m, 1);
        }
        in_place_column(colc, c, taus[c]);
        j -= 2;
    }
    if j == 1 {
        let (left, right) = a.split_at_mut(m);
        if taus[0] != T::zero() && n > 1 {
            reflect_columns(taus[0], &left[1..], right, m, n - 1);
        }
        in_place_column(left, 0, taus[0]);
    }
}

/// [`make_column`] with the vector already in place below the diagonal of `col`.
fn in_place_column<T: Scalar>(col: &mut [T], c: usize, tau: T) {
    col[..c].fill(T::zero());
    col[c] = T::one() - tau;
    let minus = -tau;
    for x in &mut col[c + 1..] {
        *x *= minus;
    }
}

/// Columns of a column-major block of any element type that the QR algorithm's rotations, real,
/// act on: the same rotation of the real and the imaginary parts.
struct Cols<'a, T> {
    /// The elements, column `j` from `j · ld`.
    data: &'a mut [T],
    /// The rows rotated.
    rows: usize,
    /// The leading dimension.
    ld: usize,
}

impl<T: Scalar> qr::Vectors<T::Real> for Cols<'_, T> {
    fn rotate(&mut self, i: usize, j: usize, c: T::Real, s: T::Real) {
        let (rows, ld) = (self.rows, self.ld);
        let (lo, hi) = (i.min(j), i.max(j));
        let (left, right) = self.data.split_at_mut(hi * ld);
        let (p, q) = (&mut left[lo * ld..lo * ld + rows], &mut right[..rows]);
        let (x, y) = if i < j { (p, q) } else { (q, p) };
        rotate(x, y, c, s);
    }

    fn negate(&mut self, j: usize) {
        for x in &mut self.data[j * self.ld..j * self.ld + self.rows] {
            *x = -*x;
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        let (rows, ld) = (self.rows, self.ld);
        let (left, right) = self.data.split_at_mut(j * ld);
        left[i * ld..i * ld + rows].swap_with_slice(&mut right[..rows]);
    }
}

/// `(x, y) = (c·x + s·y, c·y - s·x)`, the parts alike. Kept out of line, as the real one is.
#[inline(never)]
fn rotate<T: Scalar>(x: &mut [T], y: &mut [T], c: T::Real, s: T::Real) {
    let fma = qr::fma::<T::Real>;
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (*x, *y);
        *x = T::from_parts(fma(c, a.re(), s * b.re()), fma(c, a.im(), s * b.im()));
        *y = T::from_parts(fma(c, b.re(), -(s * a.re())), fma(c, b.im(), -(s * a.im())));
    }
}
