//! The Hermitian eigenproblem of small matrices, where the blocked path's fixed costs outweigh
//! its arithmetic: the reduction one column at a time, `Q` formed explicitly in the place of its
//! reflections, and the implicit QL algorithm's rotations applied straight to `Q`, as EISPACK's
//! `tred2` and `tql2`; the eigenvalues alone by the QL and QR algorithm without square roots. No
//! allocation but the results': the scratch is on the stack.

use super::super::tridiagonal;
use super::super::{Matrix, Scalar};
use super::{fma, fma_conj, lanes, lanes_of, unblocked};
use crate::math::complex::Float;

/// The most any type has solved here.
const LARGEST: usize = 160;

/// The orders whose scratch is the smallest of the sizes of it, then the middle one, so that
/// the smaller problems don't clear room for the largest.
const TIERS: [usize; 2] = [16, 48];

/// The largest order of the elements `T` solved here, as measured on an M4: past it, divide and
/// conquer and the reflections applied in blocks take less time. The rotations of the QL
/// algorithm fill the most lanes on real vectors, twice as many on single precision ones, which
/// moves their crossovers furthest. Without the vectors the paths differ in little but their
/// allocations: the large path's reduction is unblocked up to about there too.
pub(super) fn largest<T: Scalar>(want_v: bool) -> usize {
    match (want_v, T::IS_COMPLEX, size_of::<T::Real>()) {
        (false, _, _) => 128,
        (true, false, 4) => LARGEST,
        (true, false, _) => 112,
        (true, true, _) => 80,
    }
}

/// [`hermitian`](super::hermitian) of an `a` of order up to [`largest`].
pub(super) fn hermitian<T: Scalar>(
    a: &Matrix<T>,
    want_v: bool,
) -> (Vec<T::Real>, Option<Matrix<T>>) {
    let n = a.rows();
    if n <= TIERS[0] {
        solve::<T, { TIERS[0] }>(a, want_v)
    } else if n <= TIERS[1] {
        solve::<T, { TIERS[1] }>(a, want_v)
    } else {
        solve::<T, LARGEST>(a, want_v)
    }
}

/// [`hermitian`] with room for orders up to `N`.
fn solve<T: Scalar, const N: usize>(
    a: &Matrix<T>,
    want_v: bool,
) -> (Vec<T::Real>, Option<Matrix<T>>) {
    let n = a.rows();
    debug_assert!(n <= N);
    if n == 0 {
        return (Vec::new(), want_v.then(Matrix::default));
    }
    let nan = T::from_parts(T::Real::NAN, T::Real::NAN);
    let Some((mut h, scale)) = lower(a) else {
        let v = want_v.then(|| Matrix::from_column_major(n, n, vec![nan; n * n]));
        return (vec![T::Real::NAN; n], v);
    };
    let zero = T::Real::ZERO;
    let mut d = vec![zero; n];
    let (mut e, mut taus, mut w) = ([zero; N], [T::zero(); N], [T::zero(); N]);
    let m = n - 1;
    unblocked(
        &mut h,
        n,
        0,
        &mut d,
        &mut e[..m],
        &mut taus[..m],
        &mut w[..n],
    );
    let ok = if want_v {
        form_q(&mut h, n, &taus[..m]);
        tridiagonal::ql(&mut d, &e[..m], Some((&mut h, n, n)))
    } else {
        tridiagonal::values(&mut d, &mut e[..m])
    };
    if !ok {
        h.fill(nan);
        d.fill(T::Real::NAN);
    } else if scale != T::Real::ONE {
        let r = T::Real::ONE / scale;
        for x in &mut d {
            *x *= r;
        }
    }
    (d, want_v.then(|| Matrix::from_column_major(n, n, h)))
}

/// The lower triangle of `a`, scaled into the range where the reduction neither overflows nor
/// underflows as LAPACK's `heevd`, and the factor; `None` if any element of `a` is not finite.
fn lower<T: Scalar>(a: &Matrix<T>) -> Option<(Vec<T>, T::Real)> {
    let n = a.rows();
    let zero = T::Real::ZERO;
    let mut h = vec![T::zero(); n * n];
    // The largest part, within a factor √2 of the largest absolute value, which costs a square
    // root of a complex element; and whether any part is not finite, everywhere.
    let mut big = zero;
    let mut bad = false;
    for (j, (dst, col)) in h
        .chunks_exact_mut(n)
        .zip(a.data.chunks_exact(n))
        .enumerate()
    {
        for &x in &col[..j] {
            bad |= !x.is_finite();
        }
        for (y, &x) in dst[j..].iter_mut().zip(&col[j..]) {
            *y = x;
            big = big.max(x.re().abs().max(x.im().abs()));
            bad |= !x.is_finite();
        }
    }
    if bad {
        return None;
    }
    let eps = T::Real::EPSILON * T::Real::HALF;
    let rmin = (T::Real::MIN_POSITIVE / eps).sqrt();
    let rmax = (T::Real::ONE / (T::Real::MIN_POSITIVE / eps)).sqrt();
    if big >= rmin && big * T::Real::TWO.sqrt() <= rmax {
        return Some((h, T::Real::ONE));
    }
    // Near an end of the range: the largest absolute value itself decides, as `heevd`'s.
    let mut norm = zero;
    for j in 0..n {
        for x in &h[j * n + j..(j + 1) * n] {
            norm = norm.max(x.abs());
        }
    }
    // A complex element with finite parts whose absolute value overflows: its largest part
    // stands for it, within the factor √2 that the range has room for.
    if !norm.is_finite() {
        norm = big;
    }
    let scale = if norm > zero && norm < rmin {
        rmin / norm
    } else if norm > rmax {
        rmax / norm
    } else {
        return Some((h, T::Real::ONE));
    };
    for j in 0..n {
        for x in &mut h[j * n + j..(j + 1) * n] {
            *x = x.scale(scale);
        }
    }
    Some((h, scale))
}

/// `Q = H₀·H₁·…·H_{n-2}`, explicitly, over the reflections whose vectors the reduction left
/// below the subdiagonal of the `n × n` `h`, as LAPACK's `ungtr` and `ung2r`: from the last
/// reflections, two at a time, each pair applied to the columns of `Q` it changes, then turned
/// into its own two columns, which are the ones to the right of their vectors', so that nothing
/// need be moved first. `Q` has the first column and row of the identity.
fn form_q<T: Scalar>(h: &mut [T], n: usize, taus: &[T]) {
    if n == 0 {
        return;
    }
    // `H_r` acts on the rows from `r + 1`, its vector after the leading one in column `r` from
    // row `r + 2`, and makes column `r + 1` of `Q`.
    let mut r = n - 1;
    while r >= 2 {
        // `H_{r-2}·H_{r-1}`, on the rows from `r - 1`.
        let (b, a) = (r - 1, r - 2);
        let (left, right) = h.split_at_mut(r * n);
        let va = &left[a * n + b + 1..b * n];
        let vb = &left[b * n + r + 1..];
        if r + 1 < n {
            let pair = ((taus[a], va), (taus[b], vb));
            reflect_columns2(pair.0, pair.1, &mut right[n + b..], n, n - r - 1);
        }
        let col = &mut right[..n];
        make_column(col, r, taus[b], vb);
        if taus[a] != T::zero() {
            reflect_columns(taus[a], va, &mut col[b..], n, 1);
        }
        let (left, right) = h.split_at_mut(b * n);
        make_column(&mut right[..n], b, taus[a], &left[a * n + b + 1..]);
        r -= 2;
    }
    if r == 1 {
        // `H₀`, on the rows from 1.
        let (left, right) = h.split_at_mut(n);
        let v = &left[2..];
        if taus[0] != T::zero() && n > 2 {
            reflect_columns(taus[0], v, &mut right[n + 1..], n, n - 2);
        }
        make_column(&mut right[..n], 1, taus[0], v);
    }
    h[0] = T::one();
    h[1..n].fill(T::zero());
}

/// Sets `col` to the column `c` of the reflection `I - τ·(1, v)·(1, v)*` on the rows from `c`:
/// zeros above, `1 - τ` on the diagonal, `-τ·v` below.
pub(crate) fn make_column<T: Scalar>(col: &mut [T], c: usize, tau: T, v: &[T]) {
    col[..c].fill(T::zero());
    col[c] = T::one() - tau;
    let minus = -tau;
    for (x, &v) in col[c + 1..].iter_mut().zip(v) {
        *x = minus * v;
    }
}

/// `x -= τ·v·(v*·x)`, the reflection `I - τ·(1, v)·(1, v)*` applied from the left, for the
/// `cols` columns `x` of the block `c` (leading dimension `ld`), of `v.len() + 1` elements
/// each: four columns a pass, so that each element of `v` is loaded once for the four products
/// and once for the four updates, the products in lanes of two vector registers each, so that
/// their additions don't wait on one another.
pub(crate) fn reflect_columns<T: Scalar>(tau: T, v: &[T], c: &mut [T], ld: usize, cols: usize) {
    match size_of::<T>() {
        4 => reflect_columns_by::<T, 8>(tau, v, c, ld, cols),
        8 => reflect_columns_by::<T, 4>(tau, v, c, ld, cols),
        _ => reflect_columns_by::<T, 2>(tau, v, c, ld, cols),
    }
}

/// [`reflect_columns`] with `L` lanes.
fn reflect_columns_by<T: Scalar, const L: usize>(
    tau: T,
    v: &[T],
    c: &mut [T],
    ld: usize,
    cols: usize,
) {
    let len = v.len();
    let (vb, vt) = v.as_chunks::<L>();
    let body = len - vt.len();
    let mut j = 0;
    while j + 4 <= cols {
        let block = &mut c[j * ld..];
        let (c0, block) = block.split_at_mut(ld);
        let (c1, block) = block.split_at_mut(ld);
        let (c2, c3) = block.split_at_mut(ld);
        let (h0, r0) = c0[..=len].split_first_mut().expect("a row");
        let (h1, r1) = c1[..=len].split_first_mut().expect("a row");
        let (h2, r2) = c2[..=len].split_first_mut().expect("a row");
        let (h3, r3) = c3[..=len].split_first_mut().expect("a row");
        // `s_k = v*·x_k`, past the leading elements.
        let mut s = [[T::zero(); L]; 4];
        let it = vb
            .iter()
            .zip(lanes_of::<T, L>(r0))
            .zip(lanes_of::<T, L>(r1))
            .zip(lanes_of::<T, L>(r2).iter().zip(lanes_of::<T, L>(r3)));
        for (((v, x0), x1), (x2, x3)) in it {
            for l in 0..L {
                s[0][l] = fma_conj(v[l], x0[l], s[0][l]);
                s[1][l] = fma_conj(v[l], x1[l], s[1][l]);
                s[2][l] = fma_conj(v[l], x2[l], s[2][l]);
                s[3][l] = fma_conj(v[l], x3[l], s[3][l]);
            }
        }
        let mut w = [T::zero(); 4];
        for (k, (r, h)) in [&*r0, &*r1, &*r2, &*r3]
            .into_iter()
            .zip([&*h0, &*h1, &*h2, &*h3])
            .enumerate()
        {
            let mut t = lanes(s[k]);
            for (&v, &x) in vt.iter().zip(&r[body..]) {
                t = fma_conj(v, x, t);
            }
            w[k] = tau * (*h + t);
        }
        *h0 -= w[0];
        *h1 -= w[1];
        *h2 -= w[2];
        *h3 -= w[3];
        let m = [-w[0], -w[1], -w[2], -w[3]];
        let it = r0
            .iter_mut()
            .zip(r1.iter_mut())
            .zip(r2.iter_mut().zip(r3.iter_mut()))
            .zip(v);
        for (((x0, x1), (x2, x3)), &v) in it {
            *x0 = fma(v, m[0], *x0);
            *x1 = fma(v, m[1], *x1);
            *x2 = fma(v, m[2], *x2);
            *x3 = fma(v, m[3], *x3);
        }
        j += 4;
    }
    if j == cols {
        return;
    }
    for col in c[j * ld..].chunks_mut(ld).take(cols - j) {
        let (h, r) = col[..=len].split_first_mut().expect("a row");
        let mut s = [T::zero(); L];
        for (v, x) in vb.iter().zip(lanes_of::<T, L>(r)) {
            for l in 0..L {
                s[l] = fma_conj(v[l], x[l], s[l]);
            }
        }
        let mut t = lanes(s);
        for (&v, &x) in vt.iter().zip(&r[body..]) {
            t = fma_conj(v, x, t);
        }
        let w = tau * (*h + t);
        *h -= w;
        let m = -w;
        for (x, &v) in r.iter_mut().zip(v) {
            *x = fma(v, m, *x);
        }
    }
}

/// `x ← H_a·H_b·x` for the `cols` columns `x` of the block `c` (leading dimension `ld`), of
/// `va.len() + 1` elements each: `H_a = I - τ_a·(1, va)·(1, va)*` acting on all of them and
/// `H_b = I - τ_b·(1, vb)·(1, vb)*` on all but the first. The two reflections are applied in
/// one pass of products and one of updates, with each element stored once for both, through
/// `va*·H_b = va* - τ_b·(va*·vb)·vb*`; four columns a pass, the products in lanes.
pub(crate) fn reflect_columns2<T: Scalar>(
    (ta, va): (T, &[T]),
    (tb, vb): (T, &[T]),
    c: &mut [T],
    ld: usize,
    cols: usize,
) {
    match size_of::<T>() {
        4 => reflect_columns2_by::<T, 8>(ta, va, tb, vb, c, ld, cols),
        8 => reflect_columns2_by::<T, 4>(ta, va, tb, vb, c, ld, cols),
        _ => reflect_columns2_by::<T, 2>(ta, va, tb, vb, c, ld, cols),
    }
}

/// [`reflect_columns2`] with `L` lanes.
fn reflect_columns2_by<T: Scalar, const L: usize>(
    ta: T,
    va: &[T],
    tb: T,
    vb: &[T],
    c: &mut [T],
    ld: usize,
    cols: usize,
) {
    debug_assert_eq!(va.len(), vb.len() + 1);
    let len = va.len() + 1;
    // Rows from 2, where both vectors are held: `a` the rest of `va`, `b` that of `vb`.
    let (a1, a) = (va[0], &va[1..]);
    let b = vb;
    let rest = len - 2;
    let (ab, at) = a.as_chunks::<L>();
    let (bb, bt) = b.as_chunks::<L>();
    let body = rest - at.len();
    // `γ = va*·(0, 1, vb)`.
    let mut g = a1.conj();
    for (&a, &b) in a.iter().zip(b) {
        g = fma_conj(a, b, g);
    }
    // The factors of one column from its products: `w_b = τ_b·s_b`, `w_a = τ_a·(s_a - w_b·γ)`.
    let factors = |x0: T, x1: T, sa: T, sb: T| {
        let wb = tb * (x1 + sb);
        let wa = ta * (x0 + fma_conj(a1, x1, sa) - wb * g);
        (wa, wb)
    };
    let mut j = 0;
    while j + 4 <= cols {
        let block = &mut c[j * ld..];
        let (c0, block) = block.split_at_mut(ld);
        let (c1, block) = block.split_at_mut(ld);
        let (c2, c3) = block.split_at_mut(ld);
        let mut x = [c0, c1, c2, c3].map(|c| &mut c[..len]);
        let mut sa = [[T::zero(); L]; 4];
        let mut sb = [[T::zero(); L]; 4];
        {
            let [x0, x1, x2, x3] = x.each_ref().map(|x| lanes_of::<T, L>(&x[2..]));
            let it = ab
                .iter()
                .zip(bb)
                .zip(x0.iter().zip(x1))
                .zip(x2.iter().zip(x3));
            for (((a, b), (x0, x1)), (x2, x3)) in it {
                for l in 0..L {
                    sa[0][l] = fma_conj(a[l], x0[l], sa[0][l]);
                    sb[0][l] = fma_conj(b[l], x0[l], sb[0][l]);
                    sa[1][l] = fma_conj(a[l], x1[l], sa[1][l]);
                    sb[1][l] = fma_conj(b[l], x1[l], sb[1][l]);
                    sa[2][l] = fma_conj(a[l], x2[l], sa[2][l]);
                    sb[2][l] = fma_conj(b[l], x2[l], sb[2][l]);
                    sa[3][l] = fma_conj(a[l], x3[l], sa[3][l]);
                    sb[3][l] = fma_conj(b[l], x3[l], sb[3][l]);
                }
            }
        }
        let mut wa = [T::zero(); 4];
        let mut wb = [T::zero(); 4];
        for (k, x) in x.iter().enumerate() {
            let (mut ta_, mut tb_) = (lanes(sa[k]), lanes(sb[k]));
            for ((&a, &b), &x) in at.iter().zip(bt).zip(&x[2 + body..]) {
                ta_ = fma_conj(a, x, ta_);
                tb_ = fma_conj(b, x, tb_);
            }
            (wa[k], wb[k]) = factors(x[0], x[1], ta_, tb_);
        }
        for (k, x) in x.iter_mut().enumerate() {
            x[0] -= wa[k];
            x[1] -= fma(wa[k], a1, wb[k]);
        }
        let (ma, mb) = (wa.map(|w| -w), wb.map(|w| -w));
        let [x0, x1, x2, x3] = &mut x;
        let it = x0[2..]
            .iter_mut()
            .zip(x1[2..].iter_mut())
            .zip(x2[2..].iter_mut().zip(x3[2..].iter_mut()))
            .zip(a.iter().zip(b));
        for (((x0, x1), (x2, x3)), (&a, &b)) in it {
            *x0 = fma(b, mb[0], fma(a, ma[0], *x0));
            *x1 = fma(b, mb[1], fma(a, ma[1], *x1));
            *x2 = fma(b, mb[2], fma(a, ma[2], *x2));
            *x3 = fma(b, mb[3], fma(a, ma[3], *x3));
        }
        j += 4;
    }
    if j == cols {
        return;
    }
    for col in c[j * ld..].chunks_mut(ld).take(cols - j) {
        let x = &mut col[..len];
        let (mut sa, mut sb) = ([T::zero(); L], [T::zero(); L]);
        for ((a, b), y) in ab.iter().zip(bb).zip(lanes_of::<T, L>(&x[2..])) {
            for l in 0..L {
                sa[l] = fma_conj(a[l], y[l], sa[l]);
                sb[l] = fma_conj(b[l], y[l], sb[l]);
            }
        }
        let (mut ta_, mut tb_) = (lanes(sa), lanes(sb));
        for ((&a, &b), &y) in at.iter().zip(bt).zip(&x[2 + body..]) {
            ta_ = fma_conj(a, y, ta_);
            tb_ = fma_conj(b, y, tb_);
        }
        let (wa, wb) = factors(x[0], x[1], ta_, tb_);
        x[0] -= wa;
        x[1] -= fma(wa, a1, wb);
        let (ma, mb) = (-wa, -wb);
        for (y, (&a, &b)) in x[2..].iter_mut().zip(a.iter().zip(b)) {
            *y = fma(b, mb, fma(a, ma, *y));
        }
    }
}
