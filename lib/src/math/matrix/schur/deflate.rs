//! Aggressive early deflation, as LAPACK's `laqr3`: the Schur form of a window at the bottom of
//! the active block, the eigenvalues whose part of the spike is negligible deflated, the others
//! moved to the top of the window and given back as shifts. For real elements the reordering
//! moves the `2 × 2` blocks of the real Schur form, as LAPACK's `trexc`, `laexc` and `lasy2`.

use super::super::Scalar;
use super::super::blas::{Op, gemm};
use super::super::householder::{Block, PANEL, reflect_left, reflect_right, reflector};
use super::hessenberg::reduce;
use super::multishift::{multishift, nmin};
use super::{real_reflector, rotate, small_qr, standardize};
use crate::math::complex::{Complex, Float};

/// Aggressive early deflation on the window of order `nw` at the bottom of the active block
/// `ktop..=kbot` of the upper Hessenberg `h`, as LAPACK's `laqr3` (`laqr2` when `nested`, its
/// window then reduced by the QR algorithm with one or two shifts only): the window's
/// eigenvalues into `w`, those deflated at the bottom. Returns the number of shifts, the
/// undeflated eigenvalues just above the deflated ones, and the number deflated.
#[allow(clippy::too_many_arguments)]
pub(super) fn deflate<T: Scalar>(
    want_t: bool,
    h: &mut [T],
    n: usize,
    z: Option<&mut [T]>,
    ktop: usize,
    kbot: usize,
    nw: usize,
    w: &mut [Complex<T::Real>],
    nested: bool,
) -> (usize, usize) {
    let at = |i: usize, j: usize| i + j * n;
    let ulp = T::Real::EPSILON;
    let smlnum = T::Real::MIN_POSITIVE * (T::Real::from_f64(n as f64) / ulp);
    let jw = nw.min(kbot + 1 - ktop);
    if jw == 0 {
        return (0, 0);
    }
    let kwtop = kbot + 1 - jw;
    // The spike: the subdiagonal element left of the window, times `V`'s first row.
    let mut s = if kwtop == ktop {
        T::zero()
    } else {
        h[at(kwtop, kwtop - 1)]
    };
    if kbot == kwtop {
        w[kwtop] = h[at(kwtop, kwtop)].to_complex();
        if s.abs1() <= smlnum.max(ulp * h[at(kwtop, kwtop)].abs1()) {
            if kwtop > ktop {
                h[at(kwtop, kwtop - 1)] = T::zero();
            }
            return (0, 1);
        }
        return (1, 0);
    }
    let tt = |i: usize, j: usize| i + j * jw;
    let mut t = vec![T::zero(); jw * jw];
    for j in 0..jw {
        for i in 0..=(j + 1).min(jw - 1) {
            t[tt(i, j)] = h[at(kwtop + i, kwtop + j)];
        }
    }
    let mut v = vec![T::zero(); jw * jw];
    for i in 0..jw {
        v[tt(i, i)] = T::one();
    }
    let ws = &mut w[kwtop..=kbot];
    let infqr = if jw > nmin::<T>(true) && !nested {
        multishift(true, &mut t, jw, Some(&mut v), ws, true)
    } else {
        small_qr(&mut t, jw, Some(&mut v), true, ws)
    };
    for j in 0..jw {
        for i in j + 2..jw {
            t[tt(i, j)] = T::zero();
        }
    }
    // Deflation: the eigenvalues at the bottom whose spike is negligible stay, the others go to
    // the top, `ns` the undeflated so far and `ilst` where the next one goes.
    let mut ns = jw;
    let mut ilst = infqr;
    if T::IS_COMPLEX {
        for _ in infqr..jw {
            let mut size = t[tt(ns - 1, ns - 1)].abs1();
            if size == T::Real::ZERO {
                size = s.abs1();
            }
            if s.abs1() * v[tt(0, ns - 1)].abs1() <= smlnum.max(ulp * size) {
                ns -= 1;
            } else {
                move_triangular(&mut t, jw, &mut v, ns - 1, ilst);
                ilst += 1;
            }
        }
    } else {
        let re = |t: &[T], i: usize, j: usize| t[tt(i, j)].re();
        let sr = s.re();
        while ilst < ns {
            if ns == 1 || re(&t, ns - 1, ns - 2) == T::Real::ZERO {
                let mut size = re(&t, ns - 1, ns - 1).abs();
                if size == T::Real::ZERO {
                    size = sr.abs();
                }
                if (sr * v[tt(0, ns - 1)].re()).abs() <= smlnum.max(ulp * size) {
                    ns -= 1;
                } else {
                    let mut ifst = ns - 1;
                    reorder(&mut t, jw, &mut v, &mut ifst, &mut ilst);
                    ilst += 1;
                }
            } else {
                let mut size = re(&t, ns - 1, ns - 1).abs()
                    + re(&t, ns - 1, ns - 2).abs().sqrt() * re(&t, ns - 2, ns - 1).abs().sqrt();
                if size == T::Real::ZERO {
                    size = sr.abs();
                }
                let spike = (sr * v[tt(0, ns - 1)].re())
                    .abs()
                    .max((sr * v[tt(0, ns - 2)].re()).abs());
                if spike <= smlnum.max(ulp * size) {
                    ns -= 2;
                } else {
                    let mut ifst = ns - 1;
                    reorder(&mut t, jw, &mut v, &mut ifst, &mut ilst);
                    ilst += 2;
                }
            }
        }
    }
    if ns == 0 {
        s = T::zero();
    }
    if ns < jw {
        // The undeflated eigenvalues largest first, which helps graded matrices.
        if T::IS_COMPLEX {
            for i in infqr..ns {
                let mut ifst = i;
                for j in i + 1..ns {
                    if t[tt(j, j)].abs1() > t[tt(ifst, ifst)].abs1() {
                        ifst = j;
                    }
                }
                if ifst != i {
                    move_triangular(&mut t, jw, &mut v, ifst, i);
                }
            }
        } else {
            sort_real(&mut t, jw, &mut v, infqr, ns);
        }
    }
    // The eigenvalues from `T`.
    if T::IS_COMPLEX {
        for i in infqr..jw {
            w[kwtop + i] = t[tt(i, i)].to_complex();
        }
    } else {
        let re = |i: usize, j: usize| t[tt(i, j)].re();
        let mut i = jw;
        while i > infqr {
            if i == infqr + 1 || re(i - 1, i - 2) == T::Real::ZERO {
                w[kwtop + i - 1] = Complex::new(re(i - 1, i - 1), T::Real::ZERO);
                i -= 1;
            } else {
                let b = standardize(
                    re(i - 2, i - 2),
                    re(i - 2, i - 1),
                    re(i - 1, i - 2),
                    re(i - 1, i - 1),
                );
                w[kwtop + i - 2] = Complex::new(b.rt1r, b.rt1i);
                w[kwtop + i - 1] = Complex::new(b.rt2r, b.rt2i);
                i -= 2;
            }
        }
    }
    if ns < jw || s == T::zero() {
        let reflected = ns > 1 && s != T::zero();
        let mut taus = Vec::new();
        if reflected {
            // The spike's undeflated part back to a multiple of the first column, then the
            // window's leading block back to Hessenberg form.
            let mut x: Vec<T> = (0..ns).map(|j| v[tt(0, j)].conj()).collect();
            let (tau, _) = {
                let (first, rest) = x.split_at_mut(1);
                reflector(first[0], rest)
            };
            x[0] = T::one();
            reflect_left(tau, true, &x, ns, jw, &mut t, jw);
            reflect_right(tau, &x, ns, ns, &mut t, jw);
            reflect_right(tau, &x, jw, ns, &mut v, jw);
            taus = vec![T::zero(); ns.saturating_sub(2)];
            reduce(&mut t, jw, ns, &mut taus);
        }
        if kwtop > 0 {
            h[at(kwtop, kwtop - 1)] = s * v[tt(0, 0)].conj();
        }
        for j in 0..jw {
            for i in 0..=(j + 1).min(jw - 1) {
                h[at(kwtop + i, kwtop + j)] = t[tt(i, j)];
            }
        }
        if reflected && ns > 2 {
            // `V = V·Q`, `Q` the Hessenberg reduction's reflections, a panel at a time.
            let k = ns - 2;
            let mut start = 0;
            while start < k {
                let end = (start + PANEL).min(k);
                let block = Block::new(
                    &t[tt(start + 1, start)..],
                    jw,
                    ns - 1 - start,
                    &taus[start..end],
                );
                block.apply_right(jw, &mut v[(start + 1) * jw..], jw);
                start = end;
            }
        }
        // The rest of `h` and `z`: the columns of the window above it times `V`, its rows to
        // the right times `V*`.
        let top = if want_t { 0 } else { ktop };
        if top < kwtop {
            let rows = kwtop - top;
            let mut tmp = vec![T::zero(); rows * jw];
            gemm(
                rows,
                jw,
                jw,
                T::one(),
                &h[at(top, kwtop)..],
                n,
                Op::N,
                &v,
                jw,
                Op::N,
                T::zero(),
                &mut tmp,
                rows,
            );
            for j in 0..jw {
                h[at(top, kwtop + j)..at(kwtop, kwtop + j)]
                    .copy_from_slice(&tmp[j * rows..(j + 1) * rows]);
            }
        }
        if want_t && kbot + 1 < n {
            let cols = n - kbot - 1;
            let mut vh = vec![T::zero(); jw * jw];
            for j in 0..jw {
                for i in 0..jw {
                    vh[tt(j, i)] = v[tt(i, j)].conj();
                }
            }
            let mut tmp = vec![T::zero(); jw * cols];
            gemm(
                jw,
                cols,
                jw,
                T::one(),
                &vh,
                jw,
                Op::N,
                &h[at(kwtop, kbot + 1)..],
                n,
                Op::N,
                T::zero(),
                &mut tmp,
                jw,
            );
            for j in 0..cols {
                h[at(kwtop, kbot + 1 + j)..at(kbot + 1, kbot + 1 + j)]
                    .copy_from_slice(&tmp[j * jw..(j + 1) * jw]);
            }
        }
        if let Some(z) = z {
            let mut tmp = vec![T::zero(); n * jw];
            gemm(
                n,
                jw,
                jw,
                T::one(),
                &z[at(0, kwtop)..],
                n,
                Op::N,
                &v,
                jw,
                Op::N,
                T::zero(),
                &mut tmp,
                n,
            );
            z[at(0, kwtop)..at(0, kbot + 1)].copy_from_slice(&tmp);
        }
    }
    (ns - infqr, jw - ns)
}

/// The rotation `[[c, s], [-s̄, c]]`, `c` real, taking `(f, g)` to `(r, 0)`: `(c, s)`.
fn rotation<T: Scalar>(f: T, g: T) -> (T::Real, T) {
    if g == T::zero() {
        return (T::Real::ONE, T::zero());
    }
    if f == T::zero() {
        return (T::Real::ZERO, g.conj().scale(T::Real::ONE / g.abs()));
    }
    let (af, ag) = (f.abs(), g.abs());
    let d = af.hypot(ag);
    (
        af / d,
        f.scale(T::Real::ONE / af) * g.conj().scale(T::Real::ONE / d),
    )
}

/// Swaps the diagonal elements `k` and `k + 1` of the upper triangular `t` (order `n`) by a
/// rotation, accumulated into `q`, as LAPACK's complex `trexc` (and `laexc` for two real
/// `1 × 1` blocks).
fn swap_diagonal<T: Scalar>(t: &mut [T], n: usize, q: &mut [T], k: usize) {
    let at = |i: usize, j: usize| i + j * n;
    let (t11, t22) = (t[at(k, k)], t[at(k + 1, k + 1)]);
    let (c, s) = rotation(t[at(k, k + 1)], t22 - t11);
    let sc = s.conj();
    for j in k + 2..n {
        let (x, y) = (t[at(k, j)], t[at(k + 1, j)]);
        t[at(k, j)] = x.scale(c) + s * y;
        t[at(k + 1, j)] = y.scale(c) - sc * x;
    }
    let rotate_columns = |a: &mut [T], rows: usize| {
        let (x, y) = a[k * n..].split_at_mut(n);
        for (x, y) in x[..rows].iter_mut().zip(&mut y[..rows]) {
            let (a, b) = (*x, *y);
            *x = a.scale(c) + sc * b;
            *y = b.scale(c) - s * a;
        }
    };
    rotate_columns(t, k);
    rotate_columns(q, n);
    t[at(k, k)] = t22;
    t[at(k + 1, k + 1)] = t11;
}

/// Moves the diagonal element `from` of the upper triangular `t` to `to` by swaps of
/// neighbours.
fn move_triangular<T: Scalar>(t: &mut [T], n: usize, q: &mut [T], from: usize, to: usize) {
    if from > to {
        for k in (to..from).rev() {
            swap_diagonal(t, n, q, k);
        }
    } else {
        for k in from..to {
            swap_diagonal(t, n, q, k);
        }
    }
}

/// The undeflated blocks `first..ns` of the real Schur form `t` sorted by decreasing absolute
/// value of the eigenvalues, by a bubble sort, which copes with a swap refused.
fn sort_real<T: Scalar>(t: &mut [T], n: usize, q: &mut [T], first: usize, ns: usize) {
    let sub = |t: &[T], i: usize| t[i + (i - 1) * n].re() != T::Real::ZERO;
    // The size of the block at `i` within `..end`.
    let size = |t: &[T], i: usize, end: usize| if i + 1 < end && sub(t, i + 1) { 2 } else { 1 };
    let magnitude = |t: &[T], i: usize, nb: usize| {
        let d = t[i + i * n].re().abs();
        if nb == 1 {
            d
        } else {
            d + t[i + 1 + i * n].re().abs().sqrt() * t[i + (i + 1) * n].re().abs().sqrt()
        }
    };
    let mut end = ns;
    let mut sorted = false;
    while !sorted {
        sorted = true;
        let mut i = first;
        let mut k = i + size(t, i, end);
        let mut last = i;
        while k < end {
            let (ni, nk) = (size(t, i, end), size(t, k, end));
            if magnitude(t, i, ni) >= magnitude(t, k, nk) {
                i = k;
            } else {
                sorted = false;
                let (mut ifst, mut ilst) = (i, k);
                i = if reorder(t, n, q, &mut ifst, &mut ilst) {
                    ilst
                } else {
                    k
                };
            }
            last = i;
            k = i + size(t, i, end);
        }
        end = last;
    }
}

/// Moves the diagonal block of the real Schur form `t` (order `n`) at row `ifst` to row `ilst`
/// by swaps of neighbouring blocks, accumulated into `q`, as LAPACK's `trexc`: both are moved to
/// the first rows of their blocks, and `ilst` ends where the block did. Returns false when a
/// swap was refused, the block then left at `ilst`.
fn reorder<T: Scalar>(
    t: &mut [T],
    n: usize,
    q: &mut [T],
    ifst: &mut usize,
    ilst: &mut usize,
) -> bool {
    if n <= 1 {
        return true;
    }
    let sub = |t: &[T], i: usize| t[i + (i - 1) * n].re() != T::Real::ZERO;
    if *ifst > 0 && sub(t, *ifst) {
        *ifst -= 1;
    }
    let mut nbf = if *ifst + 1 < n && sub(t, *ifst + 1) {
        2
    } else {
        1
    };
    if *ilst > 0 && sub(t, *ilst) {
        *ilst -= 1;
    }
    let nbl = if *ilst + 1 < n && sub(t, *ilst + 1) {
        2
    } else {
        1
    };
    if *ifst == *ilst {
        return true;
    }
    let mut here = *ifst;
    if *ifst < *ilst {
        if nbf == 2 && nbl == 1 {
            *ilst -= 1;
        }
        if nbf == 1 && nbl == 2 {
            *ilst += 1;
        }
        while here < *ilst {
            if nbf != 3 {
                let next = if here + nbf + 1 < n && sub(t, here + nbf + 1) {
                    2
                } else {
                    1
                };
                if !swap_blocks(t, n, q, here, nbf, next) {
                    *ilst = here;
                    return false;
                }
                here += next;
                if nbf == 2 && !sub(t, here + 1) {
                    nbf = 3;
                }
            } else {
                // A block that split in two, moved one at a time.
                let mut next = if here + 3 < n && sub(t, here + 3) {
                    2
                } else {
                    1
                };
                if !swap_blocks(t, n, q, here + 1, 1, next) {
                    *ilst = here;
                    return false;
                }
                if next == 1 {
                    swap_blocks(t, n, q, here, 1, 1);
                    here += 1;
                } else {
                    if !sub(t, here + 2) {
                        next = 1;
                    }
                    if next == 2 {
                        if !swap_blocks(t, n, q, here, 1, 2) {
                            *ilst = here;
                            return false;
                        }
                    } else {
                        swap_blocks(t, n, q, here, 1, 1);
                        swap_blocks(t, n, q, here + 1, 1, 1);
                    }
                    here += 2;
                }
            }
        }
    } else {
        while here > *ilst {
            if nbf != 3 {
                let next = if here >= 2 && sub(t, here - 1) { 2 } else { 1 };
                if !swap_blocks(t, n, q, here - next, next, nbf) {
                    *ilst = here;
                    return false;
                }
                here -= next;
                if nbf == 2 && !sub(t, here + 1) {
                    nbf = 3;
                }
            } else {
                let mut next = if here >= 2 && sub(t, here - 1) { 2 } else { 1 };
                if !swap_blocks(t, n, q, here - next, next, 1) {
                    *ilst = here;
                    return false;
                }
                if next == 1 {
                    swap_blocks(t, n, q, here, 1, 1);
                    here -= 1;
                } else {
                    if !sub(t, here) {
                        next = 1;
                    }
                    if next == 2 {
                        if !swap_blocks(t, n, q, here - 1, 2, 1) {
                            *ilst = here;
                            return false;
                        }
                    } else {
                        swap_blocks(t, n, q, here, 1, 1);
                        swap_blocks(t, n, q, here - 1, 1, 1);
                    }
                    here -= 2;
                }
            }
        }
    }
    *ilst = here;
    true
}

/// `I - τ·u·uᵀ` on three rows `r..r + 3` of columns `cols` (from the left) or on three columns
/// `r..r + 3` of rows `cols` (from the right) of `t`, of real elements.
fn reflect3<T: Scalar>(
    t: &mut [T],
    n: usize,
    left: bool,
    r: usize,
    cols: core::ops::Range<usize>,
    u: [T::Real; 3],
    tau: T::Real,
) {
    let (s, step) = if left { (n, 1) } else { (1, n) };
    for j in cols {
        let i0 = r * step + j * s;
        let (x0, x1, x2) = (t[i0], t[i0 + step], t[i0 + 2 * step]);
        let sum = x0.scale(u[0]) + x1.scale(u[1]) + x2.scale(u[2]);
        t[i0] = x0 - sum.scale(tau * u[0]);
        t[i0 + step] = x1 - sum.scale(tau * u[1]);
        t[i0 + 2 * step] = x2 - sum.scale(tau * u[2]);
    }
}

/// [`reflect3`] on a small dense real block of order 4.
fn reflect3_small<R: Float>(
    d: &mut [[R; 4]; 4],
    left: bool,
    r: usize,
    cols: core::ops::Range<usize>,
    u: [R; 3],
    tau: R,
) {
    for j in cols {
        let get = |d: &[[R; 4]; 4], p: usize| if left { d[r + p][j] } else { d[j][r + p] };
        let sum = get(d, 0) * u[0] + get(d, 1) * u[1] + get(d, 2) * u[2];
        for (p, &u) in u.iter().enumerate() {
            let x = if left {
                &mut d[r + p][j]
            } else {
                &mut d[j][r + p]
            };
            *x -= sum * tau * u;
        }
    }
}

/// Swaps the neighbouring diagonal blocks of orders `n1` and `n2` at row `j1` of the real Schur
/// form `t` (order `n`), accumulated into `q`, as LAPACK's `laexc`, the new blocks of order 2
/// standardized. Returns false, leaving everything as it was, when the swap would change the
/// eigenvalues too much.
fn swap_blocks<T: Scalar>(
    t: &mut [T],
    n: usize,
    q: &mut [T],
    j1: usize,
    n1: usize,
    n2: usize,
) -> bool {
    let at = |i: usize, j: usize| i + j * n;
    if n1 == 0 || n2 == 0 || j1 + n1 >= n {
        return true;
    }
    if n1 == 1 && n2 == 1 {
        swap_diagonal(t, n, q, j1);
        return true;
    }
    let (j2, j3, j4) = (j1 + 1, j1 + 2, j1 + 3);
    let nd = n1 + n2;
    let mut d = [[T::Real::ZERO; 4]; 4];
    let mut dnorm = T::Real::ZERO;
    for (i, row) in d.iter_mut().enumerate().take(nd) {
        for (j, x) in row.iter_mut().enumerate().take(nd) {
            *x = t[at(j1 + i, j1 + j)].re();
            dnorm = dnorm.max(x.abs());
        }
    }
    let eps = T::Real::EPSILON;
    let smlnum = T::Real::MIN_POSITIVE / eps;
    let thresh = (T::Real::from_f64(10.0) * eps * dnorm).max(smlnum);
    // `T₁₁·X - X·T₂₂ = scale·T₁₂`.
    let block = |r: usize, c: usize| [[d[r][c], d[r][c + 1]], [d[r + 1][c], d[r + 1][c + 1]]];
    let (x, scale) = sylvester(n1, n2, block(0, 0), block(n1, n1), block(0, n1));
    let one = T::Real::ONE;
    if n1 == 1 {
        // `(scale, x₁₁, x₁₂)·H = (0, 0, *)`.
        let mut u = [scale, x[0][0], x[0][1]];
        let (tau, _) = {
            let (head, last) = u.split_at_mut(2);
            real_reflector(last[0], head)
        };
        u[2] = one;
        let t11 = t[at(j1, j1)];
        reflect3_small(&mut d, true, 0, 0..3, u, tau);
        reflect3_small(&mut d, false, 0, 0..3, u, tau);
        if d[2][0]
            .abs()
            .max(d[2][1].abs())
            .max((d[2][2] - t11.re()).abs())
            > thresh
        {
            return false;
        }
        reflect3(t, n, true, j1, j1..n, u, tau);
        reflect3(t, n, false, j1, 0..j2 + 1, u, tau);
        t[at(j3, j1)] = T::zero();
        t[at(j3, j2)] = T::zero();
        t[at(j3, j3)] = t11;
        reflect3(q, n, false, j1, 0..n, u, tau);
    } else if n2 == 1 {
        // `H·(-x₁₁, -x₂₁, scale) = (*, 0, 0)`.
        let mut u = [-x[0][0], -x[1][0], scale];
        let (tau, _) = {
            let (first, rest) = u.split_at_mut(1);
            real_reflector(first[0], rest)
        };
        u[0] = one;
        let t33 = t[at(j3, j3)];
        reflect3_small(&mut d, true, 0, 0..3, u, tau);
        reflect3_small(&mut d, false, 0, 0..3, u, tau);
        if d[1][0]
            .abs()
            .max(d[2][0].abs())
            .max((d[0][0] - t33.re()).abs())
            > thresh
        {
            return false;
        }
        reflect3(t, n, false, j1, 0..j3 + 1, u, tau);
        reflect3(t, n, true, j1, j2..n, u, tau);
        t[at(j1, j1)] = t33;
        t[at(j2, j1)] = T::zero();
        t[at(j3, j1)] = T::zero();
        reflect3(q, n, false, j1, 0..n, u, tau);
    } else {
        // Two reflections taking `(-X; scale·I)` to upper triangular.
        let mut u1 = [-x[0][0], -x[1][0], scale];
        let (tau1, _) = {
            let (first, rest) = u1.split_at_mut(1);
            real_reflector(first[0], rest)
        };
        u1[0] = one;
        let temp = -tau1 * (x[0][1] + u1[1] * x[1][1]);
        let mut u2 = [-temp * u1[1] - x[1][1], -temp * u1[2], scale];
        let (tau2, _) = {
            let (first, rest) = u2.split_at_mut(1);
            real_reflector(first[0], rest)
        };
        u2[0] = one;
        reflect3_small(&mut d, true, 0, 0..4, u1, tau1);
        reflect3_small(&mut d, false, 0, 0..4, u1, tau1);
        reflect3_small(&mut d, true, 1, 0..4, u2, tau2);
        reflect3_small(&mut d, false, 1, 0..4, u2, tau2);
        let worst = d[2][0]
            .abs()
            .max(d[2][1].abs())
            .max(d[3][0].abs())
            .max(d[3][1].abs());
        if worst > thresh {
            return false;
        }
        reflect3(t, n, true, j1, j1..n, u1, tau1);
        reflect3(t, n, false, j1, 0..j4 + 1, u1, tau1);
        reflect3(t, n, true, j2, j1..n, u2, tau2);
        reflect3(t, n, false, j2, 0..j4 + 1, u2, tau2);
        t[at(j3, j1)] = T::zero();
        t[at(j3, j2)] = T::zero();
        t[at(j4, j1)] = T::zero();
        t[at(j4, j2)] = T::zero();
        reflect3(q, n, false, j1, 0..n, u1, tau1);
        reflect3(q, n, false, j2, 0..n, u2, tau2);
    }
    if n2 == 2 {
        standardize_block(t, n, q, j1);
    }
    if n1 == 2 {
        standardize_block(t, n, q, j1 + n2);
    }
    true
}

/// The `2 × 2` block at row `j` of `t` in standard form, the rotation applied to the rest of
/// `t` and to `q`.
fn standardize_block<T: Scalar>(t: &mut [T], n: usize, q: &mut [T], j: usize) {
    let at = |i: usize, k: usize| i + k * n;
    let re = |t: &[T], i: usize, k: usize| t[at(i, k)].re();
    let s = standardize(
        re(t, j, j),
        re(t, j, j + 1),
        re(t, j + 1, j),
        re(t, j + 1, j + 1),
    );
    t[at(j, j)] = T::from_real(s.a);
    t[at(j, j + 1)] = T::from_real(s.b);
    t[at(j + 1, j)] = T::from_real(s.c);
    t[at(j + 1, j + 1)] = T::from_real(s.d);
    let turn = |x: &mut T, y: &mut T| {
        let (mut a, mut b) = (x.re(), y.re());
        rotate(&mut a, &mut b, s.cs, s.sn);
        *x = T::from_real(a);
        *y = T::from_real(b);
    };
    for k in j + 2..n {
        let (mut x, mut y) = (t[at(j, k)], t[at(j + 1, k)]);
        turn(&mut x, &mut y);
        t[at(j, k)] = x;
        t[at(j + 1, k)] = y;
    }
    for (a, rows) in [(&mut *t, j), (&mut *q, n)] {
        let (x, y) = a[j * n..].split_at_mut(n);
        for (x, y) in x[..rows].iter_mut().zip(&mut y[..rows]) {
            turn(x, y);
        }
    }
}

/// The solution `X` of `TL·X - X·TR = scale·B` for `TL` of order `n1` and `TR` of order `n2`,
/// each 1 or 2, as LAPACK's `lasy2`: Gaussian elimination with complete pivoting, the pivots
/// too small replaced, `scale ≤ 1` keeping `X` from overflowing.
#[allow(clippy::type_complexity)]
fn sylvester<R: Float>(
    n1: usize,
    n2: usize,
    tl: [[R; 2]; 2],
    tr: [[R; 2]; 2],
    b: [[R; 2]; 2],
) -> ([[R; 2]; 2], R) {
    let (zero, one) = (R::ZERO, R::ONE);
    let eps = R::EPSILON;
    let smlnum = R::MIN_POSITIVE / eps;
    let sgn = -one;
    let mut x = [[zero; 2]; 2];
    if n1 == 1 && n2 == 1 {
        let mut tau1 = tl[0][0] + sgn * tr[0][0];
        let mut bet = tau1.abs();
        if bet <= smlnum {
            tau1 = smlnum;
            bet = smlnum;
        }
        let mut scale = one;
        let gam = b[0][0].abs();
        if smlnum * gam > bet {
            scale = one / gam;
        }
        x[0][0] = (b[0][0] * scale) / tau1;
        return (x, scale);
    }
    if n1 + n2 == 3 {
        // A 2 × 2 system, `tmp` column after column.
        let (smin, tmp, mut btmp) = if n1 == 1 {
            let big = tl[0][0]
                .abs()
                .max(tr[0][0].abs())
                .max(tr[0][1].abs())
                .max(tr[1][0].abs())
                .max(tr[1][1].abs());
            let tmp = [
                tl[0][0] + sgn * tr[0][0],
                sgn * tr[0][1],
                sgn * tr[1][0],
                tl[0][0] + sgn * tr[1][1],
            ];
            ((eps * big).max(smlnum), tmp, [b[0][0], b[0][1]])
        } else {
            let big = tr[0][0]
                .abs()
                .max(tl[0][0].abs())
                .max(tl[0][1].abs())
                .max(tl[1][0].abs())
                .max(tl[1][1].abs());
            let tmp = [
                tl[0][0] + sgn * tr[0][0],
                tl[1][0],
                tl[0][1],
                tl[1][1] + sgn * tr[0][0],
            ];
            ((eps * big).max(smlnum), tmp, [b[0][0], b[1][0]])
        };
        const U12: [usize; 4] = [2, 3, 0, 1];
        const L21: [usize; 4] = [1, 0, 3, 2];
        const U22: [usize; 4] = [3, 2, 1, 0];
        const XSWAP: [bool; 4] = [false, false, true, true];
        const BSWAP: [bool; 4] = [false, true, false, true];
        let mut ipiv = 0;
        for i in 1..4 {
            if tmp[i].abs() > tmp[ipiv].abs() {
                ipiv = i;
            }
        }
        let mut u11 = tmp[ipiv];
        if u11.abs() <= smin {
            u11 = smin;
        }
        let u12 = tmp[U12[ipiv]];
        let l21 = tmp[L21[ipiv]] / u11;
        let mut u22 = tmp[U22[ipiv]] - u12 * l21;
        if u22.abs() <= smin {
            u22 = smin;
        }
        if BSWAP[ipiv] {
            let temp = btmp[1];
            btmp[1] = btmp[0] - l21 * temp;
            btmp[0] = temp;
        } else {
            btmp[1] -= l21 * btmp[0];
        }
        let mut scale = one;
        let two = R::TWO;
        if two * smlnum * btmp[1].abs() > u22.abs() || two * smlnum * btmp[0].abs() > u11.abs() {
            scale = R::HALF / btmp[0].abs().max(btmp[1].abs());
            btmp[0] *= scale;
            btmp[1] *= scale;
        }
        let mut x2 = [zero; 2];
        x2[1] = btmp[1] / u22;
        x2[0] = btmp[0] / u11 - (u12 / u11) * x2[1];
        if XSWAP[ipiv] {
            x2.swap(0, 1);
        }
        x[0][0] = x2[0];
        if n1 == 1 {
            x[0][1] = x2[1];
        } else {
            x[1][0] = x2[1];
        }
        return (x, scale);
    }
    // 2 × 2 by 2 × 2: the equivalent system of order 4.
    let mut smin = zero;
    for row in tr.iter().chain(tl.iter()) {
        for v in row {
            smin = smin.max(v.abs());
        }
    }
    let smin = (eps * smin).max(smlnum);
    let mut t16 = [[zero; 4]; 4];
    t16[0][0] = tl[0][0] + sgn * tr[0][0];
    t16[1][1] = tl[1][1] + sgn * tr[0][0];
    t16[2][2] = tl[0][0] + sgn * tr[1][1];
    t16[3][3] = tl[1][1] + sgn * tr[1][1];
    t16[0][1] = tl[0][1];
    t16[1][0] = tl[1][0];
    t16[2][3] = tl[0][1];
    t16[3][2] = tl[1][0];
    t16[0][2] = sgn * tr[1][0];
    t16[1][3] = sgn * tr[1][0];
    t16[2][0] = sgn * tr[0][1];
    t16[3][1] = sgn * tr[0][1];
    let mut btmp = [b[0][0], b[1][0], b[0][1], b[1][1]];
    let mut jpiv = [0usize; 4];
    for i in 0..3 {
        let mut xmax = zero;
        let (mut ipsv, mut jpsv) = (i, i);
        for (ip, row) in t16.iter().enumerate().skip(i) {
            for (jp, v) in row.iter().enumerate().skip(i) {
                if v.abs() >= xmax {
                    xmax = v.abs();
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }
        if ipsv != i {
            t16.swap(ipsv, i);
            btmp.swap(ipsv, i);
        }
        if jpsv != i {
            for row in &mut t16 {
                row.swap(jpsv, i);
            }
        }
        jpiv[i] = jpsv;
        if t16[i][i].abs() < smin {
            t16[i][i] = smin;
        }
        for j in i + 1..4 {
            t16[j][i] /= t16[i][i];
            btmp[j] -= t16[j][i] * btmp[i];
            for k in i + 1..4 {
                t16[j][k] -= t16[j][i] * t16[i][k];
            }
        }
    }
    if t16[3][3].abs() < smin {
        t16[3][3] = smin;
    }
    let mut scale = one;
    let eight = R::from_f64(8.0);
    if (0..4).any(|i| eight * smlnum * btmp[i].abs() > t16[i][i].abs()) {
        let big = btmp.iter().fold(zero, |m, v| m.max(v.abs()));
        scale = (one / eight) / big;
        for v in &mut btmp {
            *v *= scale;
        }
    }
    let mut tmp = [zero; 4];
    for k in (0..4).rev() {
        let temp = one / t16[k][k];
        tmp[k] = btmp[k] * temp;
        for j in k + 1..4 {
            tmp[k] -= (temp * t16[k][j]) * tmp[j];
        }
    }
    for k in (0..3).rev() {
        if jpiv[k] != k {
            tmp.swap(k, jpiv[k]);
        }
    }
    x[0][0] = tmp[0];
    x[1][0] = tmp[1];
    x[0][1] = tmp[2];
    x[1][1] = tmp[3];
    (x, scale)
}
