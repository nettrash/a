//! The singular value decomposition of a real upper bidiagonal matrix by divide and conquer, as
//! LAPACK's `bdsdc` and `lasd0`–`lasd3`: the matrix split in halves at a row, the halves solved,
//! and their decompositions merged through a broken arrow matrix whose singular vectors update
//! theirs in products of blocks.

use super::qr::{Columns, bidiagonal_qr};
use super::secular::arrow;
use crate::math::complex::Float;

/// `c = β·c + α·a·b` on real column-major blocks: `(m, n, k, α, a, lda, b, ldb, β, c, ldc)`.
pub(super) type Gemm<R> = fn(usize, usize, usize, R, &[R], usize, &[R], usize, R, &mut [R], usize);

/// Problems up to this order are solved by the implicit QR algorithm.
pub(super) const LEAF: usize = 25;

/// The decomposition `B = U·Σ·Vᵀ` of the `n × n` upper bidiagonal `B` with the diagonal `d` and
/// the superdiagonal `e`: `d` turned into the singular values, descending, and `u` and `v`, of
/// `n × n`, column-major, set to `U` and `V`. `false` when the implicit QR algorithm of a small
/// problem did not converge. The products of blocks are `mm`'s.
pub(super) fn bidiagonal_svd<R: Float>(
    d: &mut [R],
    e: &mut [R],
    u: &mut [R],
    v: &mut [R],
    mm: Gemm<R>,
) -> bool {
    let n = d.len();
    u.fill(R::ZERO);
    v.fill(R::ZERO);
    for j in 0..n {
        u[j + j * n] = R::ONE;
        v[j + j * n] = R::ONE;
    }
    if n <= LEAF {
        let mut e = e.to_vec();
        e.resize(n, R::ZERO);
        let mut cu = Columns {
            data: u,
            rows: n,
            ld: n,
        };
        let mut cv = Columns {
            data: v,
            rows: n,
            ld: n,
        };
        return bidiagonal_qr(d, &mut e, Some(&mut cu), Some(&mut cv));
    }
    // Scaled to the largest element, and split where the superdiagonal is negligible.
    let scale = d
        .iter()
        .chain(e.iter())
        .fold(R::ZERO, |s, &x| s.max(x.abs()));
    if scale == R::ZERO {
        return true;
    }
    for x in d.iter_mut().chain(e.iter_mut()) {
        *x /= scale;
    }
    let tiny = R::from_f64(0.9) * R::EPSILON * R::HALF;
    let mut ok = true;
    let mut order = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..n {
        if i + 1 < n && e[i].abs() >= tiny {
            continue;
        }
        if i + 1 < n {
            e[i] = R::ZERO;
        }
        let size = i + 1 - start;
        let (o, good) = solve(start, size, 0, d, e, u, v, n, mm);
        ok &= good;
        order.extend(o.into_iter().map(|j| start + j));
        start = i + 1;
    }
    for x in d.iter_mut() {
        *x *= scale;
    }
    // Descending, the columns of `U` and `V` with them.
    order.sort_by(|&a, &b| {
        d[b].partial_cmp(&d[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let (u0, v0, d0) = (u.to_vec(), v.to_vec(), d.to_vec());
    for (j, &o) in order.iter().enumerate() {
        d[j] = d0[o];
        u[j * n..(j + 1) * n].copy_from_slice(&u0[o * n..(o + 1) * n]);
        v[j * n..(j + 1) * n].copy_from_slice(&v0[o * n..(o + 1) * n]);
    }
    ok
}

/// Solves the `n × (n + sqre)` upper bidiagonal block from row and column `f`, with the diagonal
/// `d[f..f + n]` and the superdiagonal `e[f..f + n - 1 + sqre]`, into the diagonal blocks of `u`
/// (`n × n`) and `v` (`(n + sqre) × (n + sqre)`), which hold the identity there on entry: the
/// singular values into `d[f..f + n]`, and their order, ascending, returned with whether all
/// converged. With `sqre`, column `n` of the block of `v` is the null vector.
#[allow(clippy::too_many_arguments)]
fn solve<R: Float>(
    f: usize,
    n: usize,
    sqre: usize,
    d: &mut [R],
    e: &mut [R],
    u: &mut [R],
    v: &mut [R],
    ld: usize,
    mm: Gemm<R>,
) -> (Vec<usize>, bool) {
    if n <= LEAF {
        return leaf(f, n, sqre, d, e, u, v, ld);
    }
    let nl = n / 2;
    let nr = n - nl - 1;
    let (left, a) = solve(f, nl, 1, d, e, u, v, ld, mm);
    let (right, b) = solve(f + nl + 1, nr, sqre, d, e, u, v, ld, mm);
    let order = merge(f, nl, nr, sqre, d, e, u, v, ld, &left, &right, mm);
    (order, a && b)
}

/// [`solve`] by the implicit QR algorithm, a column past a square block first rotated away.
#[allow(clippy::too_many_arguments)]
fn leaf<R: Float>(
    f: usize,
    n: usize,
    sqre: usize,
    d: &mut [R],
    e: &mut [R],
    u: &mut [R],
    v: &mut [R],
    ld: usize,
) -> (Vec<usize>, bool) {
    let mut s = d[f..f + n].to_vec();
    let mut t = vec![R::ZERO; n];
    t[..n - 1].copy_from_slice(&e[f..f + n - 1]);
    let mut cv = Columns {
        data: &mut v[f + f * ld..],
        rows: n + sqre,
        ld,
    };
    if sqre == 1 {
        // Rotations from the right chase the element of the extra column up and out.
        let mut g = e[f + n - 1];
        for k in (0..n).rev() {
            let (r, cs, sn) = super::qr::rotg(s[k], g);
            s[k] = r;
            if k > 0 {
                g = -sn * t[k - 1];
                t[k - 1] *= cs;
            }
            cv.rotate(k, n, cs, sn);
        }
    }
    let mut cu = Columns {
        data: &mut u[f + f * ld..],
        rows: n,
        ld,
    };
    let ok = bidiagonal_qr(&mut s, &mut t, Some(&mut cu), Some(&mut cv));
    d[f..f + n].copy_from_slice(&s);
    ((0..n).rev().collect(), ok)
}

/// Where an entry's vectors come from: the left half, the right, or both after a rotation of
/// deflation mixed them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Rows of the left half only.
    Left,
    /// Rows of both halves.
    Both,
    /// Rows of the right half only.
    Right,
}

/// Merges the solved halves of the block of [`solve`], `nl` rows above row `f + nl` and `nr`
/// below, whose ascending orders are `left` and `right`, as LAPACK's `lasd1`: returns the order
/// of the block's singular values.
#[allow(clippy::too_many_arguments)]
fn merge<R: Float>(
    f: usize,
    nl: usize,
    nr: usize,
    sqre: usize,
    d: &mut [R],
    e: &[R],
    u: &mut [R],
    v: &mut [R],
    ld: usize,
    left: &[usize],
    right: &[usize],
    mm: Gemm<R>,
) -> Vec<usize> {
    let n = nl + nr + 1;
    let m = n + sqre;
    let mid = f + nl;
    // Scaled to the largest element.
    let mut scale = d[mid].abs().max(e[mid].abs());
    for j in (f..f + n).filter(|&j| j != mid) {
        scale = scale.max(d[j].abs());
    }
    if scale == R::ZERO {
        scale = R::ONE;
    }
    let (alpha, beta) = (d[mid] / scale, e[mid] / scale);
    // The entries of the arrow, ascending: the value, the element of `z`, the column of the
    // old vectors in the block, the side.
    let mut ent: Vec<(R, R, usize, Side)> = Vec::with_capacity(n - 1);
    {
        let (mut a, mut b) = (left.iter().peekable(), right.iter().peekable());
        let lval = |c: usize| d[f + c] / scale;
        let rval = |c: usize| d[mid + 1 + c] / scale;
        loop {
            let take_left = match (a.peek(), b.peek()) {
                (Some(&&x), Some(&&y)) => lval(x) <= rval(y),
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_left {
                let c = *a.next().unwrap_or(&0);
                ent.push((lval(c), alpha * v[mid + (f + c) * ld], c, Side::Left));
            } else {
                let c = *b.next().unwrap_or(&0);
                let col = nl + 1 + c;
                ent.push((
                    rval(c),
                    beta * v[mid + 1 + (f + col) * ld],
                    col,
                    Side::Right,
                ));
            }
        }
    }
    let za = alpha * v[mid + mid * ld];
    let zb = if sqre == 1 {
        beta * v[mid + 1 + (f + n) * ld]
    } else {
        R::ZERO
    };
    // Deflation, as LAPACK's `lasd2`: an entry whose `z` is negligible keeps its vectors, and of
    // two entries with values too close a rotation makes the first's `z` zero.
    let tol = R::from_f64(8.0) * R::EPSILON * R::HALF;
    let dmax = ent.last().map_or(R::ZERO, |x| x.0);
    let tol = tol * alpha.abs().max(beta.abs()).max(dmax);
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    let mut deflated: Vec<usize> = Vec::new();
    for j in 0..ent.len() {
        if ent[j].1.abs() <= tol {
            deflated.push(j);
            continue;
        }
        if let Some(&p) = kept.last()
            && (ent[j].0 - ent[p].0).abs() <= tol
        {
            let (zp, zj) = (ent[p].1, ent[j].1);
            let r = zj.hypot(zp);
            let (c, s) = (zj / r, -zp / r);
            ent[j].1 = r;
            ent[p].1 = R::ZERO;
            let (cp, cj) = (f + ent[p].2, f + ent[j].2);
            Columns {
                data: &mut u[f..],
                rows: n,
                ld,
            }
            .rotate(cp, cj, c, s);
            Columns {
                data: &mut v[f..],
                rows: m,
                ld,
            }
            .rotate(cp, cj, c, s);
            if ent[j].3 != ent[p].3 {
                ent[j].3 = Side::Both;
            }
            kept.pop();
            deflated.push(p);
        }
        kept.push(j);
    }
    deflated.sort_unstable();
    if let Some(&j) = kept.first()
        && ent[j].0.abs() <= tol * R::HALF
    {
        ent[j].0 = tol * R::HALF;
    }
    // The first column of the arrow: the two null vectors of the halves rotated together.
    let (z0, c, s) = if sqre == 1 {
        let r = za.hypot(zb);
        if r <= tol {
            (tol, R::ONE, R::ZERO)
        } else {
            (r, za / r, zb / r)
        }
    } else if za.abs() <= tol {
        (tol, R::ONE, R::ZERO)
    } else {
        (za, R::ONE, R::ZERO)
    };
    // The kept entries in the order of the products: the left ones, the first column, both, the
    // right ones, so that each half of the rows takes one block of columns.
    let k = kept.len() + 1;
    let count = |side: Side| kept.iter().filter(|&&j| ent[j].3 == side).count();
    let (nla, nb) = (count(Side::Left), count(Side::Both));
    let mut next = [0, nla + 1, nla + 1 + nb];
    let mut pos = vec![nla; k];
    for (p, &j) in pos[1..].iter_mut().zip(&kept) {
        let g = ent[j].3 as usize;
        *p = next[g];
        next[g] += 1;
    }
    let mut dk = vec![R::ZERO; k];
    let mut zk = vec![z0; k];
    for (i, &j) in kept.iter().enumerate() {
        dk[i + 1] = ent[j].0;
        zk[i + 1] = ent[j].1;
    }
    // The old vectors, gathered in that order.
    let mut uw = vec![R::ZERO; n * k];
    let mut vw = vec![R::ZERO; m * k];
    for (&p, &j) in pos[1..].iter().zip(&kept) {
        let col = f + ent[j].2;
        uw[p * n..(p + 1) * n].copy_from_slice(&u[f + col * ld..f + n + col * ld]);
        vw[p * m..(p + 1) * m].copy_from_slice(&v[f + col * ld..f + m + col * ld]);
    }
    uw[nl + nla * n] = R::ONE;
    let vnull: Vec<R> = (0..m)
        .map(|r| {
            let x = v[f + r + mid * ld];
            let y = if sqre == 1 {
                v[f + r + (f + n) * ld]
            } else {
                R::ZERO
            };
            vw[r + nla * m] = c * x + s * y;
            c * y - s * x
        })
        .collect();
    let defl: Vec<(R, Vec<R>, Vec<R>)> = deflated
        .iter()
        .map(|&j| {
            let col = f + ent[j].2;
            let uc = u[f + col * ld..f + n + col * ld].to_vec();
            let vc = v[f + col * ld..f + m + col * ld].to_vec();
            (ent[j].0, uc, vc)
        })
        .collect();
    // The arrow's decomposition, and the products that update the vectors.
    let mut sigma = vec![R::ZERO; k];
    let mut ua = vec![R::ZERO; k * k];
    let mut va = vec![R::ZERO; k * k];
    arrow(&dk, &zk, &pos, &mut sigma, &mut ua, &mut va);
    let (top, bottom) = (nla + 1 + nb, k - nla);
    let (one, zero) = (R::ONE, R::ZERO);
    let ub = &mut u[f + f * ld..];
    mm(nl, k, top, one, &uw, n, &ua, k, zero, ub, ld);
    for i in 0..k {
        ub[nl + i * ld] = ua[nla + i * k];
    }
    let ub = &mut u[mid + 1 + f * ld..];
    mm(
        nr,
        k,
        bottom,
        one,
        &uw[nl + 1 + nla * n..],
        n,
        &ua[nla..],
        k,
        zero,
        ub,
        ld,
    );
    let vb = &mut v[f + f * ld..];
    mm(nl + 1, k, top, one, &vw, m, &va, k, zero, vb, ld);
    let vb = &mut v[mid + 1 + f * ld..];
    let rows = nr + sqre;
    mm(
        rows,
        k,
        bottom,
        one,
        &vw[nl + 1 + nla * m..],
        m,
        &va[nla..],
        k,
        zero,
        vb,
        ld,
    );
    for (i, (_, uc, vc)) in defl.iter().enumerate() {
        let col = f + k + i;
        u[f + col * ld..f + n + col * ld].copy_from_slice(uc);
        v[f + col * ld..f + m + col * ld].copy_from_slice(vc);
    }
    if sqre == 1 {
        v[f + (f + n) * ld..f + m + (f + n) * ld].copy_from_slice(&vnull);
    }
    // The values, and their order: the arrow's ascending, merged with the deflated ones.
    for (i, &x) in sigma.iter().enumerate() {
        d[f + i] = x * scale;
    }
    for (i, (x, _, _)) in defl.iter().enumerate() {
        d[f + k + i] = *x * scale;
    }
    let mut order = Vec::with_capacity(n);
    let (mut a, mut b) = (0, k);
    while a < k || b < n {
        if b == n || (a < k && d[f + a] <= d[f + b]) {
            order.push(a);
            a += 1;
        } else {
            order.push(b);
            b += 1;
        }
    }
    order
}
