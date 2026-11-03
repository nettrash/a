//! The real symmetric tridiagonal eigenproblem the Hermitian one reduces to: the eigenvalues
//! alone by the square-root-free QL and QR algorithm, and the eigenvectors too by Cuppen's divide
//! and conquer, the rank-one updates' secular equations solved as LAPACK's `dlaed4`.

use super::Scalar;
use super::blas::{Op, gemm};
use crate::math::complex::Float;

/// Problems up to this order are left to the QL algorithm, as LAPACK's `SMLSIZ`.
const SMALL: usize = 25;

/// The iterations a root of a secular equation may take, as LAPACK's.
const MAX_ITERATIONS: usize = 30;

/// The unit roundoff, half the gap between 1 and the next float, LAPACK's `EPS`.
fn unit_roundoff<R: Float>() -> R {
    R::EPSILON * R::HALF
}

/// The eigenvalues of the symmetric tridiagonal matrix with the diagonal `d` and the
/// subdiagonal `e`, ascending, into `d`, by the QL and QR algorithm without square roots of
/// Pal, Walker and Kahan, as LAPACK's `dsterf`; `e` is destroyed. False if an eigenvalue takes
/// more than `30 · n` iterations.
pub(super) fn values<R: Float>(d: &mut [R], e: &mut [R]) -> bool {
    let n = d.len();
    let zero = R::ZERO;
    let eps = unit_roundoff::<R>();
    let eps2 = eps * eps;
    let safmin = R::MIN_POSITIVE;
    let ssfmax = (R::ONE / safmin).sqrt() / R::from_f64(3.0);
    let ssfmin = safmin.sqrt() / eps2;
    let max_iterations = 30 * n;
    let mut iterations = 0;
    let mut l1 = 0;
    while l1 < n {
        if l1 > 0 {
            e[l1 - 1] = zero;
        }
        // The block `start..=end`, split off where the subdiagonal is negligible.
        let mut m = l1;
        while m + 1 < n {
            if e[m].abs() <= d[m].abs().sqrt() * d[m + 1].abs().sqrt() * eps {
                e[m] = zero;
                break;
            }
            m += 1;
        }
        let (start, end) = (l1, m);
        l1 = m + 1;
        if end == start {
            continue;
        }
        // Scaled into range, with the squares of the subdiagonal.
        let mut norm = zero;
        for i in start..=end {
            norm = norm.max(d[i].abs());
            if i < end {
                norm = norm.max(e[i].abs());
            }
        }
        if norm == zero {
            continue;
        }
        let scale = if norm > ssfmax {
            Some(ssfmax)
        } else if norm < ssfmin {
            Some(ssfmin)
        } else {
            None
        };
        if let Some(s) = scale {
            let f = s / norm;
            for x in &mut d[start..=end] {
                *x *= f;
            }
            for x in &mut e[start..end] {
                *x *= f;
            }
        }
        for x in &mut e[start..end] {
            *x = *x * *x;
        }
        // QL, converging at the top, when the top is the smaller end, QR otherwise.
        let ok = if d[end].abs() >= d[start].abs() {
            values_ql(d, e, start, end, &mut iterations, max_iterations, eps2)
        } else {
            values_qr(d, e, start, end, &mut iterations, max_iterations, eps2)
        };
        if let Some(s) = scale {
            let f = norm / s;
            for x in &mut d[start..=end] {
                *x *= f;
            }
        }
        if !ok {
            return false;
        }
    }
    sort(d);
    true
}

/// The QL sweeps of [`values`] on the block `l..=end`, `e` holding squares, from the top.
fn values_ql<R: Float>(
    d: &mut [R],
    e: &mut [R],
    mut l: usize,
    end: usize,
    iterations: &mut usize,
    max_iterations: usize,
    eps2: R,
) -> bool {
    let zero = R::ZERO;
    loop {
        let mut m = l;
        while m < end && e[m].abs() > eps2 * (d[m] * d[m + 1]).abs() {
            m += 1;
        }
        if m < end {
            e[m] = zero;
        }
        let mut p = d[l];
        if m == l {
            // An eigenvalue.
            d[l] = p;
            l += 1;
            if l > end {
                return true;
            }
            continue;
        }
        if m == l + 1 {
            let (rt1, rt2) = eigenvalues_2x2(d[l], e[l].sqrt(), d[l + 1]);
            d[l] = rt1;
            d[l + 1] = rt2;
            e[l] = zero;
            l += 2;
            if l > end {
                return true;
            }
            continue;
        }
        if *iterations == max_iterations {
            return false;
        }
        *iterations += 1;
        // The shift.
        let rte = e[l].sqrt();
        let mut sigma = (d[l + 1] - p) / (R::TWO * rte);
        let r = sigma.hypot(R::ONE);
        sigma = p - rte / (sigma + r.copysign(sigma));
        let (mut c, mut s) = (R::ONE, zero);
        let mut gamma = d[m] - sigma;
        p = gamma * gamma;
        for i in (l..m).rev() {
            let bb = e[i];
            let r = p + bb;
            if i != m - 1 {
                e[i + 1] = s * r;
            }
            let oldc = c;
            c = p / r;
            s = bb / r;
            let oldgam = gamma;
            let alpha = d[i];
            gamma = c * (alpha - sigma) - s * oldgam;
            d[i + 1] = oldgam + (alpha - gamma);
            p = if c != zero {
                gamma * gamma / c
            } else {
                oldc * bb
            };
        }
        e[l] = s * p;
        d[l] = sigma + gamma;
    }
}

/// The QR sweeps of [`values`] on the block `start..=end`, `e` holding squares, from the bottom.
fn values_qr<R: Float>(
    d: &mut [R],
    e: &mut [R],
    start: usize,
    end: usize,
    iterations: &mut usize,
    max_iterations: usize,
    eps2: R,
) -> bool {
    let zero = R::ZERO;
    // `l` runs down from `end` to `start`.
    let lend = start;
    let mut l = end;
    loop {
        let mut m = l;
        while m > lend && e[m - 1].abs() > eps2 * (d[m] * d[m - 1]).abs() {
            m -= 1;
        }
        if m > lend {
            e[m - 1] = zero;
        }
        let mut p = d[l];
        if m == l {
            d[l] = p;
            if l == lend {
                return true;
            }
            l -= 1;
            continue;
        }
        if m + 1 == l {
            let (rt1, rt2) = eigenvalues_2x2(d[l], e[l - 1].sqrt(), d[l - 1]);
            d[l] = rt1;
            d[l - 1] = rt2;
            e[l - 1] = zero;
            if l < lend + 2 {
                return true;
            }
            l -= 2;
            continue;
        }
        if *iterations == max_iterations {
            return false;
        }
        *iterations += 1;
        let rte = e[l - 1].sqrt();
        let mut sigma = (d[l - 1] - p) / (R::TWO * rte);
        let r = sigma.hypot(R::ONE);
        sigma = p - rte / (sigma + r.copysign(sigma));
        let (mut c, mut s) = (R::ONE, zero);
        let mut gamma = d[m] - sigma;
        p = gamma * gamma;
        for i in m..l {
            let bb = e[i];
            let r = p + bb;
            if i != m {
                e[i - 1] = s * r;
            }
            let oldc = c;
            c = p / r;
            s = bb / r;
            let oldgam = gamma;
            let alpha = d[i + 1];
            gamma = c * (alpha - sigma) - s * oldgam;
            d[i] = oldgam + (alpha - gamma);
            p = if c != zero {
                gamma * gamma / c
            } else {
                oldc * bb
            };
        }
        e[l - 1] = s * p;
        d[l] = sigma + gamma;
    }
}

/// The eigenvalues of `[[a, b], [b, c]]`, the larger in absolute value first, as LAPACK's
/// `dlae2`.
fn eigenvalues_2x2<R: Float>(a: R, b: R, c: R) -> (R, R) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (R::ONE + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (R::ONE + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * R::TWO.sqrt()
    };
    if sm < R::ZERO {
        let rt1 = R::HALF * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else if sm > R::ZERO {
        let rt1 = R::HALF * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else {
        (R::HALF * rt, -R::HALF * rt)
    }
}

/// Sorts ascending, NaNs last.
fn sort<R: Float>(d: &mut [R]) {
    d.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| b.is_nan().cmp(&a.is_nan()))
    });
}

/// The eigenvalues, ascending, into `d`, and the eigenvectors, the columns of the returned
/// `n × n` matrix, of the symmetric tridiagonal matrix with the diagonal `d` and the
/// subdiagonal `e`, as LAPACK's `dstedc`: split where the subdiagonal is negligible, each block
/// scaled to 1 and solved by divide and conquer, its pieces of up to [`SMALL`] by the QL
/// algorithm. `e` is destroyed. `None` if an iteration fails to converge.
pub(super) fn eigen<S: Scalar>(d: &mut [S::Real], e: &mut [S::Real]) -> Option<Vec<S>> {
    let n = d.len();
    let mut q = vec![S::zero(); n * n];
    for j in 0..n {
        q[j + j * n] = S::one();
    }
    if n <= SMALL {
        return ql(d, e, Some((&mut q, n, n))).then_some(q);
    }
    let eps = unit_roundoff::<S::Real>();
    let mut start = 0;
    while start < n {
        let mut end = start;
        while end + 1 < n && e[end].abs() > eps * d[end].abs().sqrt() * d[end + 1].abs().sqrt() {
            end += 1;
        }
        let m = end - start + 1;
        let block = &mut q[start + start * n..];
        if m > SMALL {
            let mut norm = S::Real::ZERO;
            for i in start..=end {
                norm = norm.max(d[i].abs());
                if i < end {
                    norm = norm.max(e[i].abs());
                }
            }
            let r = S::Real::ONE / norm;
            for x in &mut d[start..=end] {
                *x *= r;
            }
            for x in &mut e[start..end] {
                *x *= r;
            }
            if !divide(&mut d[start..=end], &mut e[start..end], block, n) {
                return None;
            }
            for x in &mut d[start..=end] {
                *x *= norm;
            }
        } else if m > 1 && !ql(&mut d[start..=end], &e[start..end], Some((block, n, m))) {
            return None;
        }
        start = end + 1;
    }
    // The blocks' eigenvalues merged into one ascending order, the vectors along.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        d[i].partial_cmp(&d[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let sorted: Vec<S::Real> = order.iter().map(|&i| d[i]).collect();
    d.copy_from_slice(&sorted);
    permute_columns(&mut q, n, n, &order);
    Some(q)
}

/// Divide and conquer on the unreduced `m × m` problem `(d, e)`, its eigenvectors into the
/// identity block `q` (leading dimension `ldq`): the matrix is cut in two by a rank-one change,
/// the halves solved, then their eigensystems merged, as LAPACK's `dlaed0`.
fn divide<S: Scalar>(d: &mut [S::Real], e: &mut [S::Real], q: &mut [S], ldq: usize) -> bool {
    let m = d.len();
    if m <= SMALL {
        return ql(d, e, Some((q, ldq, m)));
    }
    let m1 = m / 2;
    let rho = e[m1 - 1];
    d[m1 - 1] -= rho.abs();
    d[m1] -= rho.abs();
    let (d1, d2) = d.split_at_mut(m1);
    let (e1, e2) = e.split_at_mut(m1 - 1);
    if !divide(d1, e1, q, ldq) || !divide(d2, &mut e2[1..], &mut q[m1 + m1 * ldq..], ldq) {
        return false;
    }
    merge(d, q, ldq, m1, rho)
}

/// Kinds of the columns of the merged eigenvectors, by where their nonzeros are.
const UPPER: usize = 0;
/// Nonzeros in both halves.
const DENSE: usize = 1;
/// Nonzeros in the lower half only.
const LOWER: usize = 2;
/// Deflated: the column is an eigenvector already.
const DEFLATED: usize = 3;

/// Merges the eigensystems of the two halves of an `n × n` problem cut after row `n1` by the
/// subdiagonal `rho`: `d` holds both halves' eigenvalues, each ascending, and `q` their
/// eigenvectors, block diagonal. The eigensystem of `diag(D₁, D₂) + ρ·z·zᵀ`, `z` from the last
/// row of `Q₁` and the first of `Q₂`, replaces them, ascending: deflated where a component of
/// `z` is negligible or two eigenvalues are close (LAPACK's `dlaed2`), the rest from the roots
/// of the secular equation, with `z` recomputed from them by Gu and Eisenstat's formula so that
/// the vectors are orthogonal, then multiplied into `q` (LAPACK's `dlaed3`).
fn merge<S: Scalar>(d: &mut [S::Real], q: &mut [S], ldq: usize, n1: usize, rho: S::Real) -> bool {
    let n = d.len();
    let n2 = n - n1;
    let zero = S::Real::ZERO;
    let mut z: Vec<S::Real> = (0..n1)
        .map(|j| q[n1 - 1 + j * ldq].re())
        .chain((n1..n).map(|j| q[n1 + j * ldq].re()))
        .collect();
    // `z` of norm 1 (each half's is a row of an orthogonal matrix) and `ρ > 0`.
    let r = S::Real::HALF.sqrt();
    for (j, x) in z.iter_mut().enumerate() {
        *x = if j >= n1 && rho < zero {
            -*x * r
        } else {
            *x * r
        };
    }
    let rho = (S::Real::TWO * rho).abs();
    // The order of both halves' eigenvalues together.
    let mut indx = Vec::with_capacity(n);
    let (mut i, mut j) = (0, n1);
    while i < n1 || j < n {
        if j == n || (i < n1 && d[i] <= d[j]) {
            indx.push(i);
            i += 1;
        } else {
            indx.push(j);
            j += 1;
        }
    }
    let zmax = z.iter().fold(zero, |m, x| m.max(x.abs()));
    let dmax = d.iter().fold(zero, |m, x| m.max(x.abs()));
    let tol = S::Real::from_f64(8.0) * unit_roundoff::<S::Real>() * dmax.max(zmax);
    if rho * zmax <= tol {
        // The change is negligible: only the order.
        let sorted: Vec<S::Real> = indx.iter().map(|&i| d[i]).collect();
        d.copy_from_slice(&sorted);
        permute_columns(q, ldq, n, &indx);
        return true;
    }
    // Deflation: columns whose `z` is negligible, and pairs of close eigenvalues, rotated so that
    // one of the pair's `z` vanishes. The rest, `k` of them, go on in `dlamda` and `w`, in the
    // order of `indxp`; the deflated ones are at the end of `indxp`, descending.
    let mut kind: Vec<usize> = (0..n).map(|j| if j < n1 { UPPER } else { LOWER }).collect();
    let mut indxp = vec![0; n];
    let mut dlamda = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut k2 = n;
    let mut pending: Option<usize> = None;
    for &nj in &indx {
        if rho * z[nj].abs() <= tol {
            k2 -= 1;
            kind[nj] = DEFLATED;
            indxp[k2] = nj;
            continue;
        }
        let Some(pj) = pending else {
            pending = Some(nj);
            continue;
        };
        let tau = z[nj].hypot(z[pj]);
        let t = d[nj] - d[pj];
        let (c, s) = (z[nj] / tau, -z[pj] / tau);
        if (t * c * s).abs() <= tol {
            // Close enough: rotate `z[pj]` to zero.
            z[nj] = tau;
            z[pj] = zero;
            if kind[nj] != kind[pj] {
                kind[nj] = DENSE;
            }
            kind[pj] = DEFLATED;
            rotate_columns(q, ldq, n, pj, nj, c, s);
            let t = d[pj] * c * c + d[nj] * s * s;
            d[nj] = d[pj] * s * s + d[nj] * c * c;
            d[pj] = t;
            // Into the deflated ones, keeping them descending.
            k2 -= 1;
            let mut i = k2 + 1;
            while i < n && d[pj] < d[indxp[i]] {
                indxp[i - 1] = indxp[i];
                i += 1;
            }
            indxp[i - 1] = pj;
        } else {
            indxp[dlamda.len()] = pj;
            dlamda.push(d[pj]);
            w.push(z[pj]);
        }
        pending = Some(nj);
    }
    if let Some(pj) = pending {
        indxp[dlamda.len()] = pj;
        dlamda.push(d[pj]);
        w.push(z[pj]);
    }
    let k = dlamda.len();
    // The columns grouped by kind, upper, dense, lower, deflated, packed into `q2` without
    // their zero halves: `grouped[g]` is the column, `indxc[g]` its place in `indxp`.
    let mut count = [0; 4];
    for &c in &kind {
        count[c] += 1;
    }
    debug_assert_eq!(k, n - count[DEFLATED]);
    let mut next = [
        0,
        count[0],
        count[0] + count[1],
        count[0] + count[1] + count[2],
    ];
    let mut grouped = vec![0; n];
    let mut indxc = vec![0; n];
    for (j, &js) in indxp.iter().enumerate() {
        let c = kind[js];
        grouped[next[c]] = js;
        indxc[next[c]] = j;
        next[c] += 1;
    }
    let n12 = count[UPPER] + count[DENSE];
    let n23 = count[DENSE] + count[LOWER];
    let lower_at = n1 * n12;
    let deflated_at = lower_at + n2 * n23;
    let mut q2 = vec![S::zero(); deflated_at + n * count[DEFLATED]];
    let mut values = vec![zero; n];
    for (g, &js) in grouped.iter().enumerate() {
        let col = &q[js * ldq..js * ldq + n];
        if g < n12 {
            q2[g * n1..(g + 1) * n1].copy_from_slice(&col[..n1]);
        }
        if g >= count[UPPER] && g < k {
            let l = g - count[UPPER];
            q2[lower_at + l * n2..lower_at + (l + 1) * n2].copy_from_slice(&col[n1..]);
        }
        if g >= k {
            let l = g - k;
            q2[deflated_at + l * n..deflated_at + (l + 1) * n].copy_from_slice(col);
        }
        values[g] = d[js];
    }
    // The deflated eigensystem goes back at the end.
    for l in 0..n - k {
        q[(k + l) * ldq..(k + l) * ldq + n]
            .copy_from_slice(&q2[deflated_at + l * n..deflated_at + (l + 1) * n]);
        d[k + l] = values[k + l];
    }
    if k > 0 {
        // The roots of the secular equation, each with its `dlamda - λ` in a column of `u`.
        let mut u = vec![zero; k * k];
        for j in 0..k {
            match secular(j, &dlamda, &w, &mut u[j * k..(j + 1) * k], rho) {
                Some(lambda) => d[j] = lambda,
                None => return false,
            }
        }
        let mut v = vec![S::zero(); k * k];
        if k <= 2 {
            // The roots came with their normalized eigenvectors.
            for j in 0..k {
                for g in 0..k {
                    v[g + j * k] = S::from_real(u[indxc[g] + j * k]);
                }
            }
        } else {
            // `z` again from the roots, by Löwner's formula, then the eigenvectors
            // `(dlamda - λ_j)⁻¹·z`, normalized.
            let sign = w.clone();
            for i in 0..k {
                w[i] = u[i + i * k];
            }
            for j in 0..k {
                let (col, dj) = (&u[j * k..(j + 1) * k], dlamda[j]);
                let (left, right) = w.split_at_mut(j);
                for ((w, &u), &dl) in left.iter_mut().zip(col).zip(&dlamda[..j]) {
                    *w *= u / (dl - dj);
                }
                let (col, dl) = (&col[j + 1..], &dlamda[j + 1..]);
                for ((w, &u), &dl) in right[1..].iter_mut().zip(col).zip(dl) {
                    *w *= u / (dl - dj);
                }
            }
            for (w, &s) in w.iter_mut().zip(&sign) {
                *w = (-*w).sqrt().copysign(s);
            }
            let mut s = vec![zero; k];
            for j in 0..k {
                for ((s, &w), &u) in s.iter_mut().zip(&w).zip(&u[j * k..(j + 1) * k]) {
                    *s = w / u;
                }
                let r = S::Real::ONE / norm(&s);
                for (x, &g) in v[j * k..(j + 1) * k].iter_mut().zip(&indxc) {
                    *x = S::from_real(s[g] * r);
                }
            }
        }
        // The eigenvectors: the upper rows from the upper and dense columns, the lower rows
        // from the dense and lower ones.
        let (one, nil) = (S::one(), S::zero());
        if n23 > 0 {
            let vl = &v[count[UPPER]..];
            gemm(
                n2,
                k,
                n23,
                one,
                &q2[lower_at..],
                n2,
                Op::N,
                vl,
                k,
                Op::N,
                nil,
                &mut q[n1..],
                ldq,
            );
        } else {
            for j in 0..k {
                q[j * ldq + n1..j * ldq + n].fill(nil);
            }
        }
        if n12 > 0 {
            gemm(n1, k, n12, one, &q2, n1, Op::N, &v, k, Op::N, nil, q, ldq);
        } else {
            for j in 0..k {
                q[j * ldq..j * ldq + n1].fill(nil);
            }
        }
    }
    // The roots ascending, the deflated eigenvalues descending: merged.
    let mut order = Vec::with_capacity(n);
    let (mut i, mut j) = (0, n);
    while i < k || j > k {
        if j == k || (i < k && d[i] <= d[j - 1]) {
            order.push(i);
            i += 1;
        } else {
            j -= 1;
            order.push(j);
        }
    }
    let sorted: Vec<S::Real> = order.iter().map(|&i| d[i]).collect();
    d.copy_from_slice(&sorted);
    permute_columns(q, ldq, n, &order);
    true
}

/// The 2-norm, in lanes, of a vector whose squares neither overflow nor underflow, as those of
/// the merged eigenvectors of a problem scaled to 1.
fn norm<R: Float>(x: &[R]) -> R {
    let mut s = [R::ZERO; 4];
    let (quads, rest) = x.as_chunks::<4>();
    for c in quads {
        for l in 0..4 {
            s[l] += c[l] * c[l];
        }
    }
    for &v in rest {
        s[0] += v * v;
    }
    ((s[0] + s[2]) + (s[1] + s[3])).sqrt()
}

/// `(x, y) = (c·x + s·y, c·y - s·x)` for the columns `x` and `y` of the `rows`-row block `q`.
fn rotate_columns<S: Scalar>(
    q: &mut [S],
    ldq: usize,
    rows: usize,
    x: usize,
    y: usize,
    c: S::Real,
    s: S::Real,
) {
    for r in 0..rows {
        let (a, b) = (q[r + x * ldq], q[r + y * ldq]);
        q[r + x * ldq] = a.scale(c) + b.scale(s);
        q[r + y * ldq] = b.scale(c) - a.scale(s);
    }
}

/// Reorders the columns of the `rows`-row block `q`: column `j` becomes the old column
/// `order[j]`, cycle by cycle.
fn permute_columns<S: Scalar>(q: &mut [S], ldq: usize, rows: usize, order: &[usize]) {
    let mut done = vec![false; order.len()];
    let mut saved = vec![S::zero(); rows];
    for start in 0..order.len() {
        if done[start] || order[start] == start {
            continue;
        }
        saved.copy_from_slice(&q[start * ldq..start * ldq + rows]);
        let mut j = start;
        loop {
            done[j] = true;
            let from = order[j];
            if from == start {
                q[j * ldq..j * ldq + rows].copy_from_slice(&saved);
                break;
            }
            q.copy_within(from * ldq..from * ldq + rows, j * ldq);
            j = from;
        }
    }
}

/// `δ_j = (d_j - base) - offset`, the differences from a pole first so that they are exact.
fn shift<R: Float>(delta: &mut [R], d: &[R], base: R, offset: R) {
    for (x, &dj) in delta.iter_mut().zip(d) {
        *x = (dj - base) - offset;
    }
}

/// After `δ_j -= η`, the sums over the poles of `p_j = z_j²/δ_j`, of `(z_j/δ_j)²` and of
/// `p_j·w_j` with the weights `w_j = w₀ - j` (`w₀ > 0`, counting down) or `j + 1` (`w₀ < 0`):
/// the last, LAPACK's sum of the partial sums of the first in the order it adds them, is its
/// rounding error bound, which this computes in lanes instead of in order.
fn sums<R: Float>(delta: &mut [R], z: &[R], eta: R, w0: R) -> (R, R, R) {
    let zero = R::ZERO;
    let (mut s, mut ds, mut es) = ([zero; 4], [zero; 4], [zero; 4]);
    let one = R::ONE;
    let (start, step) = if w0 > zero { (w0, -one) } else { (one, one) };
    let mut w = [
        start,
        start + step,
        start + step + step,
        start + step + step + step,
    ];
    let step4 = step * R::from_f64(4.0);
    let (head, tail) = delta.as_chunks_mut::<4>();
    let (zh, zt) = z.as_chunks::<4>();
    for (dv, zv) in head.iter_mut().zip(zh) {
        for l in 0..4 {
            dv[l] -= eta;
            let t = zv[l] / dv[l];
            let p = zv[l] * t;
            s[l] += p;
            ds[l] += t * t;
            es[l] += p * w[l];
            w[l] += step4;
        }
    }
    for (dv, &zv) in tail.iter_mut().zip(zt) {
        *dv -= eta;
        let t = zv / *dv;
        let p = zv * t;
        s[0] += p;
        ds[0] += t * t;
        es[0] += p * w[0];
        w[0] += step;
    }
    let sum = |v: [R; 4]| (v[0] + v[2]) + (v[1] + v[3]);
    (sum(s), sum(ds), sum(es))
}

/// The `i`-th smallest root `λ` of the secular equation `1/ρ + Σ z_j²/(d_j - λ) = 0`, for
/// ascending `d`, `ρ > 0` and `z` of norm 1, as LAPACK's `dlaed4`: `delta` gets `d_j - λ`,
/// computed from the nearest pole so that it is accurate, or for two poles the normalized
/// eigenvector. Each step interpolates the function by rational functions of the poles near the
/// root (two, or three by LAPACK's `dlaed6`), safeguarded by a bracket. `None` if it fails to
/// converge.
fn secular<R: Float>(i: usize, d: &[R], z: &[R], delta: &mut [R], rho: R) -> Option<R> {
    let n = d.len();
    if n == 1 {
        delta[0] = R::ONE;
        return Some(d[0] + rho * z[0] * z[0]);
    }
    if n == 2 {
        return Some(secular2(i, d, z, delta, rho));
    }
    if i == n - 1 {
        secular_last(d, z, delta, rho)
    } else {
        secular_inner(i, d, z, delta, rho)
    }
}

/// [`secular`] of the largest root, beyond the last pole.
fn secular_last<R: Float>(d: &[R], z: &[R], delta: &mut [R], rho: R) -> Option<R> {
    let n = d.len();
    let (zero, two, four) = (R::ZERO, R::TWO, R::from_f64(4.0));
    let eps = unit_roundoff::<R>();
    let rhoinv = R::ONE / rho;
    let i = n - 1;
    let ii = n - 2;
    // The first guess, from the two last poles.
    let midpt = rho / two;
    shift(delta, d, d[i], midpt);
    let (psi, _, _) = sums(&mut delta[..n - 2], &z[..n - 2], zero, zero);
    let c = rhoinv + psi;
    let w = c + z[ii] * z[ii] / delta[ii] + z[i] * z[i] / delta[i];
    let guess = |c: R| {
        let del = d[i] - d[ii];
        let a = -c * del + z[ii] * z[ii] + z[i] * z[i];
        let b = z[i] * z[i] * del;
        if a < zero {
            two * b / ((a * a + four * b * c).sqrt() - a)
        } else {
            (a + (a * a + four * b * c).sqrt()) / (two * c)
        }
    };
    let (mut tau, mut lower, mut upper);
    if w <= zero {
        let temp = z[ii] * z[ii] / (d[i] - d[ii] + rho) + z[i] * z[i] / rho;
        tau = if c <= temp { rho } else { guess(c) };
        // `d_n + ρ/2 ≤ λ_n < d_n + τ ≤ d_n + ρ`.
        lower = midpt;
        upper = rho;
    } else {
        tau = guess(c);
        lower = zero;
        upper = midpt;
    }
    shift(delta, d, d[i], tau);
    // The function, its derivatives on both sides of the last pole and a bound on the rounding
    // error, after `δ -= η`.
    let evaluate = |delta: &mut [R], eta: R, tau: R| {
        let (psi, dpsi, erretm) = sums(
            &mut delta[..=ii],
            &z[..=ii],
            eta,
            R::from_f64((ii + 1) as f64),
        );
        delta[i] -= eta;
        let t = z[i] / delta[i];
        let phi = z[i] * t;
        let dphi = t * t;
        let erretm = R::from_f64(8.0) * (-phi - psi) + erretm.abs() - phi
            + rhoinv
            + tau.abs() * (dpsi + dphi);
        (rhoinv + phi + psi, dpsi, dphi, erretm)
    };
    let (mut w, mut dpsi, mut dphi, mut erretm) = evaluate(delta, zero, tau);
    for iteration in 1..=MAX_ITERATIONS {
        if w.abs() <= eps * erretm {
            return Some(d[i] + tau);
        }
        if w <= zero {
            lower = lower.max(tau);
        } else {
            upper = upper.min(tau);
        }
        // The step by the rational interpolation of the two last poles.
        let mut c = w - delta[ii] * dpsi - delta[i] * dphi;
        let a = (delta[ii] + delta[i]) * w - delta[ii] * delta[i] * (dpsi + dphi);
        let b = delta[ii] * delta[i] * w;
        if iteration == 1 && c < zero {
            c = c.abs();
        }
        let newton = -w / (dpsi + dphi);
        let mut eta = if c == zero {
            newton
        } else if a >= zero {
            (a + (a * a - four * b * c).abs().sqrt()) / (two * c)
        } else {
            two * b / (a - (a * a - four * b * c).abs().sqrt())
        };
        // `η` must have the sign of `-w`; then it must stay in the bracket.
        if w * eta > zero || (w * eta).is_nan() {
            eta = newton;
        }
        let t = tau + eta;
        if !(t <= upper && t >= lower) {
            eta = if w < zero {
                (upper - tau) / two
            } else {
                (lower - tau) / two
            };
        }
        tau += eta;
        (w, dpsi, dphi, erretm) = evaluate(delta, eta, tau);
    }
    None
}

/// [`secular`] of a root between the poles `d_i` and `d_{i+1}`.
fn secular_inner<R: Float>(i: usize, d: &[R], z: &[R], delta: &mut [R], rho: R) -> Option<R> {
    let n = d.len();
    let (zero, two, four) = (R::ZERO, R::TWO, R::from_f64(4.0));
    let eps = unit_roundoff::<R>();
    let rhoinv = R::ONE / rho;
    let ip1 = i + 1;
    // The first guess, from the two nearest poles; the origin is the nearer one.
    let del = d[ip1] - d[i];
    let midpt = del / two;
    shift(delta, d, d[i], midpt);
    let (psi, _, _) = sums(&mut delta[..i], &z[..i], zero, zero);
    let (phi, _, _) = sums(&mut delta[i + 2..], &z[i + 2..], zero, zero);
    let c = rhoinv + psi + phi;
    let w = c + z[i] * z[i] / delta[i] + z[ip1] * z[ip1] / delta[ip1];
    let orgati = w > zero;
    let (mut tau, mut lower, mut upper);
    if orgati {
        // `d_i < λ < (d_i + d_{i+1})/2`, from `d_i`.
        let a = c * del + z[i] * z[i] + z[ip1] * z[ip1];
        let b = z[i] * z[i] * del;
        tau = if a > zero {
            two * b / (a + (a * a - four * b * c).abs().sqrt())
        } else {
            (a - (a * a - four * b * c).abs().sqrt()) / (two * c)
        };
        lower = zero;
        upper = midpt;
    } else {
        // `(d_i + d_{i+1})/2 ≤ λ < d_{i+1}`, from `d_{i+1}`.
        let a = c * del - z[i] * z[i] - z[ip1] * z[ip1];
        let b = z[ip1] * z[ip1] * del;
        tau = if a < zero {
            two * b / (a - (a * a + four * b * c).abs().sqrt())
        } else {
            -(a + (a * a + four * b * c).abs().sqrt()) / (two * c)
        };
        lower = -midpt;
        upper = zero;
    }
    let origin = if orgati { d[i] } else { d[ip1] };
    shift(delta, d, origin, tau);
    let ii = if orgati { i } else { ip1 };
    // After `δ -= η`, the function without its `ii` term, split at the pole `ii`, with the
    // derivatives and a bound on the rounding error.
    let evaluate = |delta: &mut [R], eta: R| {
        let (left, right) = delta.split_at_mut(ii);
        let (psi, dpsi, epsi) = sums(left, &z[..ii], eta, R::from_f64(ii as f64));
        let (phi, dphi, ephi) = sums(&mut right[1..], &z[ii + 1..], eta, -R::ONE);
        right[0] -= eta;
        let t = z[ii] / right[0];
        let dw = dpsi + dphi + t * t;
        let t = z[ii] * t;
        (psi, dpsi, phi, dphi, t, dw, epsi.abs() + ephi)
    };
    let (mut psi, mut dpsi, mut phi, mut dphi, temp, mut dw, mut erretm) = evaluate(delta, zero);
    let w0 = rhoinv + phi + psi;
    // Three poles when the root is on the far side of `ii` from its neighbour pole.
    let swtch3 = (if orgati { w0 < zero } else { w0 > zero }) && ii != 0 && ii != n - 1;
    let mut w = w0 + temp;
    let error = |erretm: R, phi: R, psi: R, temp: R, tau: R, dw: R| {
        R::from_f64(8.0) * (phi - psi)
            + erretm
            + two * rhoinv
            + R::from_f64(3.0) * temp.abs()
            + tau.abs() * dw
    };
    erretm = error(erretm, phi, psi, temp, tau, dw);
    let zii2 = z[ii] * z[ii];
    let mut swtch = false;
    for iteration in 1..=MAX_ITERATIONS {
        if w.abs() <= eps * erretm {
            return Some(origin + tau);
        }
        if w <= zero {
            lower = lower.max(tau);
        } else {
            upper = upper.min(tau);
        }
        let mut eta;
        if !swtch3 {
            let c = if swtch {
                let t = z[ii] / delta[ii];
                if orgati {
                    dpsi += t * t;
                } else {
                    dphi += t * t;
                }
                w - delta[i] * dpsi - delta[ip1] * dphi
            } else if orgati {
                let t = z[i] / delta[i];
                w - delta[ip1] * dw - (d[i] - d[ip1]) * t * t
            } else {
                let t = z[ip1] / delta[ip1];
                w - delta[i] * dw - (d[ip1] - d[i]) * t * t
            };
            let mut a = (delta[i] + delta[ip1]) * w - delta[i] * delta[ip1] * dw;
            let b = delta[i] * delta[ip1] * w;
            eta = if c == zero {
                if a == zero {
                    a = if swtch {
                        delta[i] * delta[i] * dpsi + delta[ip1] * delta[ip1] * dphi
                    } else if orgati {
                        z[i] * z[i] + delta[ip1] * delta[ip1] * (dpsi + dphi)
                    } else {
                        z[ip1] * z[ip1] + delta[i] * delta[i] * (dpsi + dphi)
                    };
                }
                b / a
            } else if a <= zero {
                (a - (a * a - four * b * c).abs().sqrt()) / (two * c)
            } else {
                two * b / (a + (a * a - four * b * c).abs().sqrt())
            };
        } else {
            // Rational interpolation of the three poles around the root.
            let (iim1, iip1) = (ii - 1, ii + 1);
            let temp = rhoinv + psi + phi;
            let (c, z1, z3);
            if swtch {
                c = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
                z1 = delta[iim1] * delta[iim1] * dpsi;
                z3 = delta[iip1] * delta[iip1] * dphi;
            } else if orgati {
                let t = z[iim1] / delta[iim1];
                let t = t * t;
                c = temp - delta[iip1] * (dpsi + dphi) - (d[iim1] - d[iip1]) * t;
                z1 = z[iim1] * z[iim1];
                z3 = delta[iip1] * delta[iip1] * ((dpsi - t) + dphi);
            } else {
                let t = z[iip1] / delta[iip1];
                let t = t * t;
                c = temp - delta[iim1] * (dpsi + dphi) - (d[iip1] - d[iim1]) * t;
                z1 = delta[iim1] * delta[iim1] * (dpsi + (dphi - t));
                z3 = z[iip1] * z[iip1];
            }
            let poles = [delta[iim1], delta[ii], delta[iip1]];
            eta = rational3(iteration + 1, orgati, c, &poles, &[z1, zii2, z3], w)?;
        }
        // `η` must have the sign of `-w`; then it must stay in the bracket.
        if w * eta >= zero || (w * eta).is_nan() {
            eta = -w / dw;
        }
        let t = tau + eta;
        if !(t <= upper && t >= lower) {
            eta = if w < zero {
                (upper - tau) / two
            } else {
                (lower - tau) / two
            };
        }
        tau += eta;
        let last = w;
        let temp;
        (psi, dpsi, phi, dphi, temp, dw, erretm) = evaluate(delta, eta);
        w = rhoinv + phi + psi + temp;
        erretm = error(erretm, phi, psi, temp, tau, dw);
        let tenth = last.abs() / R::from_f64(10.0);
        if iteration == 1 {
            swtch = if orgati { -w > tenth } else { w > tenth };
        } else if w * last > zero && w.abs() > tenth {
            swtch = !swtch;
        }
    }
    None
}

/// [`secular`] of two poles, as LAPACK's `dlaed5`: `delta` gets the normalized eigenvector.
fn secular2<R: Float>(i: usize, d: &[R], z: &[R], delta: &mut [R], rho: R) -> R {
    let (two, four) = (R::TWO, R::from_f64(4.0));
    let del = d[1] - d[0];
    let (z0, z1) = (z[0] * z[0], z[1] * z[1]);
    let lambda;
    if i == 0 {
        let w = R::ONE + two * rho * (z1 - z0) / del;
        if w > R::ZERO {
            let b = del + rho * (z0 + z1);
            let c = rho * z0 * del;
            // `b > 0` always.
            let tau = two * c / (b + (b * b - four * c).abs().sqrt());
            lambda = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        } else {
            let b = -del + rho * (z0 + z1);
            let c = rho * z1 * del;
            let tau = if b > R::ZERO {
                -two * c / (b + (b * b + four * c).sqrt())
            } else {
                (b - (b * b + four * c).sqrt()) / two
            };
            lambda = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }
    } else {
        let b = -del + rho * (z0 + z1);
        let c = rho * z1 * del;
        let tau = if b > R::ZERO {
            (b + (b * b + four * c).sqrt()) / two
        } else {
            two * c / (-b + (b * b + four * c).sqrt())
        };
        lambda = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
    }
    let t = delta[0].hypot(delta[1]);
    delta[0] /= t;
    delta[1] /= t;
    lambda
}

/// The step `τ` to the root of `ρ + Σ z_j/(d_j - x)` over three poles nearest to 0, from
/// `f(0) = finit`, by Gragg, Thornton and Warner's cubically convergent iteration, as LAPACK's
/// `dlaed6`; `orgati` tells whether the root is between `d₂` and `d₃` (or `d₁` and `d₂`).
fn rational3<R: Float>(
    kniter: usize,
    orgati: bool,
    rho: R,
    d: &[R; 3],
    z: &[R; 3],
    finit: R,
) -> Option<R> {
    let (zero, two, four) = (R::ZERO, R::TWO, R::from_f64(4.0));
    let eps = unit_roundoff::<R>();
    let (mut lbd, mut ubd) = if orgati { (d[1], d[2]) } else { (d[0], d[1]) };
    if finit < zero {
        lbd = zero;
    } else {
        ubd = zero;
    }
    let mut tau = zero;
    let solve = |a: R, b: R, c: R| {
        let m = a.abs().max(b.abs()).max(c.abs());
        let (a, b, c) = (a / m, b / m, c / m);
        if c == zero {
            b / a
        } else if a <= zero {
            (a - (a * a - four * b * c).abs().sqrt()) / (two * c)
        } else {
            two * b / (a + (a * a - four * b * c).abs().sqrt())
        }
    };
    if kniter == 2 {
        let (a, b, c);
        if orgati {
            let t = (d[2] - d[1]) / two;
            c = rho + z[0] / ((d[0] - d[1]) - t);
            a = c * (d[1] + d[2]) + z[1] + z[2];
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
        } else {
            let t = (d[0] - d[1]) / two;
            c = rho + z[2] / ((d[2] - d[1]) - t);
            a = c * (d[0] + d[1]) + z[0] + z[1];
            b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
        }
        tau = solve(a, b, c);
        if !(tau >= lbd && tau <= ubd) {
            tau = (lbd + ubd) / two;
        }
        if d[0] == tau || d[1] == tau || d[2] == tau {
            tau = zero;
        } else {
            let f = finit
                + tau * z[0] / (d[0] * (d[0] - tau))
                + tau * z[1] / (d[1] * (d[1] - tau))
                + tau * z[2] / (d[2] * (d[2] - tau));
            if f <= zero {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if finit.abs() <= f.abs() {
                tau = zero;
            }
        }
    }
    // Scaled up when a pole is so near that `1/(d - τ)³` would overflow.
    let exponent = (R::MIN_POSITIVE.to_f64().log2() / 3.0).trunc() as i32;
    let small1 = two.powi(exponent);
    let sminv1 = R::ONE / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;
    let near = if orgati {
        (d[1] - tau).abs().min((d[2] - tau).abs())
    } else {
        (d[0] - tau).abs().min((d[1] - tau).abs())
    };
    let (mut ds, mut zs) = (*d, *z);
    let mut unscale = R::ONE;
    if near <= small1 {
        let (f, inv) = if near <= small2 {
            (sminv2, small2)
        } else {
            (sminv1, small1)
        };
        for j in 0..3 {
            ds[j] *= f;
            zs[j] *= f;
        }
        tau *= f;
        lbd *= f;
        ubd *= f;
        unscale = inv;
    }
    let (mut fc, mut df, mut ddf) = (zero, zero, zero);
    for j in 0..3 {
        let t = R::ONE / (ds[j] - tau);
        let t1 = zs[j] * t;
        let t2 = t1 * t;
        fc += t1 / ds[j];
        df += t2;
        ddf += t2 * t;
    }
    let mut f = finit + tau * fc;
    if f.abs() <= zero {
        return Some(tau * unscale);
    }
    if f <= zero {
        lbd = tau;
    } else {
        ubd = tau;
    }
    for _ in 2..=40 {
        let (t1, t2) = if orgati {
            (ds[1] - tau, ds[2] - tau)
        } else {
            (ds[0] - tau, ds[1] - tau)
        };
        let a = (t1 + t2) * f - t1 * t2 * df;
        let b = t1 * t2 * f;
        let c = f - (t1 + t2) * df + t1 * t2 * ddf;
        let mut eta = solve(a, b, c);
        if f * eta >= zero || (f * eta).is_nan() {
            eta = -f / df;
        }
        tau += eta;
        if !(tau >= lbd && tau <= ubd) {
            tau = (lbd + ubd) / two;
        }
        let (mut erretm, mut pole) = (zero, false);
        fc = zero;
        df = zero;
        ddf = zero;
        for j in 0..3 {
            if ds[j] - tau == zero {
                pole = true;
                break;
            }
            let t = R::ONE / (ds[j] - tau);
            let t1 = zs[j] * t;
            let t2 = t1 * t;
            let t4 = t1 / ds[j];
            fc += t4;
            erretm += t4.abs();
            df += t2;
            ddf += t2 * t;
        }
        if pole {
            return Some(tau * unscale);
        }
        f = finit + tau * fc;
        let erretm = R::from_f64(8.0) * (finit.abs() + tau.abs() * erretm) + tau.abs() * df;
        if f.abs() <= four * eps * erretm || (ubd - lbd) <= four * eps * tau.abs() {
            return Some(tau * unscale);
        }
        if f <= zero {
            lbd = tau;
        } else {
            ubd = tau;
        }
    }
    None
}

/// The eigenvalues of the symmetric tridiagonal matrix with the diagonal `d` and the
/// subdiagonal `e`, ascending, into `d`, by the implicit QL algorithm with Wilkinson's shifts,
/// EISPACK's `tql2`; its rotations are applied to the columns of the block `q` of `rows` rows
/// and the leading dimension `ldq`, when given, which turns the identity into the eigenvectors.
/// False if an eigenvalue takes more than `30 · max(n, 10)` iterations.
///
/// Each step of a sweep waits on the one before through a square root and a division, which on
/// small problems costs more than the rotations of the vectors, so the step is kept short: the
/// length of `(p, e_i)` from the sum of the squares where it can neither overflow nor underflow
/// enough to matter, `hypot` only elsewhere, and the next `p` divided by it directly, in
/// parallel with the rotation's cosine and sine.
pub(super) fn ql<S: Scalar>(
    d: &mut [S::Real],
    sub: &[S::Real],
    mut q: Option<(&mut [S], usize, usize)>,
) -> bool {
    let n = d.len();
    if n == 0 {
        return true;
    }
    let zero = S::Real::ZERO;
    let one = S::Real::ONE;
    // The subdiagonal with a zero after it, on the stack for the orders that are common here:
    // the leaves of divide and conquer and the small matrices of the Hermitian eigenproblem.
    let mut stack = [zero; 160];
    let mut heap = Vec::new();
    let e: &mut [S::Real] = if n <= stack.len() {
        &mut stack[..n]
    } else {
        heap.resize(n, zero);
        &mut heap
    };
    e[..n - 1].copy_from_slice(&sub[..n - 1]);
    let eps = S::Real::EPSILON;
    let mut f = zero;
    let mut tst1 = zero;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > eps * tst1 {
            m += 1;
        }
        if m > l {
            let mut its = 0;
            loop {
                its += 1;
                if its > 30 * n.max(10) {
                    return false;
                }
                // The shift from the leading 2 × 2 block.
                let g = d[l];
                let p = (d[l + 1] - g) / (S::Real::TWO * e[l]);
                let r = length(p, one);
                let r = if p < zero { -r } else { r };
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let h = g - d[l];
                for x in &mut d[l + 2..] {
                    *x -= h;
                }
                f += h;
                // The QL sweep, from the bottom of the block up.
                let mut p = d[m];
                let (mut c, mut c2, mut c3) = (one, one, one);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (zero, zero);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let (ei, di) = (e[i], d[i]);
                    let g = c * ei;
                    let h = c * p;
                    let r = length(p, ei);
                    e[i + 1] = s * r;
                    // `c·d_i - s·g` with `c = p/r` and `s = e_i/r`, divided once.
                    let t = fma(p, di, -(ei * g));
                    s = ei / r;
                    c = p / r;
                    p = t / r;
                    d[i + 1] = fma(s, fma(c, g, s * di), h);
                    if let Some((q, ldq, rows)) = q.as_mut() {
                        let (left, right) = q.split_at_mut((i + 1) * *ldq);
                        let x = &mut left[i * *ldq..i * *ldq + *rows];
                        rotate(x, &mut right[..*rows], c, s);
                    }
                }
                let p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = zero;
    }
    // Ascending, the vectors along.
    for i in 0..n {
        let mut k = i;
        for j in i + 1..n {
            if d[j] < d[k] {
                k = j;
            }
        }
        if k != i {
            d.swap(i, k);
            if let Some((q, ldq, rows)) = q.as_mut() {
                let (left, right) = q.split_at_mut(k * *ldq);
                left[i * *ldq..i * *ldq + *rows].swap_with_slice(&mut right[..*rows]);
            }
        }
    }
    true
}

/// `√(a² + b²)`: from the sum of the squares when it is in the range where no square can have
/// overflowed or underflowed to matter, as it nearly always is, and by `hypot` otherwise.
#[inline(always)]
fn length<R: Float>(a: R, b: R) -> R {
    let ss = fma(a, a, b * b);
    if ss >= R::MIN_POSITIVE / R::EPSILON && ss <= R::MAX {
        ss.sqrt()
    } else {
        a.hypot(b)
    }
}

/// Whether the target fuses multiply-adds in hardware; elsewhere `mul_add` is a library call.
const FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `a·b + c`, rounded once where the target fuses multiply-adds.
#[inline(always)]
fn fma<R: Float>(a: R, b: R, c: R) -> R {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `(x, y) = (c·x - s·y, s·x + c·y)` for the columns `x` and `y`, of any element type: the
/// same rotation of the real and the imaginary parts. Kept out of line: inlined into the sweep,
/// the compiler moves the sweep's next divisions below this loop, and each step of the sweep
/// then waits for the rotation of the vectors instead of running beside it.
#[inline(never)]
fn rotate<S: Scalar>(x: &mut [S], y: &mut [S], c: S::Real, s: S::Real) {
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (*x, *y);
        *x = S::from_parts(fma(-b.re(), s, a.re() * c), fma(-b.im(), s, a.im() * c));
        *y = S::from_parts(fma(a.re(), s, b.re() * c), fma(a.im(), s, b.im() * c));
    }
}
