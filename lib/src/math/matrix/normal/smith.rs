//! The Smith normal form: its diagonal modulo a multiple of the determinant (Cohen, Algorithm
//! 2.4.14, for a matrix of any shape and rank), and the form with its transformations by
//! alternating Hermite forms of the rows and of the columns (Kannan and Bachem, 1979).
//!
//! **The diagonal.** For `A` of rank `r`, the product `d_1⋯d_r` of the nonzero elementary
//! divisors is the greatest common divisor of the `r × r` minors, so it divides `D = |det M|` for
//! the nonsingular minor `M` of the rank profiles, and the Smith form of `A` over `Z/DZ` is
//! `diag(d_1, …, d_r, 0, …)`. Position `t` takes a nonzero element of the remaining block as its
//! pivot, then clears its column and its row by unimodular steps of two rows or two columns
//! (a subtraction when the pivot divides, the extended Euclidean step otherwise, which makes the
//! pivot a proper divisor of itself), until both are clear; when the pivot's `g = gcd(pivot, R)`
//! does not divide some element of the block, that element's row is added to the pivot's and the
//! clearing goes on, with a smaller pivot. Then `d_t = g`, and the block is reduced modulo
//! `R / d_t`, a multiple of the product of the divisors left. The elements stay below `D`;
//! the work is `O(m·n·r)` operations on numbers as long as `D`, and each position repeats its
//! clearing at most `log₂ D` times.
//!
//! **The form with its transformations.** The Hermite form `U₁·A = H` (with the transformation
//! of [`super::hermite`]), then that of the columns of its top `r` rows, `H[..r]·V₁ = [K | 0]`, `K`
//! lower triangular of order `r`, `V₁ = Wᵀ` for the transformation `W` of the form of
//! `[H[..r]ᵀ | E_K]` (see [`transposed`]). Both are held as
//! [`Transform`](super::transform::Transform)s, their first `r` rows
//! dense and their last ones, bases of the left kernels of `A` and of `H[..r]ᵀ`, sparse; only the
//! first rows change from here on, so the work stays linear in `m` and `n`. `K` is put in its
//! Hermite form by rows and by columns in turn, each time with the unique transformation of a
//! nonsingular matrix, until it is diagonal: the first element either becomes a proper divisor
//! of itself or, when it divides the rest of its column (or row), its row and its column are
//! cleared and stay so, which leaves the block after it; so the turns are at most `r` plus the
//! number of prime factors of the determinants of the blocks. Then `diag(a, b)` becomes
//! `diag(gcd, lcm)` by `P = [[x, y], [−b/g, a/g]]` and `Q = [[1, −y·b/g], [1, x·a/g]]`,
//! `P·diag(a, b)·Q`, or `Qᵀ·diag(a, b)·Pᵀ`, for `g = x·a + y·b`: gcd and lcm sort the exponent of
//! every prime at once, so the steps can be the comparators of a sorting network, Batcher's
//! odd-even merge sort, `O(r·log² r)` of them and `O(log² r)` on each position; the pairs one
//! after the other took each position through `r` steps, each lengthening its lines by about the
//! length of the larger element, which made the transformations of a diagonal of `r` elements
//! of `b` bits about `r²·b/2` bits long. Last, the first rows of `U` and `W` are reduced modulo
//! their kernel rows.

use super::super::Matrix;
use super::hermite::{Rows, form_and_transform, rows_of, square_and_transform};
use super::profile::Basis;
use crate::math::numbers::{Divisor, Number};

/// `x` modulo `d`, in `[0, |d|)`.
#[inline]
fn reduce(x: &mut Number, d: &Divisor) {
    if !x.is_zero() {
        *x = d.rem_euclid(x);
    }
}

/// The nonzero elementary divisors `d_1 | … | d_r` of `a`, of rank `r ≥ 1`, given a positive
/// multiple `det` of their product.
pub(super) fn divisors_mod(mut a: Rows, r: usize, det: &Number) -> Vec<Number> {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    let mut modulus = det.abs();
    let mut d = Divisor::new(&modulus);
    for row in &mut a {
        row.iter_mut().for_each(|x| reduce(x, &d));
    }
    let mut out = Vec::with_capacity(r);
    for t in 0..r {
        if modulus == 1u8 {
            out.push(Number::from(1u8));
            continue;
        }
        let Some((pi, pj)) =
            (t..m).find_map(|i| (t..n).find(|&j| !a[i][j].is_zero()).map(|j| (i, j)))
        else {
            // The block is zero modulo `R`: its first divisor is `R`, and the rest are 1.
            out.push(modulus.clone());
            modulus = Number::from(1u8);
            continue;
        };
        a.swap(t, pi);
        for row in &mut a[t..] {
            row.swap(t, pj);
        }
        let g = loop {
            clear(&mut a, t, &d);
            let g = a[t][t].gcd(&modulus);
            // An element of the block that `g` does not divide: its row is added to row `t`.
            let found = (t + 1..m).find(|&i| a[i][t + 1..].iter().any(|x| !x.is_multiple_of(&g)));
            let Some(i) = found else { break g };
            let (top, rest) = a.split_at_mut(i);
            for (x, y) in top[t][t + 1..].iter_mut().zip(&rest[0][t + 1..]) {
                *x += y;
                reduce(x, &d);
            }
        };
        modulus = modulus.div_exact(&g);
        out.push(g);
        d = Divisor::new(&modulus);
        for row in &mut a[t + 1..] {
            row[t + 1..].iter_mut().for_each(|x| reduce(x, &d));
        }
    }
    out
}

/// Clears column `t` below the pivot `a[t][t]`, which is not zero, and row `t` after it, modulo
/// `d`, by unimodular steps of two rows or two columns, until both are clear.
fn clear(a: &mut Rows, t: usize, d: &Divisor) {
    let m = a.len();
    let n = a[t].len();
    loop {
        for i in t + 1..m {
            if a[i][t].is_zero() {
                continue;
            }
            let (top, rest) = a.split_at_mut(i);
            step(&mut top[t][t..], &mut rest[0][t..], d);
        }
        let mut refilled = false;
        for j in t + 1..n {
            if a[t][j].is_zero() {
                continue;
            }
            let (x, y) = (a[t][t].clone(), a[t][j].clone());
            let (q, rem) = y.div_rem(&x);
            if rem.is_zero() {
                // Column `j` minus `q` times column `t`: column `t` is unchanged.
                for row in &mut a[t..] {
                    if !row[t].is_zero() {
                        let s = &q * &row[t];
                        row[j] -= s;
                        reduce(&mut row[j], d);
                    }
                }
                continue;
            }
            let (e, u, v) = x.gcd_ext(&y);
            let (p, s) = (x.div_exact(&e), y.div_exact(&e));
            for row in &mut a[t..] {
                let (ct, cj) = (&row[t], &row[j]);
                if ct.is_zero() && cj.is_zero() {
                    continue;
                }
                let mut nt = &u * ct + &v * cj;
                let mut nj = &p * cj - &s * ct;
                reduce(&mut nt, d);
                reduce(&mut nj, d);
                row[t] = nt;
                row[j] = nj;
            }
            refilled = true;
        }
        if !refilled || a[t + 1..].iter().all(|row| row[t].is_zero()) {
            return;
        }
    }
}

/// The step of two rows `g` and `v` at their first elements, `g[0]` not zero, modulo `d`: `v` gets
/// a 0 there, by `v ← v − q·g` when `g[0]` divides `v[0]`, and otherwise by
/// `(g, v) ← (x·g + y·v, (g₀/e)·v − (v₀/e)·g)` for `e = gcd(g₀, v₀) = x·g₀ + y·v₀`.
fn step(g: &mut [Number], v: &mut [Number], d: &Divisor) {
    let (q, rem) = v[0].div_rem(&g[0]);
    if rem.is_zero() {
        for (x, y) in v.iter_mut().zip(g.iter()) {
            if !y.is_zero() {
                *x -= &q * y;
                reduce(x, d);
            }
        }
        return;
    }
    let (e, x, y) = g[0].gcd_ext(&v[0]);
    let (a, b) = (g[0].div_exact(&e), v[0].div_exact(&e));
    for (gj, vj) in g.iter_mut().zip(v.iter_mut()) {
        if gj.is_zero() && vj.is_zero() {
            continue;
        }
        let mut ng = &x * &*gj + &y * &*vj;
        let mut nv = &a * &*vj - &b * &*gj;
        reduce(&mut ng, d);
        reduce(&mut nv, d);
        *gj = ng;
        *vj = nv;
    }
}

/// The elementary divisors of `a`, its rank profiles `b` proved: `min(m, n)` of them, the zeros
/// last.
pub(super) fn divisors(a: &Matrix<Number>, b: &Basis) -> Vec<Number> {
    let k = a.rows().min(a.cols());
    let mut out = if b.rank == 0 {
        Vec::with_capacity(k)
    } else {
        divisors_mod(rows_of(a), b.rank, &b.det)
    };
    out.resize(k, Number::default());
    out
}

/// The profiles of `H[..r]ᵀ` that make its transformation, for the Hermite form `H` of a matrix of
/// the profiles `b`: its `r` columns are independent; its rows `J` are independent, the minor
/// `H[..r, J]ᵀ` lower triangular with the pivots on its diagonal, and its other rows, `K`, are
/// `Y = Xᵀ` times those, as `H[..r, K] = H[..r, J]·X`. `J` is in general not the row rank profile
/// from the bottom, so [`super::transform`] gives the transformation `W` of the Hermite form of
/// `[H[..r]ᵀ | E_K]` (`E_K` the columns of the identity for the rows `K`), not that of
/// `[H[..r]ᵀ | I]`. Proving profiles again would take the determinant of a matrix whose elements
/// are as long as the determinant of `M`, by bounds far above it.
fn transposed(h: &Matrix<Number>, b: &Basis) -> Basis {
    let r = b.rank;
    let minor = Matrix::from_fn(r, r, |i, j| h[(j, b.cols[i])].clone());
    let det = (0..r).map(|k| minor[(k, k)].clone()).product();
    Basis {
        rank: r,
        cols: (0..r).collect(),
        free: Vec::new(),
        rows: b.cols.clone(),
        others: b.free.clone(),
        minor,
        det,
        x: Matrix::zeros(r, 0),
        y: Some(b.x.transpose()),
    }
}

/// Whether row `s` and column `s` of the square `k` are zero but for the diagonal.
fn is_clear(k: &Matrix<Number>, s: usize) -> bool {
    let r = k.rows();
    (0..r).all(|j| j == s || (k[(s, j)].is_zero() && k[(j, s)].is_zero()))
}

/// The rows `from..from + p.rows()` of `rows` replaced by `p` times them.
fn mul_rows(rows: &mut [Vec<Number>], from: usize, p: &Matrix<Number>) {
    let k = p.rows();
    let len = rows.first().map_or(0, Vec::len);
    let block: Vec<Vec<Number>> = (0..k)
        .map(|i| {
            (0..len)
                .map(|c| {
                    let mut acc = Number::default();
                    for t in 0..k {
                        let (x, y) = (&p[(i, t)], &rows[from + t][c]);
                        if !x.is_zero() && !y.is_zero() {
                            acc += x * y;
                        }
                    }
                    acc
                })
                .collect()
        })
        .collect();
    for (i, row) in block.into_iter().enumerate() {
        rows[from + i] = row;
    }
}

/// Rows `i` and `j` of `rows` replaced by `[[p00, p01], [p10, p11]]` times them.
fn mul_pair(rows: &mut [Vec<Number>], i: usize, j: usize, p: &[[Number; 2]; 2]) {
    let (lo, hi) = rows.split_at_mut(j);
    for (x, y) in lo[i].iter_mut().zip(hi[0].iter_mut()) {
        if x.is_zero() && y.is_zero() {
            continue;
        }
        let nx = &p[0][0] * &*x + &p[0][1] * &*y;
        let ny = &p[1][0] * &*x + &p[1][1] * &*y;
        *x = nx;
        *y = ny;
    }
}

/// The length in bits of the longest element of rows `i` and `j`.
fn bits(rows: &[Vec<Number>], i: usize, j: usize) -> u64 {
    rows[i]
        .iter()
        .chain(&rows[j])
        .map(Number::bits)
        .max()
        .unwrap_or(0)
}

/// The comparators of Batcher's odd-even merge sort of `k` elements, `(i, j)` with `i < j`, each
/// to put the smaller element at `i`: the sort of the next power of two, without the comparators
/// that reach past `k`, which is the sort of `k` elements and as many larger ones after them.
fn network(k: usize) -> Vec<(usize, usize)> {
    /// The comparators of the merge of the two sorted halves of `lo..=hi` taken `r` apart.
    fn merge(lo: usize, hi: usize, r: usize, out: &mut Vec<(usize, usize)>) {
        let step = 2 * r;
        if step < hi - lo {
            merge(lo, hi, step, out);
            merge(lo + r, hi, step, out);
            let mut i = lo + r;
            while i + r < hi {
                out.push((i, i + r));
                i += step;
            }
        } else {
            out.push((lo, lo + r));
        }
    }
    /// The comparators of the sort of `lo..=hi`.
    fn sort(lo: usize, hi: usize, out: &mut Vec<(usize, usize)>) {
        if hi > lo {
            let mid = lo + (hi - lo) / 2;
            sort(lo, mid, out);
            sort(mid + 1, hi, out);
            merge(lo, hi, 1, out);
        }
    }
    let mut out = Vec::new();
    if k > 1 {
        sort(0, k.next_power_of_two() - 1, &mut out);
    }
    out.retain(|&(_, j)| j < k);
    out
}

/// The Smith normal form of `a` with its transformations, `U·a·V = S`, its rank profiles `b`
/// proved with the rows from the bottom.
pub(super) fn form(
    a: &Matrix<Number>,
    b: &Basis,
) -> (Matrix<Number>, Matrix<Number>, Matrix<Number>) {
    let (m, n) = a.shape();
    let r = b.rank;
    let (h, mut u) = form_and_transform(a, b);
    // The Hermite form of the columns of the top `r` rows: `G = W·H[..r]ᵀ = [T; 0]`, and `V = Wᵀ`.
    // Both transformations keep their last rows, a basis of a left kernel each, apart: only their
    // first `r` rows change from here on.
    let top = Matrix::from_fn(n, r, |i, j| h[(j, i)].clone());
    let (g, mut w) = form_and_transform(&top, &transposed(&h, b));
    // `K = Tᵀ`, lower triangular, `U·A·Wᵀ = diag(K, 0)` from here on.
    let mut k = Matrix::from_fn(r, r, |i, j| g[(j, i)].clone());
    let mut rows_turn = true;
    let mut s = 0;
    loop {
        while s < r && is_clear(&k, s) {
            s += 1;
        }
        if s == r {
            break;
        }
        // The block after the clear rows and columns, triangular: its determinant is the product
        // of its diagonal.
        let size = r - s;
        let block = Matrix::from_fn(size, size, |i, j| k[(s + i, s + j)].clone());
        let det: Number = (0..size).map(|i| block[(i, i)].abs()).product();
        let new = if rows_turn {
            let (hb, p) = square_and_transform(&block, &det);
            mul_rows(&mut u.top, s, &p);
            hb
        } else {
            // `q·blockᵀ = hb`: the columns of `V` times `qᵀ`, the rows of `W` times `q`.
            let (hb, q) = square_and_transform(&block.transpose(), &det);
            mul_rows(&mut w.top, s, &q);
            hb.transpose()
        };
        for i in 0..size {
            for j in 0..size {
                k[(s + i, s + j)] = new[(i, j)].clone();
            }
        }
        rows_turn = !rows_turn;
    }
    // Each element dividing the next: `diag(a, b) → diag(gcd, lcm)` by
    // `P = [[x, y], [−b/g, a/g]]` on the rows of `U` and `Q = [[1, −y·b/g], [1, x·a/g]]` on the
    // columns of `V`, which is `Qᵀ` on the rows of `W`; or the other way round, `Qᵀ` on `U` and
    // `P` on `W`, as `Qᵀ·diag(a, b)·Pᵀ` is the same. `Q`, the larger, goes to the side whose two
    // rows are the shorter. The comparators are those of a sorting network, as gcd and lcm sort
    // the exponent of every prime at once; Batcher's takes each position through
    // `O(log² r)` steps, where the pairs one after the other would take it through `r`, each step
    // lengthening the rows by about the length of the larger element.
    let mut diag: Vec<Number> = (0..r).map(|i| k[(i, i)].clone()).collect();
    for (i, j) in network(r) {
        if diag[j].is_multiple_of(&diag[i]) {
            continue;
        }
        let (x, y) = (diag[i].clone(), diag[j].clone());
        let (gcd, p, q) = x.gcd_ext(&y);
        let (xg, yg) = (x.div_exact(&gcd), y.div_exact(&gcd));
        let small = [[p.clone(), q.clone()], [-&yg, xg.clone()]];
        let large = [
            [Number::from(1u8), Number::from(1u8)],
            [-(&q * &yg), &p * &xg],
        ];
        if bits(&u.top, i, j) <= bits(&w.top, i, j) {
            mul_pair(&mut u.top, i, j, &large);
            mul_pair(&mut w.top, i, j, &small);
        } else {
            mul_pair(&mut u.top, i, j, &small);
            mul_pair(&mut w.top, i, j, &large);
        }
        diag[j] = &xg * &y;
        diag[i] = gcd;
    }
    // The first rows of each, modulo its last rows, the left kernels.
    u.reduce_top();
    w.reduce_top();
    let mut smith = Matrix::zeros(m, n);
    for (i, x) in diag.into_iter().enumerate() {
        smith[(i, i)] = x;
    }
    (smith, u.into_matrix(false), w.into_matrix(true))
}
