//! The Hermite normal form of the rows modulo a multiple of the determinant (Domich, Kannan and
//! Trotter, 1987; Cohen, Algorithm 2.4.8, here for rows), and the forms of any matrix and their
//! transformations built on it.
//!
//! A lattice `L ⊆ Zⁿ` of rank `n` whose determinant divides `D` contains `D·Zⁿ`, so its Hermite
//! form is that of its generators with `D·Zⁿ` added, and every element can be kept in `[0, D)`.
//! Column `k` is taken in turn, modulo `R_k`, `R_0 = D`: the rows with an element there are
//! combined, by the extended Euclidean algorithm on two at a time, into one row `g` with the
//! greatest common divisor `g_k` of the column and rows with a 0 there; the pivot is
//! `d_k = gcd(g_k, R_k)`, the row of the form
//! `u·g` for `u·g_k ≡ d_k (mod R_k)`; and the vectors of `L` with zeros up to column `k`, which the
//! other rows span with `R_{k+1}·Z^{n−k−1}`, `R_{k+1} = R_k / d_k`, have the determinant
//! `det / (d_0⋯d_k)`, so the rest is computed modulo `R_{k+1}`. Last, each element above a pivot
//! is brought into `[0, d_l)` by the pivot's row, the elements after it kept below `R_{l+1}`. The
//! elements stay below `D`, and the work is `O(m·n²)` operations on numbers as long as `D`, for
//! `m` rows.
//!
//! For a matrix `A` of rank `r` with the column rank profile `J` and the independent rows `R`
//! (see [`super::profile`]), `M = A[R, J]` and `D = |det M|`: the columns `J` of the form are the
//! Hermite form of the rows of `A[:, J]`, a lattice of rank `r` whose determinant divides `D`,
//! from order [`pernet::FROM`] that of `M` by Pernet and Stein's method ([`super::pernet`]) with
//! the other rows added to it, below that (or when that method does not apply) the form modulo
//! `D` above (in words when `D` is below `2^63`, [`super::word`]); the other columns follow from them as `H[:, K] = H[:, J]·X`, `X = M⁻¹·A[R, K]`, since a vector
//! of the row space is fixed by its elements in `J`. The transformation is that of the square
//! matrix `B = [A[:, J] | E_S]` of the same determinant, `E_S` the columns of the identity for
//! the rows `S` outside `R`: `U = H_B·B⁻¹`, whose columns `S` are those of `H_B` and whose columns
//! `R` are `(H_B[:, J] − H_B[:, S]·A[S, J])·M⁻¹`, unimodular as the transformation between two
//! bases of one lattice. With `R` the row rank profile from the bottom, `H_B` is the Hermite form of
//! `[A | I]` on the columns of its pivots, so `U` is the transformation that form holds. `H_B` is
//! not computed as such: [`super::transform`] builds `U` from the structure of `B`, its unit
//! columns and the few pivots above 1 of its kernel part, in time linear in `m`.

use super::super::Matrix;
use super::inverse::times_inverse;
use super::profile::{Basis, rationals, select};
use super::transform::{Transform, transform};
use super::{pernet, word};
use crate::math::numbers::{Divisor, Number};
use crate::math::rational::Rational;

/// A matrix held row after row, for the row operations.
pub(super) type Rows = Vec<Vec<Number>>;

/// The rows of `a`.
pub(super) fn rows_of(a: &Matrix<Number>) -> Rows {
    (0..a.rows()).map(|i| a.row(i).cloned().collect()).collect()
}

/// The `rows × cols` matrix of the rows `r`.
pub(super) fn matrix_of(r: &[Vec<Number>], rows: usize, cols: usize) -> Matrix<Number> {
    Matrix::from_fn(rows, cols, |i, j| r[i][j].clone())
}

/// `x` modulo `d`, in `[0, |d|)`.
#[inline]
fn reduce(x: &mut Number, d: &Divisor) {
    if !x.is_zero() {
        *x = d.rem_euclid(x);
    }
}

/// The Hermite normal form of the lattice that the rows `active`, of `n` elements each, span, of rank
/// `n`, given a positive multiple `det` of its determinant: the `n × n` upper triangular basis with
/// a positive diagonal and the elements above each diagonal element in `[0, it)`.
pub(super) fn hermite_mod(mut active: Rows, n: usize, det: &Number) -> Rows {
    let one = Number::from(1u8);
    let mut modulus = det.abs();
    debug_assert!(!modulus.is_zero(), "a lattice of full rank");
    let d = Divisor::new(&modulus);
    for row in &mut active {
        row.iter_mut().for_each(|x| reduce(x, &d));
    }
    active.retain(|row| !row.iter().all(Number::is_zero));
    let mut h: Rows = Vec::with_capacity(n);
    // `moduli[k]` is `R_k`, by which everything from column `k` on may be reduced.
    let mut moduli = Vec::with_capacity(n + 1);
    for k in 0..n {
        moduli.push(modulus.clone());
        let d = Divisor::new(&modulus);
        let mut g: Option<Vec<Number>> = None;
        let mut rest = Vec::with_capacity(active.len());
        for mut v in active.drain(..) {
            if v[k].is_zero() {
                rest.push(v);
                continue;
            }
            let Some(g) = g.as_mut() else {
                g = Some(v);
                continue;
            };
            combine(g, &mut v, k, &d);
            if !v[k + 1..].iter().all(Number::is_zero) {
                rest.push(v);
            }
        }
        // The pivot `d_k = gcd(g_k, R_k)`, and the row `u·g` with `u·g_k ≡ d_k`.
        let row = match g {
            None => {
                // `gcd(0, R_k) = R_k`: the row `R_k·e_k`.
                let mut row = vec![Number::default(); n];
                row[k] = modulus.clone();
                row
            }
            Some(mut g) => {
                let (dk, u, _) = g[k].gcd_ext(&modulus);
                if u != one {
                    for x in &mut g[k + 1..] {
                        *x = &u * &*x;
                    }
                }
                g[k] = dk;
                g
            }
        };
        let next = modulus.div_exact(&row[k]);
        let d = Divisor::new(&next);
        let mut row = row;
        row[k + 1..].iter_mut().for_each(|x| reduce(x, &d));
        for v in &mut rest {
            v[k + 1..].iter_mut().for_each(|x| reduce(x, &d));
        }
        rest.retain(|v| !v[k + 1..].iter().all(Number::is_zero));
        h.push(row);
        active = rest;
        modulus = next;
    }
    moduli.push(modulus);
    // Each element above a pivot into `[0, pivot)`, by the pivot's row; the rows above it have
    // their elements after it below `R_l`, and get them below `R_{l+1}`.
    for l in 1..n {
        let (above, below) = h.split_at_mut(l);
        let pivot_row = &below[0];
        let pivot = &pivot_row[l];
        let shrinks = moduli[l + 1] != moduli[l];
        let d = Divisor::new(&moduli[l + 1]);
        for row in above {
            let q = row[l].div_floor(pivot);
            if !q.is_zero() {
                for (x, y) in row[l..].iter_mut().zip(&pivot_row[l..]) {
                    if !y.is_zero() {
                        *x -= &q * y;
                    }
                }
            }
            if shrinks || !q.is_zero() {
                row[l + 1..].iter_mut().for_each(|x| reduce(x, &d));
            }
        }
    }
    h
}

/// Combines the rows `g` and `v`, both with an element in column `k`, into `g` with their greatest
/// common divisor there and `v` with a 0, by a unimodular step, from column `k` on, modulo `d`:
/// `v ← v − q·g` when `g_k` divides `v_k`, and otherwise `(g, v) ← (x·g + y·v, (g_k/e)·v − (v_k/e)·g)`
/// for `e = gcd(g_k, v_k) = x·g_k + y·v_k`.
fn combine(g: &mut [Number], v: &mut [Number], k: usize, d: &Divisor) {
    let (gk, vk) = (&g[k], &v[k]);
    let (q, r) = vk.div_rem(gk);
    if r.is_zero() {
        for (x, y) in v[k..].iter_mut().zip(&g[k..]) {
            if !y.is_zero() {
                *x -= &q * y;
                reduce(x, d);
            }
        }
        return;
    }
    let (e, x, y) = gk.gcd_ext(vk);
    let a = gk.div_exact(&e);
    let b = vk.div_exact(&e);
    for (gj, vj) in g[k..].iter_mut().zip(v[k..].iter_mut()) {
        if gj.is_zero() && vj.is_zero() {
            continue;
        }
        let mut new_g = &x * &*gj + &y * &*vj;
        let mut new_v = &a * &*vj - &b * &*gj;
        reduce(&mut new_g, d);
        reduce(&mut new_v, d);
        *gj = new_g;
        *vj = new_v;
    }
}

/// The integer matrix of the rationals `x`, which must all be integers.
pub(super) fn integers(x: Matrix<Rational>) -> Matrix<Number> {
    let (rows, cols) = x.shape();
    let data = x
        .into_vec()
        .into_iter()
        .map(|q| {
            // Integers by construction: a fraction here would be a fault of this module, and the
            // check costs nothing beside the solution that made it.
            assert!(q.is_integer(), "a fraction in a transformation of integers");
            q.into_parts().0
        })
        .collect();
    Matrix::from_column_major(rows, cols, data)
}

/// The Hermite normal form of `a` from its rank profiles `b` and the Hermite form `hj` of the rows
/// of `A[:, J]`: `H[:, J] = hj`, `H[:, K] = hj·X`, the rows from the rank on zero.
fn assemble(a: &Matrix<Number>, b: &Basis, hj: &Matrix<Number>) -> Matrix<Number> {
    let (m, n) = a.shape();
    let r = b.rank;
    let mut h = Matrix::zeros(m, n);
    for (k, &c) in b.cols.iter().enumerate() {
        for i in 0..r {
            h[(i, c)] = hj[(i, k)].clone();
        }
    }
    if r > 0 && !b.free.is_empty() {
        let hk = integers(&rationals(hj) * &b.x);
        for (f, &c) in b.free.iter().enumerate() {
            for i in 0..r {
                h[(i, c)] = hk[(i, f)].clone();
            }
        }
    }
    h
}

/// The Hermite form of the rows of `A[:, J]`: for a minor of order [`pernet::FROM`] or more, the
/// form of `M` by [`pernet::square`] (which gives `|det M|`, the product of its pivots, too) and
/// the other rows added to it; otherwise, or when that method does not apply, modulo
/// `|det M|`.
fn form_of_profile(a: &Matrix<Number>, b: &Basis) -> Matrix<Number> {
    if let Some(h) = b.hj.get() {
        return h.clone();
    }
    let r = b.rank;
    if r >= pernet::FROM.max(3)
        && let Some(mut h) = pernet::square(&b.minor)
    {
        let det: Number = (0..r).map(|i| h[i][i].clone()).product();
        let _ = b.det.set(det);
        for &i in &b.others {
            pernet::add_row(&mut h, b.cols.iter().map(|&c| a[(i, c)].clone()).collect());
        }
        if !b.others.is_empty() {
            pernet::reduce_above(&mut h);
        }
        return matrix_of(&h, r, r);
    }
    let det = b.det();
    if det.bits() < 63 {
        // In words, the residues read from `a` in place.
        return matrix_of(&word::hermite_mod_cols(a, &b.cols, det), r, r);
    }
    let aj = select(a, &(0..a.rows()).collect::<Vec<_>>(), &b.cols);
    matrix_of(&pernet::hermite_mod_any(rows_of(&aj), r, det), r, r)
}

/// The Hermite normal form of `a`, its rank profiles `b` proved.
pub(super) fn form(a: &Matrix<Number>, b: &Basis) -> Matrix<Number> {
    assemble(a, b, &form_of_profile(a, b))
}

/// The Hermite normal form of `a` and its transformation `U`, `U·a = H`, for its column rank
/// profile `J`, independent rows `R` and `Y` in `b`: with `R` the row rank profile from the
/// bottom, `U` is the one of the Hermite form of `[A | I]`; with other rows, the one of the form of
/// `[A[:, J] | E_S]`. `U` comes as a [`Transform`], from [`transform`].
pub(super) fn form_and_transform(a: &Matrix<Number>, b: &Basis) -> (Matrix<Number>, Transform) {
    let hj = form_of_profile(a, b);
    let h = assemble(a, b, &hj);
    (h, transform(a.rows(), b, &hj))
}

/// The Hermite normal form `H` of the square nonsingular `a`, of determinant `±det`, and the
/// transformation `P = H·a⁻¹`, unimodular: `H` by Pernet and Stein's method from order
/// [`pernet::FROM`] (modulo `det` below it, or when it does not apply), `P` from its residues
/// modulo primes.
pub(super) fn square_and_transform(
    a: &Matrix<Number>,
    det: &Number,
) -> (Matrix<Number>, Matrix<Number>) {
    let n = a.rows();
    let rows = if n >= pernet::FROM.max(3) {
        pernet::square(a)
    } else {
        None
    };
    let rows = rows.unwrap_or_else(|| pernet::hermite_mod_any(rows_of(a), n, det));
    let h = matrix_of(&rows, n, n);
    let p = times_inverse(&h, a, &det.abs(), &Number::from(1u8));
    (h, p)
}
