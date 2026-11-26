//! The lower bound on the norm of every relation, proved in exact arithmetic from the rows of the
//! unimodular `A` that PSLQ holds.
//!
//! Ferguson, Bailey and Arno's bound `1/max |Hⱼⱼ|` (Math. Comp. 68, 1999, Theorem 1) rests on
//! this: let `m ≠ 0` be an integer relation of a real vector `ξ`, and `j` the first row of `A`
//! with `⟨m, aⱼ⟩ ≠ 0`, an integer, so at least 1. Write `aⱼ = c₀·X + Σ_{i<j} cᵢ·aᵢ + rⱼ` with any
//! real `c`; then `⟨m, aᵢ⟩ = 0` for `i < j` gives `1 ≤ |c₀·⟨m, X⟩ + ⟨m, rⱼ⟩|`. For `ξ` within the
//! errors `δ` of `X`, `|⟨m, X⟩| = |⟨m, X − ξ⟩| ≤ |m|·|δ|`, so `|m| ≥ 1/(|c₀|·|δ| + |rⱼ|)`.
//! With `c` the coefficients of the orthogonal projection, `|rⱼ|` is the Gram–Schmidt norm of `aⱼ`
//! after `X` and the rows before it, which is `|Hⱼⱼ|`; but any `c` gives a true bound. So `c` is
//! computed in floating point, taken as the exact dyadic rationals it is, and `rⱼ` and the norms
//! are computed from it exactly, in integers. The last row needs no such bound: an `m`
//! orthogonal to all the others is a multiple of the last column of `B = A⁻¹`, which is a
//! relation exactly when its multiples are, and the search has tested it.

use super::pslq::Input;
use crate::math::float::{BigFloat, Round};
use crate::math::numbers::Number;

/// A lower bound on the norm of every integer relation of every vector within the errors of the
/// numbers, at 64 bits rounded down, from the rows `a₀, …, a_{n−2}` of `a` (`n × n` by rows),
/// given that the last column of `A⁻¹` is not a relation. `frac` is the precision of the search.
pub(super) fn proved_bound(input: &Input, a: &[Number], frac: u64) -> BigFloat {
    let n = input.x.len();
    let prec = frac + 64;
    let rows: Vec<&[Number]> = core::iter::once(&input.x[..])
        .chain((0..n - 1).map(|j| &a[j * n..(j + 1) * n]))
        .collect();
    let l = gram_schmidt(&rows, prec);
    // |δ| ≤ ⌊√D⌋ + 1, D = Σ δᵢ², each δᵢ a power of two.
    let d = input.err.iter().fold(Number::from(0), |s, &e| {
        s + (Number::from(1) << (2 * e) as usize)
    });
    let delta = d.isqrt() + Number::from(1);
    let mut largest = BigFloat::zero(64);
    for j in 1..n {
        let c = coefficients(&l, j);
        let u = residual_bound(&rows, j, &c, &delta);
        if u > largest {
            largest = u;
        }
    }
    let one = BigFloat::from(1);
    one.div_round(&largest, 64, Round::Down).0
}

/// The lower triangle `L` of the rows `v₀, …, v_{n−1}` = `L·Q`, `Q` orthonormal, by modified
/// Gram–Schmidt run twice over each row (twice is enough, Giraud, Langou and Rozložník); `L[j]`
/// holds `j + 1` entries. A row dependent on those before it to the precision gets a zero
/// diagonal and no direction.
fn gram_schmidt(rows: &[&[Number]], prec: u64) -> Vec<Vec<BigFloat>> {
    let n = rows[0].len();
    let mut q: Vec<Option<Vec<BigFloat>>> = Vec::with_capacity(rows.len());
    let mut l: Vec<Vec<BigFloat>> = Vec::with_capacity(rows.len());
    for (j, row) in rows.iter().enumerate() {
        let mut w: Vec<BigFloat> = row
            .iter()
            .map(|v| BigFloat::from_number(v, prec, Round::Nearest).0)
            .collect();
        let mut lj = vec![BigFloat::zero(prec); j + 1];
        for _ in 0..2 {
            for (i, qi) in q.iter().enumerate() {
                let Some(qi) = qi else { continue };
                let c = BigFloat::dot_round(&w, qi, prec, Round::Nearest).0;
                if c.is_zero() {
                    continue;
                }
                let minus = -c.clone();
                for (wk, qk) in w.iter_mut().zip(qi) {
                    *wk = minus.mul_add_round(qk, wk, prec, Round::Nearest).0;
                }
                lj[i] = lj[i].add_round(&c, prec, Round::Nearest).0;
            }
        }
        let norm = BigFloat::dot_round(&w, &w, prec, Round::Nearest)
            .0
            .sqrt_round(prec, Round::Nearest)
            .0;
        if norm.is_zero() {
            q.push(None);
        } else {
            let dir = w
                .iter()
                .map(|wk| wk.div_round(&norm, prec, Round::Nearest).0)
                .collect::<Vec<_>>();
            debug_assert_eq!(dir.len(), n);
            q.push(Some(dir));
        }
        lj[j] = norm;
        l.push(lj);
    }
    l
}

/// The coefficients `c₀, …, c_{j−1}` of the projection of `vⱼ` on `v₀, …, v_{j−1}`: the solution
/// of `Σᵢ cᵢ·L[i][k] = L[j][k]` for `k < j`, by back substitution; `cₖ = 0` where `L[k][k]` is.
fn coefficients(l: &[Vec<BigFloat>], j: usize) -> Vec<BigFloat> {
    let prec = l[j][j].prec();
    let mut c = vec![BigFloat::zero(prec); j];
    for k in (0..j).rev() {
        if l[k][k].is_zero() {
            continue;
        }
        let mut s = l[j][k].clone();
        for i in k + 1..j {
            if !c[i].is_zero() {
                let minus = -c[i].clone();
                s = minus.mul_add_round(&l[i][k], &s, prec, Round::Nearest).0;
            }
        }
        c[k] = s.div_round(&l[k][k], prec, Round::Nearest).0;
        // Any c gives a true bound: a coefficient far below the precision, which would only make
        // the exact integers of the residual longer, is dropped.
        if c[k].exponent().is_some_and(|e| e < -4 * prec as i64) {
            c[k] = BigFloat::zero(prec);
        }
    }
    c
}

/// An upper bound, at 64 bits, on `|c₀|·|δ| + |rⱼ|` for `rⱼ = vⱼ − Σ_{i<j} cᵢ·vᵢ`, computed
/// exactly from the dyadic `c`; `delta ≥ |δ|`.
fn residual_bound(rows: &[&[Number]], j: usize, c: &[BigFloat], delta: &Number) -> BigFloat {
    // cᵢ = Cᵢ / 2^k with integers Cᵢ.
    let parts: Vec<Option<(Number, i64)>> = c.iter().map(BigFloat::mantissa_exponent).collect();
    let k = parts
        .iter()
        .flatten()
        .map(|(_, e)| -*e)
        .max()
        .unwrap_or(0)
        .max(0);
    let big: Vec<Option<Number>> = parts
        .into_iter()
        .map(|p| p.map(|(m, e)| m << (e + k) as usize))
        .collect();
    let mut s = Number::from(0);
    for (col, v) in rows[j].iter().enumerate() {
        let mut r = v << k as usize;
        for (ci, row) in big.iter().zip(rows) {
            if let Some(ci) = ci {
                r -= ci * &row[col];
            }
        }
        s += &r * &r;
    }
    let c0 = big[0].as_ref().map(Number::abs).unwrap_or_default();
    let numer = s.isqrt() + Number::from(1) + &c0 * delta;
    BigFloat::from_parts(&numer, -k, 64, Round::Up).0
}
