//! The search of the enumeration: which levels can hold a solution, the float search where it
//! can follow the data and the exact one where it cannot, and the exact check of every leaf.

use super::enum_exact;
use super::enum_float::{FloatLevels, TINY};
use super::enum_gso::{Gso, log2_ratio, ratio, scale};
use crate::math::numbers::Number;

/// A target of the enumeration, reduced by Babai's nearest plane method so that
/// `|λ_{t,l}| ≤ d_{l+1}/2` at every level.
pub(super) struct Target<'a> {
    /// `λ_{t,j} = d_{j+1}·μ_{t,j}`.
    pub(super) lam: &'a [Number],
    /// `T_j = d_j·|π_j(t)|²`, for `j` up to the number of rows.
    pub(super) norms: &'a [Number],
}

/// What to enumerate: the vectors `w = Σ_{lo ≤ l < hi} x_l·b_l − t` of the rows `lo..hi` of an
/// orthogonalised basis, by their projections `π_lo(w)`, and without a target (`t = 0`) the
/// nonzero ones only, one of each `±w`. Their measure is the integer `L_lo = d_lo·|π_lo(w)|²`.
pub(super) struct Levels<'a> {
    pub(super) gso: &'a Gso,
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) target: Option<Target<'a>>,
}

impl Levels<'_> {
    /// `d_hi·|π_hi(w)|²`, the same for every `w`: `T_hi`, or 0 without a target.
    pub(super) fn top(&self) -> Number {
        match &self.target {
            Some(t) => t.norms[self.hi].clone(),
            None => Number::default(),
        }
    }
}

/// The coefficients and the measure of the least `L_lo ≤ bound`; `None` if there is none.
pub(super) fn least(levels: &Levels, bound: &Number) -> Option<(Vec<i64>, Number)> {
    run(levels, bound, true).pop()
}

/// The coefficients and the measure of every vector with `L_lo ≤ bound`.
pub(super) fn all(levels: &Levels, bound: &Number) -> Vec<(Vec<i64>, Number)> {
    run(levels, bound, false)
}

/// The levels from `hi` down that no solution can use: there every coefficient is 0. Without a
/// target, the rows `K..hi` whose Gram–Schmidt norms all exceed the radius span a projected
/// lattice with no nonzero vector within it. With a reduced target, a coefficient `x_l ≠ 0` with
/// those above it zero is at least `1/2` from its centre `μ_{t,l}`, so it adds at least `r_l/4`
/// to the part `|π_{l+1}(t)|²` above it.
fn trim<'a>(levels: &Levels<'a>, bound: &Number) -> Levels<'a> {
    let g = levels.gso;
    let (lo, mut hi) = (levels.lo, levels.hi);
    let dlo = &g.d[lo];
    while hi > lo {
        let l = hi - 1;
        let forced = match &levels.target {
            // r_l > bound/d_lo.
            None => &g.d[l + 1] * dlo > bound * &g.d[l],
            // r_l/4 + T_{l+1}/d_{l+1} > bound/d_lo.
            Some(t) => {
                let left =
                    &(&(&g.d[l + 1] * &g.d[l + 1]) + &(&(&g.d[l] * &t.norms[l + 1]) * 4u8)) * dlo;
                left > &(&(&g.d[l] * &g.d[l + 1]) * bound) * 4u8
            }
        };
        if !forced {
            break;
        }
        hi = l;
    }
    Levels {
        gso: g,
        lo,
        hi,
        target: levels.target.as_ref().map(|t| Target {
            lam: t.lam,
            norms: t.norms,
        }),
    }
}

/// The radius of the enumerated levels as a rational `(num, den)`: `ℓ ≤ bound/d_lo − top/d_hi`.
fn radius_of(levels: &Levels, bound: &Number) -> (Number, Number) {
    let g = levels.gso;
    let (dlo, dhi) = (&g.d[levels.lo], &g.d[levels.hi]);
    (&(bound * dhi) - &(&levels.top() * dlo), dlo * dhi)
}

/// The float radius for `bound`, scaled by `2^-s` and enlarged past the rounding of the search;
/// −1 when no vector can be within it.
fn float_radius(levels: &Levels, bound: &Number, s: i64) -> f64 {
    let (num, den) = radius_of(levels, bound);
    if num.is_negative() {
        return -1.0;
    }
    let n = (levels.hi - levels.lo) as f64;
    let (m, e) = ratio(&num, &den);
    // Rounded once (below 1.01·2⁻⁵³), then enlarged past the (1 + u)^{5n} of the search and
    // that rounding. A radius below 2⁻¹⁰⁰⁰ could be subnormal and lose its relative accuracy;
    // 2⁻¹⁰⁰⁰ is above it, which only widens the search.
    let r = scale(m, e - s).max(1.0e-301);
    r * (1.0 + (8.0 * n + 32.0) * (f64::EPSILON / 2.0))
}

/// The float data of the levels, scaled by `2^-s` for the radius of `bound`, and `s`; `None`
/// where floats cannot follow them: a Gram–Schmidt norm below `2⁻⁴⁰` of the radius, which
/// rounding in the radius could swamp.
fn float_levels(levels: &Levels, bound: &Number) -> Option<(FloatLevels, i64)> {
    let g = levels.gso;
    let (lo, hi) = (levels.lo, levels.hi);
    let n = hi - lo;
    let (num, den) = radius_of(levels, bound);
    let s = if num.is_positive() {
        log2_ratio(&num, &den)
    } else {
        0
    };
    let radius = float_radius(levels, bound, s);
    let floor = radius * f64::from_bits((1023 - 40) << 52);
    let top = f64::from_bits((1023 + 1000) << 52);
    let mut r = Vec::with_capacity(n);
    for l in lo..hi {
        let (m, e) = ratio(&g.d[l + 1], &g.d[l]);
        let v = scale(m, e - s);
        // Below 2^-1000 an f64 may be subnormal, and its relative error unbounded.
        let low = if v < 1.0e-301 {
            0.0
        } else {
            (v * (1.0 - 2.0 * f64::EPSILON)).min(top)
        };
        if !(low >= floor && low > 0.0) {
            return None;
        }
        r.push(low);
    }
    let quotient = |a: &Number, b: &Number| -> Option<f64> {
        let (m, e) = ratio(a, b);
        let v = scale(m, e);
        if !v.is_finite() {
            return None;
        }
        Some(if v.abs() < TINY { 0.0 } else { v })
    };
    let mut mu = vec![0.0f64; n * n];
    for j in 0..n {
        for l in 0..j {
            mu[j * n + l] = quotient(&g.lam[lo + j][lo + l], &g.d[lo + l + 1])?;
        }
    }
    let t = match &levels.target {
        Some(t) => (lo..hi)
            .map(|l| quotient(&t.lam[l], &g.d[l + 1]))
            .collect::<Option<Vec<f64>>>()?,
        None => vec![0.0; n],
    };
    let mmax = FloatLevels::column_maxima(n, &mu);
    Some((
        FloatLevels {
            n,
            mu,
            r,
            t,
            mmax,
            symmetric: levels.target.is_none(),
        },
        s,
    ))
}

fn run(levels: &Levels, bound: &Number, least: bool) -> Vec<(Vec<i64>, Number)> {
    let levels = trim(levels, bound);
    if levels.hi == levels.lo || radius_of(&levels, bound).0.is_negative() {
        return Vec::new();
    }
    let Some((fl, s)) = float_levels(&levels, bound) else {
        return enum_exact::search(&levels, bound, least);
    };
    let mut found: Vec<(Vec<i64>, Number)> = Vec::new();
    let mut current = bound.clone();
    let outcome = fl.run(float_radius(&levels, &current, s), &mut |x: &[f64]| {
        let xi: Vec<i64> = x.iter().map(|&v| v as i64).collect();
        let v = enum_exact::value(&levels, &xi);
        if v > current {
            return None;
        }
        found.push((xi, v));
        if least {
            current = &found[found.len() - 1].1 - 1u8;
            return Some(float_radius(&levels, &current, s));
        }
        None
    });
    match outcome {
        Ok(()) => found,
        // A coefficient beyond 2⁵⁰: the rest exactly, from the bound reached.
        Err(_) if least => {
            found.extend(enum_exact::search(&levels, &current, true));
            found
        }
        Err(_) => enum_exact::search(&levels, bound, false),
    }
}
