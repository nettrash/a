//! `I₀`, `I₁`, `K₀` and `K₁` of an `f64`, plain and scaled: polynomials in `x²` for small `x`
//! (with the logarithm of `K` apart), `I` in pieces of `[1, 7.75]`, and `e^(∓x)·√x·(I, K)` as
//! polynomials in pieces of `1/x`.

use super::dd::{Dd, two_prod, two_sum};
use super::int01::{dd2, poly_dd};
use super::kernels::{estrin_at, exp_parts_lite, ldexp, ln_dd, mla};
use super::tables_ik::*;

/// The piece `j` of `cuts` with `cuts[j] ≤ t < cuts[j + 1]` (the last piece closed).
#[inline(always)]
fn piece(t: f64, cuts: &[f64]) -> usize {
    let mut j = 0;
    for (i, &c) in cuts[1..cuts.len() - 1].iter().enumerate() {
        if t >= c {
            j = i + 1;
        }
    }
    j
}

/// `Σ cᵢ·uⁱ` of a row `[t₀, c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …]`, `u = t − t₀`: Estrin's scheme,
/// the first two terms exactly (an unnormalised double-double).
#[inline(always)]
fn piece_poly<const W: usize>(t: f64, row: &[f64; W]) -> Dd {
    let u = t - row[0];
    let t2 = (u * u) * estrin_at(u, row, 5);
    let (p, pe) = two_prod(u, row[3]);
    let (s, se) = two_sum(row[1], p);
    Dd {
        hi: s,
        lo: se + (pe + mla(u, row[4], row[2] + t2)),
    }
}

/// `v·e^e` of a double-double `v`, rounded once (to ∞ or the subnormals as it falls).
#[inline(always)]
fn times_exp(v: Dd, e: f64) -> f64 {
    let (m, k) = exp_parts_lite(e);
    let (p, pe) = two_prod(v.hi, m.hi);
    ldexp(p + (pe + mla(v.hi, m.lo, v.lo * m.hi)), k)
}

/// `q/√x` as an unnormalised double-double, `t = 1/x` rounded: `1/√x = inv + il` from
/// `√x = r + rl`.
#[inline(always)]
fn over_sqrt(q: Dd, x: f64, t: f64) -> Dd {
    let r = x.sqrt();
    let (p, e) = two_prod(r, r);
    let rl = ((x - p) - e) * (0.5 * t) * r;
    // 1/√x ≈ √x·(1/x), corrected below.
    let inv = r * t;
    let (p, e) = two_prod(r, inv);
    let il = inv * (((1.0 - p) - e) - rl * inv);
    let (h, l) = two_prod(q.hi, inv);
    Dd {
        hi: h,
        lo: l + mla(q.hi, il, q.lo * inv),
    }
}

/// `I₀(x)` for `0 ≤ x < 1` as a double-double.
#[inline(always)]
fn i0_small(x: f64) -> Dd {
    let (sh, sl) = two_prod(x, x);
    poly_dd(sh, sl, Dd::new(1.0), dd2(&I0_SMALL, 0), &I0_SMALL[2..])
}

/// `I₁(x)` for `0 ≤ x < 1` as a double-double.
#[inline(always)]
fn i1_small(x: f64) -> Dd {
    let (sh, sl) = two_prod(x, x);
    poly_dd(sh, sl, Dd::new(0.5), dd2(&I1_SMALL, 0), &I1_SMALL[2..]).mul_f64(x)
}

/// `I₀(x)`, or `e^(−|x|)·I₀(x)` when `scaled`.
pub(crate) fn i0(x: f64, scaled: bool) -> f64 {
    let ax = x.abs();
    if ax < 1.0 {
        let v = i0_small(ax);
        return if scaled { times_exp(v, -ax) } else { v.value() };
    }
    if ax < 7.75 {
        let v = piece_poly(ax, &I0_MID[piece(ax, &I_MID_CUTS)]);
        return if scaled { times_exp(v, -ax) } else { v.value() };
    }
    if ax.is_nan() {
        return x;
    }
    if ax.is_infinite() {
        return if scaled { 0.0 } else { f64::INFINITY };
    }
    let t = 1.0 / ax;
    let v = over_sqrt(piece_poly(t, &I0_LARGE[piece(t, &I_CUTS)]), ax, t);
    if scaled {
        v.value()
    } else if ax > 720.0 {
        f64::INFINITY
    } else {
        times_exp(v, ax)
    }
}

/// `I₁(x)`, or `e^(−|x|)·I₁(x)` when `scaled`.
pub(crate) fn i1(x: f64, scaled: bool) -> f64 {
    let ax = x.abs();
    let v = if ax < 1.0 {
        let q = i1_small(ax);
        if scaled { times_exp(q, -ax) } else { q.value() }
    } else if ax < 7.75 {
        let v = piece_poly(ax, &I1_MID[piece(ax, &I_MID_CUTS)]);
        if scaled { times_exp(v, -ax) } else { v.value() }
    } else if ax.is_nan() {
        return x;
    } else if ax.is_infinite() {
        if scaled { 0.0 } else { f64::INFINITY }
    } else {
        let t = 1.0 / ax;
        let v = over_sqrt(piece_poly(t, &I1_LARGE[piece(t, &I_CUTS)]), ax, t);
        if scaled {
            v.value()
        } else if ax > 720.0 {
            f64::INFINITY
        } else {
            times_exp(v, ax)
        }
    };
    if x < 0.0 { -v } else { v }
}

/// `K₀(x)`, or `e^x·K₀(x)` when `scaled` (NaN for `x < 0`, `∞` at 0).
pub(crate) fn k0(x: f64, scaled: bool) -> f64 {
    if x <= 1.0 {
        if x <= 0.0 {
            return if x == 0.0 { f64::INFINITY } else { f64::NAN };
        }
        // R₀(x²) − ln x·I₀(x).
        let l = ln_dd(x).mul(i0_small(x)).neg();
        let (sh, sl) = two_prod(x, x);
        let v = l.add(poly_dd(
            sh,
            sl,
            dd2(&K0_SMALL, 0),
            dd2(&K0_SMALL, 2),
            &K0_SMALL[4..],
        ));
        return if scaled { times_exp(v, x) } else { v.value() };
    }
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let t = 1.0 / x;
    let v = over_sqrt(piece_poly(t, &K0_LARGE[piece(t, &K_CUTS)]), x, t);
    if scaled {
        v.value()
    } else if x > 750.0 {
        0.0
    } else {
        times_exp(v, -x)
    }
}

/// `K₁(x)`, or `e^x·K₁(x)` when `scaled` (NaN for `x < 0`, `∞` at 0).
pub(crate) fn k1(x: f64, scaled: bool) -> f64 {
    if x <= 1.0 {
        if x <= 0.0 {
            return if x == 0.0 { f64::INFINITY } else { f64::NAN };
        }
        if x < 1e-300 {
            return 1.0 / x;
        }
        // 1/x + ln x·I₁(x) + x·R₁(x²).
        let inv = Dd::new(1.0).div(Dd::new(x));
        let v = inv.add(ln_dd(x).mul(i1_small(x))).add({
            let (sh, sl) = two_prod(x, x);
            poly_dd(sh, sl, dd2(&K1_SMALL, 0), dd2(&K1_SMALL, 2), &K1_SMALL[4..]).mul_f64(x)
        });
        return if scaled { times_exp(v, x) } else { v.value() };
    }
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let t = 1.0 / x;
    let v = over_sqrt(piece_poly(t, &K1_LARGE[piece(t, &K_CUTS)]), x, t);
    if scaled {
        v.value()
    } else if x > 750.0 {
        0.0
    } else {
        times_exp(v, -x)
    }
}
