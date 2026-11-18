//! `J₀`, `J₁`, `Y₀` and `Y₁` of an `f64`: below about 25 by expansions about each zero (so that
//! the error stays relative near the zeros), beyond by the modulus and the phase
//! (`J = M cos θ`, `Y = M sin θ`) with the phase reduced in double-double.

use super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::kernels::{PIO4, estrin, estrin_at, ln_dd, mla, rem_pio2};
use super::tables01::*;
use core::f64::consts::FRAC_1_PI;

/// `2/π` and `√(2/π)` as double-doubles.
pub(crate) const TWO_OVER_PI: Dd = Dd {
    hi: core::f64::consts::FRAC_2_PI,
    lo: -3.935_735_335_036_497e-17,
};
const SQRT_2_OVER_PI: Dd = Dd {
    hi: 0.797_884_560_802_865_4,
    lo: -4.984_654_404_555_46e-17,
};
/// `3π/4`.
const PI3O4: Dd = Dd {
    hi: 2.356_194_490_192_345,
    lo: 9.184_850_993_605_148e-17,
};

/// The first extremum of `Y₀` and of `Y₁` (where their first expansion ends).
const Y0_E1: f64 = 2.197_141_326_031_017;
const Y1_E1: f64 = 3.683_022_856_585_177_7;

/// `Σ c[i]·d^i` by Horner's rule.
#[inline(always)]
pub(crate) fn horner(d: f64, c: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for &k in c.iter().rev() {
        acc = acc.mul_add(d, k);
    }
    acc
}

/// The interval `k` with `b[k] ≤ x < b[k + 1]` of bounds spaced a little more than `π` apart.
#[inline(always)]
fn interval(x: f64, b: &[f64]) -> usize {
    let n = b.len() - 1;
    let mut k = (((x - b[1]) * FRAC_1_PI) as isize + 1).clamp(0, n as isize - 1) as usize;
    if x < b[k] {
        k -= 1;
    }
    k
}

/// `c₀ + c₁d + c₂d² + …` for `d = dh + dl`: Horner's rule in `f64` but for the last two steps,
/// which are in double-double (`c₀`, `c₁` double-doubles).
#[inline(always)]
pub(crate) fn poly_dd(dh: f64, dl: f64, c0: Dd, c1: Dd, rest: &[f64]) -> Dd {
    let d = dh + dl;
    let h = horner(d, rest);
    let (p, pe) = two_prod(d, h);
    let (q1h, q1l) = two_sum(c1.hi, p);
    let q1l = q1l + (pe + c1.lo);
    let (p, pe) = two_prod(dh, q1h);
    let (q0h, q0l) = two_sum(c0.hi, p);
    let q0l = q0l + (pe + c0.lo + dh.mul_add(q1l, dl * q1h));
    Dd { hi: q0h, lo: q0l }
}

/// `f(x)` from the expansion `f(z + d) = d·(c₀ + c₁d + …)` about the zero `z` (double-double),
/// `c = [c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …]`, by Horner's rule with its last two steps in
/// double-double (for wide intervals, as the Airy functions').
#[inline(always)]
pub(crate) fn about_zero(x: f64, z: &[f64; 2], c: &[f64]) -> f64 {
    let dh = x - z[0];
    let zl = z[1];
    let q = poly_dd(dh, -zl, dd2(c, 0), dd2(c, 2), &c[4..]);
    dh.mul_add(q.hi, dh.mul_add(q.lo, -zl * q.hi))
}

/// `f(x)` on a piece `[centre hi, centre lo, c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …]`: about a zero
/// (`zero`), `f(z + d) = d·(c₀ + d·(c₁ + d·P(d)))`, else `f(m + u) = c₀ + c₁u + u²·P(u)`; `P` by
/// Estrin's scheme, the leading terms exactly, and about a zero the product with the exact
/// `x − z_hi`.
#[inline(always)]
fn piece_eval<const W: usize>(x: f64, r: &[f64; W], zero: bool) -> f64 {
    let dh = x - r[0];
    let d = dh - r[1];
    if zero {
        let t = mla(d, mla(d, estrin_at(d, r, 6), r[4]), r[3]);
        // (dh − zl)·(c₀ + t) = dh·c₀ (exact) − zl·c₀ + d·t.
        let (ph, pl) = two_prod(dh, r[2]);
        ph + (pl + mla(d, t, -r[1] * r[2]))
    } else {
        // c₀ + c₁u + u²·P(u), the first two terms exactly.
        let t2 = (d * d) * estrin_at(d, r, 6);
        let (p, pe) = two_prod(d, r[4]);
        let (s, se) = two_sum(r[2], p);
        s + (se + (pe + mla(d, r[5], r[3] + t2)))
    }
}

/// `f(x)` from the tables of an order: the interval by `b`, its piece by `pb`.
#[inline(always)]
fn by_pieces<const W: usize>(x: f64, b: &[f64], pb: &[f64], c: &[[f64; W]]) -> f64 {
    let k = interval(x, b);
    let j = (x >= pb[3 * k + 1]) as usize + (x >= pb[3 * k + 2]) as usize;
    piece_eval(x, &c[3 * k + j], j == 1)
}

/// The double-double `c[i] + c[i + 1]`.
#[inline(always)]
pub(crate) fn dd2(c: &[f64], i: usize) -> Dd {
    Dd {
        hi: c[i],
        lo: c[i + 1],
    }
}

/// `ln(x/z)/(x − z)` as a double-double, `d = x − z = dh − zl`.
#[inline(always)]
fn log_ratio(x: f64, z: &[f64; 2], lnz: &[f64; 2], dh: f64) -> Dd {
    let zl = z[1];
    let d = dh - zl;
    let dd = {
        let (h, l) = fast_two_sum(dh, -zl);
        Dd { hi: h, lo: l }
    };
    if d.abs() < 0.0625 * z[0] {
        // ln(1 + u)/u = 1 − u/2 + u²/3 − …, u = d/z, |u| < 1/16.
        let u = d / z[0];
        let mut s = 0.0f64;
        for k in (1..=17).rev() {
            s = s.mul_add(u, if k % 2 == 1 { -1.0 } else { 1.0 } / (k + 1) as f64);
        }
        let (h, l) = fast_two_sum(1.0, u * s);
        Dd { hi: h, lo: l }.div(Dd { hi: z[0], lo: zl })
    } else {
        ln_dd(x)
            .sub(Dd {
                hi: lnz[0],
                lo: lnz[1],
            })
            .div(dd)
    }
}

/// `Y_n(x) = d·((2/π)·λ·J_n(x) + g(d)[/x])`, `λ = ln(x/z)/d`, about one of the first two zeros.
#[inline(always)]
fn y_log_form<const W: usize>(
    x: f64,
    z: &[f64; 2],
    lnz: &[f64; 2],
    c: &[f64; W],
    jn: f64,
    over_x: bool,
) -> f64 {
    let dh = x - z[0];
    let zl = z[1];
    let d = dh - zl;
    let lam = log_ratio(x, z, lnz, dh);
    // g = c₀ + c₁d + d²·P(d), the first two terms in double-double: Estrin's scheme for Y₀,
    // Horner's rule for Y₁ (whose wider expansions lose up to two ulps by Estrin's).
    let g = if over_x {
        poly_dd(dh, -zl, dd2(c, 0), dd2(c, 2), &c[4..])
    } else {
        let t2 = (d * d) * estrin_at(d, c, 4);
        let (p, pe) = two_prod(dh, c[2]);
        let (s, se) = two_sum(c[0], p);
        Dd {
            hi: s,
            lo: se + (pe + (c[1] + mla(dh, c[3], mla(-zl, c[2], t2)))),
        }
    };
    let g = if over_x {
        // g/x by the reciprocal and one correction.
        let inv = 1.0 / x;
        let (q, qe) = two_prod(x, inv);
        let e = (1.0 - q) - qe;
        let h = g.hi * inv;
        Dd {
            hi: h,
            lo: mla(g.hi, inv, -h) + (g.lo * inv + h * e),
        }
    } else {
        g
    };
    let b = lam.mul(TWO_OVER_PI).mul_f64(jn).add(g);
    dh.mul_add(b.hi, dh.mul_add(b.lo, -zl * b.hi))
}

/// The reduced phase `u` and its quadrant `n` (`θ = nπ/2 + u`) and the modulus `M` as a
/// double-double, for `x ≥ 24`, `n` 0 or 1.
#[inline(always)]
fn modulus_phase(x: f64, one: bool) -> (i64, Dd, f64, f64) {
    let (m, a, a0, phase) = if one {
        (&M1, &A1, 0.375, PI3O4)
    } else {
        (&M0, &A0, -0.125, PIO4)
    };
    // t + tl = 1/x.
    let t = 1.0 / x;
    let (p, e) = two_prod(x, t);
    let tl = ((1.0 - p) - e) * t;
    let s = t * t;
    let mm = s * estrin(s, m);
    // α = a₀t + t³·A(t²), the first term in double-double.
    let (ah, al) = two_prod(a0, t);
    let al = al + mla(a0, tl, t * s * estrin(s, a));
    let phi = Dd { hi: ah, lo: al }.sub(phase);
    let (n, r) = rem_pio2(x);
    let u = r.add(phi);
    let k = (u.hi * core::f64::consts::FRAC_2_PI).round_ties_even();
    let u = u.sub(super::kernels::PIO2.mul_f64(k));
    // M = √(2/π)·(1 + mm)/√x: √x = r + rl, 1/√x = inv + il.
    let r = x.sqrt();
    let (p, e) = two_prod(r, r);
    let rl = ((x - p) - e) * (0.5 / r);
    let inv = 1.0 / r;
    let (p, e) = two_prod(r, inv);
    let il = inv * (((1.0 - p) - e) - rl * inv);
    let (mh, ml) = two_prod(SQRT_2_OVER_PI.hi, inv);
    let ml = ml + (SQRT_2_OVER_PI.hi * il + SQRT_2_OVER_PI.lo * inv + mh * mm);
    (n.wrapping_add(k as i64), u, mh, ml)
}

/// `M·v` for a double-double `v`, rounded once.
#[inline(always)]
fn times(mh: f64, ml: f64, v: Dd) -> f64 {
    let (p, e) = two_prod(mh, v.hi);
    p + (e + mla(mh, v.lo, ml * v.hi))
}

/// The modulus-phase form for `x ≥ 24`: `J_n(x)` (`want_y` false) or `Y_n(x)`.
#[inline(always)]
fn asymptotic1(x: f64, one: bool, want_y: bool) -> f64 {
    let (n, u, mh, ml) = modulus_phase(x, one);
    times(mh, ml, super::kernels::sin_or_cos_quadrant_dd(n, u, want_y))
}

/// The modulus-phase form for `x ≥ 24`: `(J_n(x), Y_n(x))`.
#[inline(always)]
fn asymptotic(x: f64, one: bool) -> (f64, f64) {
    let (n, u, mh, ml) = modulus_phase(x, one);
    let (s_, c_) = super::kernels::sin_cos_quadrant_dd(n, u);
    (times(mh, ml, c_), times(mh, ml, s_))
}

/// `J₀(x)`.
pub(crate) fn j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < J0_B[0] {
        let (sh, sl) = two_prod(ax, ax);
        return poly_dd(sh, sl, Dd::new(1.0), dd2(&J0_SMALL, 0), &J0_SMALL[2..]).value();
    }
    if ax < J0_B[J0_B.len() - 1] {
        return by_pieces(ax, &J0_B, &J0_PB, &J0_C);
    }
    if ax.is_nan() {
        return x;
    }
    if ax.is_infinite() {
        return 0.0;
    }
    asymptotic1(ax, false, false)
}

/// `J₁(x)`.
pub(crate) fn j1(x: f64) -> f64 {
    let ax = x.abs();
    let v = if ax < J1_B[0] {
        let (sh, sl) = two_prod(ax, ax);
        let q = poly_dd(sh, sl, Dd::new(0.5), dd2(&J1_SMALL, 0), &J1_SMALL[2..]);
        ax.mul_add(q.hi, ax * q.lo)
    } else if ax < J1_B[J1_B.len() - 1] {
        by_pieces(ax, &J1_B, &J1_PB, &J1_C)
    } else if ax.is_nan() {
        return x;
    } else if ax.is_infinite() {
        0.0
    } else {
        asymptotic1(ax, true, false)
    };
    if x < 0.0 { -v } else { v }
}

/// `Y₀(x)` (NaN for `x < 0`, `−∞` at 0).
pub(crate) fn y0(x: f64) -> f64 {
    if x < Y0_B[0] {
        if x <= 0.0 {
            return if x == 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if x < Y0_SMALL_X {
            // (2/π)·ln x·J₀(x) + R₀(x²).
            let l = ln_dd(x).mul(TWO_OVER_PI).mul_f64(j0(x));
            let (sh, sl) = two_prod(x, x);
            return l
                .add(poly_dd(sh, sl, dd2(&Y0_R, 0), dd2(&Y0_R, 2), &Y0_R[4..]))
                .value();
        }
        let k = (x >= Y0_E1) as usize;
        return y_log_form(x, &Y0_LZ[k], &Y0_LLN[k], &Y0_LC[k], j0(x), false);
    }
    if x < Y0_B[Y0_B.len() - 1] {
        return by_pieces(x, &Y0_B, &Y0_PB, &Y0_C);
    }
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    asymptotic1(x, false, true)
}

/// `Y₁(x)` (NaN for `x < 0`, `−∞` at 0).
pub(crate) fn y1(x: f64) -> f64 {
    if x < Y1_B[0] {
        if x <= 0.0 {
            return if x == 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if x < 1e-300 {
            // −2/(πx), which overflows below about 3.5·10⁻³⁰⁹.
            return -TWO_OVER_PI.hi / x;
        }
        if x < Y1_SMALL_X {
            // (2/π)·(ln x·J₁(x) − 1/x) + x·R₁(x²).
            let inv = Dd::new(1.0).div(Dd::new(x));
            let l = ln_dd(x).mul_f64(j1(x)).sub(inv).mul(TWO_OVER_PI);
            let (sh, sl) = two_prod(x, x);
            let r = poly_dd(sh, sl, dd2(&Y1_R, 0), dd2(&Y1_R, 2), &Y1_R[4..]).mul_f64(x);
            return l.add(r).value();
        }
        let k = (x >= Y1_E1) as usize;
        return y_log_form(x, &Y1_LZ[k], &Y1_LLN[k], &Y1_LC[k], j1(x), true);
    }
    if x < Y1_B[Y1_B.len() - 1] {
        return by_pieces(x, &Y1_B, &Y1_PB, &Y1_C);
    }
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    asymptotic1(x, true, true)
}

/// `(J_n(x), Y_n(x))` for `n` 0 or 1 and `x ≥ 24`, together (one reduction).
pub(crate) fn jy01_large(x: f64, one: bool) -> (f64, f64) {
    asymptotic(x, one)
}
