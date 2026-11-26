//! The Airy functions and their derivatives, in `f64` (an `f32` through it, rounded once):
//!
//! * `x ≥ 4` (`Bi`, `Bi′`: `x ≥ 12`): `Ai = e^(−ζ)/(2√π·x^(1/4))·(1 + t·A(t))`, `t = 1/ζ`, `ζ = (2/3)·x^(3/2)` in
//!   double-double (`Ai′`, `Bi`, `Bi′` alike), `A` a Remez fit;
//! * from the first extremum left of 0 up to 1 Remez fits in pieces, from 1 to 4 (12) fits of
//!   `e^(±ζ)·f` in pieces;
//! * from there down to about −8 expansions about each zero, `f(z + d) = d·q(d)` with the zero
//!   in double-double, so that the error stays relative near the zeros;
//! * below: the modulus and the phase, `Ai(−y) = M cos θ`, `Bi(−y) = M sin θ`,
//!   `θ = π/4 − ζ + α(1/ζ)` (and `N`, `φ = 3π/4 − ζ + β` for the derivatives), the phase reduced in
//!   double-double.

mod tables;

use super::Real;
use super::bessel::kernels::{
    PIO2, PIO4, exp_parts, ldexp, rem_pio2, sin_cos_quadrant_dd, sin_cos_shifted_dd,
};
use super::bessel::{about_zero, dd2, horner, poly_dd3};
use super::dd::{Dd, fast_two_sum, two_sum};
use super::via;
use tables::*;

const SQRT_PI: Dd = Dd {
    hi: 1.772_453_850_905_516,
    lo: -7.666_586_499_825_799e-17,
};
const TWO_THIRDS: Dd = Dd {
    hi: 0.666_666_666_666_666_6,
    lo: 3.700_743_415_417_188e-17,
};

/// Which of the four.
#[derive(Clone, Copy, PartialEq)]
enum Which {
    Ai,
    AiP,
    Bi,
    BiP,
}

/// `(ζ, y^(1/4))` of `y > 0` as double-doubles.
#[inline]
fn zeta_parts(y: f64) -> (Dd, Dd) {
    let s = Dd::new(y).sqrt();
    (TWO_THIRDS.mul(s.mul_f64(y)), s.sqrt())
}

/// The piece `j` with `b[j] ≤ x < b[j + 1]`.
#[inline(always)]
fn piece(x: f64, b: &[f64]) -> usize {
    let mut j = 0;
    for (i, &c) in b[1..b.len() - 1].iter().enumerate() {
        if x >= c {
            j = i + 1;
        }
    }
    j
}

/// `Σ cᵢ·uⁱ` of a row `[x₀, c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …]`, `u = x − x₀` exactly (as a
/// double-double: `x` and `x₀` may differ by more than a factor of 2), the last three steps of
/// Horner's rule in double-double (`c₂u²` reaches the size of the value at the ends).
#[inline(always)]
fn piece_poly(x: f64, row: &[f64]) -> Dd {
    let (uh, ul) = two_sum(x, -row[0]);
    poly_dd3(uh, ul, dd2(row, 1), dd2(row, 3), row[5], &row[6..])
}

/// The interval `k` with `zb[k + 1] ≤ x < zb[k]` (bounds from the right, going down).
#[inline(always)]
fn zero_interval(x: f64, zb: &[f64]) -> usize {
    let mut k = 0;
    for (i, &c) in zb[1..zb.len() - 1].iter().enumerate() {
        if x < c {
            k = i + 1;
        }
    }
    k
}

/// `x ≥ 4`.
fn positive(x: f64, w: Which) -> f64 {
    if x > 1.0e5 {
        return match w {
            Which::Ai => 0.0,
            Which::AiP => -0.0,
            _ => f64::INFINITY,
        };
    }
    let (z, q) = zeta_parts(x);
    if z.hi > 1.0e6 {
        return match w {
            Which::Ai => 0.0,
            Which::AiP => -0.0,
            _ => f64::INFINITY,
        };
    }
    let t = 1.0 / z.hi;
    let a: &[f64] = match w {
        Which::Ai => &AI_A,
        Which::AiP => &AIP_A,
        Which::Bi => &BI_A,
        Which::BiP => &BIP_A,
    };
    // g = 1 + t·(c₀ + t·(c₁ + …)).
    let g = Dd::new(1.0).add(dd2(a, 0).add_f64(t * horner(t, &a[2..])).mul_f64(t));
    let (v, e) = match w {
        Which::Ai => (g.div(SQRT_PI.mul(q).mul_f64(2.0)), z.neg()),
        Which::AiP => (g.mul(q).div(SQRT_PI).mul_f64(-0.5), z.neg()),
        Which::Bi => (g.div(SQRT_PI.mul(q)), z),
        Which::BiP => (g.mul(q).div(SQRT_PI), z),
    };
    let (m, k) = exp_parts(e.hi, e.lo);
    ldexp(v.mul(m).value(), k)
}

/// `[start, 4)`: the plain pieces up to 1, the scaled ones beyond.
fn middle(x: f64, w: Which) -> f64 {
    let (rows, b): (&[[f64; P_N]], &[f64]) = match w {
        Which::Ai => (&AI_P, &AI_PB),
        Which::AiP => (&AIP_P, &AIP_PB),
        Which::Bi => (&BI_P, &BI_PB),
        Which::BiP => (&BIP_P, &BIP_PB),
    };
    let v = piece_poly(x, &rows[piece(x, b)]);
    if x < 1.0 {
        return v.value();
    }
    let (z, _) = zeta_parts(x);
    let e = if matches!(w, Which::Ai | Which::AiP) {
        z.neg()
    } else {
        z
    };
    let (m, k) = exp_parts(e.hi, e.lo);
    ldexp(v.mul(m).value(), k)
}

/// About the zeros, `x` in `[zb[last], start)`.
fn near_zeros(x: f64, w: Which) -> f64 {
    let (zb, z, c): (&[f64], &[[f64; 2]], &[[f64; C_N]]) = match w {
        Which::Ai => (&AI_ZB, &AI_Z, &AI_C),
        Which::AiP => (&AIP_ZB, &AIP_Z, &AIP_C),
        Which::Bi => (&BI_ZB, &BI_Z, &BI_C),
        Which::BiP => (&BIP_ZB, &BIP_Z, &BIP_C),
    };
    let k = zero_interval(x, zb);
    about_zero(x, &z[k], &c[k])
}

/// `x` below the zero tables: `y = −x` and the modulus and phase.
fn negative(x: f64, w: Which) -> f64 {
    let y = -x;
    let deriv = matches!(w, Which::AiP | Which::BiP);
    if y > 1.0e200 {
        // ζ near or beyond the overflow: no phase.
        return if deriv { f64::NAN } else { 0.0 };
    }
    let (z, q) = zeta_parts(y);
    if !z.hi.is_finite() {
        // ζ overflows (y beyond about 10²⁰⁵): no phase.
        return if deriv { f64::NAN } else { 0.0 };
    }
    let t = 1.0 / z.hi;
    let s2 = t * t;
    let (modt, ph): (&[f64], &[f64]) = if deriv {
        (&N_MOD, &N_PHASE)
    } else {
        (&M_MOD, &M_PHASE)
    };
    let mm = s2 * horner(s2, modt);
    let alpha = t * horner(s2, ph);
    let one = {
        let (h, l) = fast_two_sum(1.0, mm);
        Dd { hi: h, lo: l }
    };
    let amp = if deriv {
        one.mul(q).div(SQRT_PI)
    } else {
        one.div(SQRT_PI.mul(q))
    };
    // Ai(−y) = M cos(ζ − π/4 − α), Bi(−y) = −M sin(ζ − π/4 − α);
    // Ai′(−y) = N cos(ζ − 3π/4 − β), Bi′(−y) = −N sin(ζ − 3π/4 − β).
    let base = if deriv {
        PIO4.mul_f64(-3.0)
    } else {
        PIO4.neg()
    };
    let (s, c) = if z.lo.abs() < 0.785 {
        let phi = Dd::new(z.lo).add(base).add_f64(-alpha);
        sin_cos_shifted_dd(z.hi, phi)
    } else {
        far_phase(z, base, alpha)
    };
    match w {
        Which::Ai | Which::AiP => amp.mul(c).value(),
        _ => amp.mul(s).neg().value(),
    }
}

/// `(sin, cos)` of `ζ + base − α` for `|x|` from about `10¹¹` on, where the low part of `ζ` passes
/// `π/4` (and from about `6·10²¹` `2⁵³`): it is reduced modulo `π/2` as the high part is (left as it
/// was, the quadrant of the sum was inexact there and the kernels got arguments far beyond `π/4`,
/// for values beyond the modulus and from `1.7·10³⁶` NaN). The phase error stays that of `ζ`.
#[cold]
#[inline(never)]
fn far_phase(z: Dd, base: Dd, alpha: f64) -> (Dd, Dd) {
    let (n1, r1) = rem_pio2(z.hi);
    let (n2, r2) = rem_pio2(z.lo);
    let u = r1.add(r2).add(base).add_f64(-alpha);
    let k = (u.hi * core::f64::consts::FRAC_2_PI).round_ties_even();
    let u = u.sub(PIO2.mul_f64(k));
    sin_cos_quadrant_dd(n1.wrapping_add(n2).wrapping_add(k as i64), u)
}

#[inline]
fn airy_f64(x: f64, w: Which) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return match w {
            Which::Ai => 0.0,
            Which::AiP => -0.0,
            _ => f64::INFINITY,
        };
    }
    let xp = if matches!(w, Which::Bi | Which::BiP) {
        X_POS_B
    } else {
        X_POS
    };
    if x >= xp {
        return positive(x, w);
    }
    let (pb0, zbl) = match w {
        Which::Ai => (AI_PB[0], AI_ZB[AI_ZB.len() - 1]),
        Which::AiP => (AIP_PB[0], AIP_ZB[AIP_ZB.len() - 1]),
        Which::Bi => (BI_PB[0], BI_ZB[BI_ZB.len() - 1]),
        Which::BiP => (BIP_PB[0], BIP_ZB[BIP_ZB.len() - 1]),
    };
    if x >= pb0 {
        return middle(x, w);
    }
    if x >= zbl {
        return near_zeros(x, w);
    }
    if x == f64::NEG_INFINITY {
        return if matches!(w, Which::AiP | Which::BiP) {
            f64::NAN
        } else {
            0.0
        };
    }
    negative(x, w)
}

/// The Airy function `Ai(x)`.
pub fn airy_ai<T: Real>(x: T) -> T {
    via(x, |x| airy_f64(x, Which::Ai))
}

/// The derivative `Ai′(x)`.
pub fn airy_ai_prime<T: Real>(x: T) -> T {
    via(x, |x| airy_f64(x, Which::AiP))
}

/// The Airy function `Bi(x)`.
pub fn airy_bi<T: Real>(x: T) -> T {
    via(x, |x| airy_f64(x, Which::Bi))
}

/// The derivative `Bi′(x)`.
pub fn airy_bi_prime<T: Real>(x: T) -> T {
    via(x, |x| airy_f64(x, Which::BiP))
}
