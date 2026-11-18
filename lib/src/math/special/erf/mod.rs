//! The error functions, their inverses, the Faddeeva function, the Fresnel integrals and
//! Dawson's integral.
//!
//! The real functions share a few pieces, all with a double-double head so that the final
//! rounding is the only large error:
//!
//! * `|x| < 1/2`: `erf(x) = x + x·q(x²)`, a minimax polynomial of degree 9;
//! * `1/2 ≤ x < 8`: `erfcx(x)` from 32 polynomials of degree 10, one per eighth of a binade,
//!   found from the bits of `x`;
//! * `x ≥ 8`: `x·erfcx(x)` as a polynomial of degree 9 in `1/x²`;
//! * `e^a` of a double-double `a` to about 2⁻⁶⁰ ([`exp_dd`]), so that
//!   `erfc(x) = e^(−x²)·erfcx(x)` loses nothing to the exponential.
//!
//! Dawson's integral and the Fresnel auxiliary functions use the same pieces. The coefficients
//! are this area's own minimax fits (`tables.rs`, made by `gen_tables.py` with mpmath).

mod complex;
mod tables;

use super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::{Real, via};
use crate::math::complex::Complex;
use core::f64::consts::LN_2;
use tables::*;

pub(crate) use complex::{erf_c64, erfc_c64, erfcx_c64};

/// `√π/2` (the nearest double).
const SQRT_PI_2: f64 = 0.886_226_925_452_758;
/// `√π/2 − SQRT_PI_2`.
const SQRT_PI_2_LO: f64 = -3.833_293_249_912_899_3e-17;

/// `a·b + c`, fused where the target has a fused multiply-add and as two operations elsewhere
/// (for the polynomials only; the exact products of [`two_prod`] use `mul_add` always).
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(any(
        target_arch = "aarch64",
        target_arch = "s390x",
        target_feature = "fma"
    )) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `c[0] + c[1]·x + … ` by Horner's rule.
#[inline(always)]
fn poly(x: f64, c: &[f64]) -> f64 {
    let n = c.len();
    let mut s = c[n - 1];
    let mut i = n - 1;
    while i > 0 {
        i -= 1;
        s = fma(s, x, c[i]);
    }
    s
}

/// `2^e` for `−1022 ≤ e ≤ 1023`.
#[inline(always)]
fn pow2(e: i64) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `v·2^e` for `|e| ≤ 2044`, rounded once (in two exact-or-final steps, so that an overflow is
/// `∞` and a subnormal result is rounded once).
#[inline(always)]
fn scale(v: f64, e: i64) -> f64 {
    let e1 = e >> 1;
    v * pow2(e1) * pow2(e - e1)
}

/// `(p.hi + p.lo)·2^k` rounded once, also where it is subnormal (`|k| ≤ 2044`, `p` normalised).
#[inline(always)]
fn scale_dd(p: Dd, k: i64) -> f64 {
    let s = scale(p.hi, k);
    if s.abs() >= f64::MIN_POSITIVE || s == 0.0 && p.hi == 0.0 {
        return scale(p.hi + p.lo, k);
    }
    // s is p.hi on the subnormal grid; add the exact remainder, rounded onto the grid.
    let d = (p.hi - scale(s, -k)) + p.lo;
    s + scale(d, k)
}

/// `1.5·2⁵²`: adding and subtracting it rounds to an integer (ties to even) any `|v| < 2⁵¹`.
const ROUND: f64 = 6_755_399_441_055_744.0;

/// `e^(a_hi + a_lo) = m·2^k` with `m` a double-double in `[0.98, 2.03)`, to about 2⁻⁶⁰, for
/// `|a_hi| < 2⁹·ln 2·2⁸` (far beyond the needs here: `|a| ≤ 750`).
#[inline(always)]
fn exp_dd(a_hi: f64, a_lo: f64) -> (Dd, i64) {
    const INV: f64 = 64.0 / LN_2;
    // ln 2/64 = L1 + L2, L1 with 35 significant bits so that k·L1 is exact for |k| < 2¹⁷.
    const L1: f64 = 0.010_830_424_696_450_791;
    const L2: f64 = -2.016_456_292_199_553_7e-13;
    let z = a_hi * INV + ROUND;
    let k = (z.to_bits() as i64).wrapping_sub(ROUND.to_bits() as i64);
    let kf = z - ROUND;
    let r = fma(-kf, L2, fma(-kf, L1, a_hi)) + a_lo;
    // e^r − 1, |r| ≤ ln 2/128 + 2⁻⁴⁰: Taylor to r⁶ (error < 2⁻⁶⁵).
    let p = fma(
        r * r,
        poly(r, &[0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0]),
        r,
    );
    let t = EXP2_64[(k & 63) as usize];
    let (hi, lo) = fast_two_sum(t[0], fma(t[0], p, t[1]));
    (Dd { hi, lo }, k >> 6)
}

/// `q(x²)` with `erf(x) = x + x·q(x²)` for `|x| < 1/2`.
#[inline(always)]
fn erf_q(t: f64) -> f64 {
    poly(t, &ERF_SMALL)
}

/// The value `c0 + r·(c1 + c2·r + …)` of a table row `[c0 hi, c0 lo, c1, …]` as a double-double.
#[inline(always)]
fn head(row: &[f64], r: f64) -> Dd {
    let q = r * poly(r, &row[2..]);
    let (hi, e) = fast_two_sum(row[0], q);
    Dd { hi, lo: e + row[1] }
}

/// The piece of a table of eighths of binades from `2^b0` that holds `x`, and `x` minus its
/// centre (exact).
#[inline(always)]
fn piece(x: f64, b0: u64) -> (usize, f64) {
    let b = x.to_bits() >> 49;
    let c = f64::from_bits((b << 49) | (1 << 48));
    ((b - b0) as usize, x - c)
}

/// `bits(0.5) >> 49` and `bits(1.0) >> 49`.
const B_HALF: u64 = 0x3FE0_0000_0000_0000 >> 49;
const B_ONE: u64 = 0x3FF0_0000_0000_0000 >> 49;

/// `1/x` as a double-double, for a finite `x ≠ 0` (`|x| < 2¹⁰²²`).
#[inline(always)]
fn recip_dd(x: f64) -> Dd {
    let hi = 1.0 / x;
    Dd {
        hi,
        lo: hi * (-x).mul_add(hi, 1.0),
    }
}

/// `a·b` of two double-doubles, not renormalised.
#[inline(always)]
fn mul_dd(a: Dd, b: Dd) -> Dd {
    let (h, l) = two_prod(a.hi, b.hi);
    Dd {
        hi: h,
        lo: l + fma(a.hi, b.lo, a.lo * b.hi),
    }
}

/// `erfcx(x)` for `1/2 ≤ x < ∞`, as a double-double.
#[inline(always)]
fn erfcx_pos(x: f64) -> Dd {
    if x < 8.0 {
        let (i, r) = piece(x, B_HALF);
        head(&ERFCX_T[i], r)
    } else {
        // x·erfcx(x) = P(1/x²).
        let ix = recip_dd(x);
        let u = ix.hi * ix.hi;
        mul_dd(ix, head(&ERFCX_ASYM, u))
    }
}

/// `erfc(x) = m·2^k` for `1/2 ≤ x ≤ 27.3`, `m` a double-double.
#[inline(always)]
fn erfc_pos(x: f64) -> (Dd, i64) {
    let (sh, sl) = two_prod(x, x);
    let (e, k) = exp_dd(-sh, -sl);
    (mul_dd(e, erfcx_pos(x)), k)
}

/// `erf(x)` of an `f64`.
pub(crate) fn erf_f64(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        return x.mul_add(erf_q(x * x), x);
    }
    if a.is_nan() || a >= 6.0 {
        // erfc(6) < 2⁻⁵⁴: erf is ±1 to the last bit.
        return if a.is_nan() { x } else { 1.0f64.copysign(x) };
    }
    let (p, k) = erfc_pos(a);
    let (ph, pl) = (scale(p.hi, k), scale(p.lo, k));
    let (s, e) = two_sum(1.0, -ph);
    (s + (e - pl)).copysign(x)
}

/// `erfc(x)` of an `f64`.
pub(crate) fn erfc_f64(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        // (1 − x) − x·q, with 1 − x exact as two doubles.
        let (s, e) = two_sum(1.0, -x);
        return s + (-x).mul_add(erf_q(x * x), e);
    }
    if x > 0.0 {
        if x >= 27.3 {
            return 0.0;
        }
        let (p, k) = erfc_pos(x);
        let (hi, lo) = fast_two_sum(p.hi, p.lo);
        return scale_dd(Dd { hi, lo }, k);
    }
    if x.is_nan() {
        return x;
    }
    // 2 − erfc(|x|); erfc(6) < 2⁻⁵³, so 2 from there on.
    let (p, k) = erfc_pos(a.min(6.0));
    let (ph, pl) = (scale(p.hi, k), scale(p.lo, k));
    let (s, e) = two_sum(2.0, -ph);
    s + (e - pl)
}

/// `erfcx(x) = e^(x²)·erfc(x)` of an `f64`.
pub(crate) fn erfcx_f64(x: f64) -> f64 {
    if x >= 0.5 {
        if x == f64::INFINITY {
            return 0.0;
        }
        return erfcx_pos(x).value();
    }
    if x > -0.5 {
        // 8 pieces of width 1/8.
        let i = (((x + 0.5) * 8.0) as usize).min(7);
        let r = x - (-0.5 + 0.0625 + 0.125 * i as f64);
        return head(&ERFCX_S[i], r).value();
    }
    if x.is_nan() {
        return x;
    }
    if x < -26.7 {
        // 2e^(x²) > f64::MAX from x < −26.6287.
        return f64::INFINITY;
    }
    // 2e^(x²) − erfcx(−x) = (2m − erfcx(−x)·2^(−k))·2^k.
    let a = -x;
    let f = erfcx_pos(a);
    let (sh, sl) = two_prod(a, a);
    let (m, k) = exp_dd(sh, sl);
    let (fh, fl) = (scale(f.hi, -k), scale(f.lo, -k));
    let (s, e) = two_sum(2.0 * m.hi, -fh);
    scale(s + (e + 2.0f64.mul_add(m.lo, -fl)), k)
}

/// The inverse error function: the `y` with `erf(y) = x`, for `−1 ≤ x ≤ 1` (±∞ at ±1).
pub fn erf_inv<T: Real>(x: T) -> T {
    via(x, erf_inv_f64)
}

/// The inverse complementary error function: the `y` with `erfc(y) = x`, for `0 ≤ x ≤ 2`.
pub fn erfc_inv<T: Real>(x: T) -> T {
    via(x, erfc_inv_f64)
}

/// `erf_inv(x)` for `|x| ≤ 1/2`: a polynomial to 2⁻³⁴, then one Newton step on a residual
/// `erf(y₀) − x` that carries no rounding error of its own size.
#[inline(always)]
fn erf_inv_small(x: f64) -> f64 {
    let y0 = x * poly(x * x, &ERFINV_SMALL);
    let t = y0 * y0;
    // erf(y₀) − x = (y₀ − x) + y₀·q, y₀ − x exact (y₀/x is in [0.88, 0.96]).
    let res = y0.mul_add(erf_q(t), y0 - x);
    // e^t to 2⁻⁴⁰ (t < 0.23): enough for the step's derivative.
    let et = poly(
        t,
        &[
            1.0,
            1.0,
            0.5,
            1.0 / 6.0,
            1.0 / 24.0,
            1.0 / 120.0,
            1.0 / 720.0,
            1.0 / 5040.0,
            1.0 / 40320.0,
            1.0 / 362_880.0,
        ],
    );
    (-res).mul_add(SQRT_PI_2 * et, y0)
}

/// `erfc_inv(c)` for `0 < c ≤ 1/2`: a polynomial in `√(−ln c)` to 2⁻³¹, then one Halley step.
#[inline(always)]
fn erfc_inv_tail(c: f64) -> f64 {
    let s = (-c.ln()).sqrt();
    let b = s.to_bits() >> 51;
    let r = s - f64::from_bits((b << 51) | (1 << 50));
    let y0 = poly(r, &ERFCINV_T[(b - (0x3FE0_0000_0000_0000 >> 51)) as usize]);
    // u = (erfc(y₀) − c)/erfc'(y₀), the Newton step.
    let u = if y0 < 0.5 {
        let (s1, e1) = two_sum(1.0, -y0);
        let d = Dd::new(s1)
            .add_f64((-y0).mul_add(erf_q(y0 * y0), e1))
            .add_f64(-c);
        // e^(y₀²), y₀² < 1/4: the exponent of exp_dd is 0.
        -d.value() * SQRT_PI_2 * exp_dd(y0 * y0, 0.0).0.value()
    } else {
        // (erfc(y₀) − c)·e^(y₀²) = erfcx(y₀) − c·e^(y₀²), all without underflow.
        let f = erfcx_pos(y0);
        let (sh, sl) = two_prod(y0, y0);
        let (m, k) = exp_dd(sh, sl);
        let ce = scale(c, k);
        let (p, pe) = two_prod(ce, m.hi);
        let d = f.sub(Dd {
            hi: p,
            lo: fma(ce, m.lo, pe),
        });
        -d.value() * SQRT_PI_2
    };
    // Halley: erfc''/erfc' = −2y.
    y0 - u / y0.mul_add(u, 1.0)
}

fn erf_inv_f64(x: f64) -> f64 {
    let a = x.abs();
    if a <= 0.5 {
        if a < 1e-150 {
            if a == 0.0 {
                return x;
            }
            // erf_inv(x) = x·√π/2·(1 + πx²/12 + …), rounded once (also when subnormal).
            let b = x * pow2(200);
            let (h, l) = two_prod(b, SQRT_PI_2);
            let (hi, lo) = fast_two_sum(h, b.mul_add(SQRT_PI_2_LO, l));
            return scale_dd(Dd { hi, lo }, -200);
        }
        return erf_inv_small(x);
    }
    if a < 1.0 {
        return erfc_inv_tail(1.0 - a).copysign(x);
    }
    if a == 1.0 {
        return f64::INFINITY.copysign(x);
    }
    f64::NAN
}

fn erfc_inv_f64(c: f64) -> f64 {
    if c > 0.0 && c <= 0.5 {
        return erfc_inv_tail(c);
    }
    if c > 0.5 && c < 1.5 {
        return erf_inv_small(1.0 - c);
    }
    if (1.5..2.0).contains(&c) {
        return -erfc_inv_tail(2.0 - c);
    }
    if c == 0.0 {
        return f64::INFINITY;
    }
    if c == 2.0 {
        return f64::NEG_INFINITY;
    }
    f64::NAN
}

/// The Faddeeva function `w(z) = e^(−z²) erfc(−iz)`.
pub fn faddeeva<T: Real>(z: Complex<T>) -> Complex<T> {
    let w = complex::faddeeva_c64(Complex::new(z.re.to_f64(), z.im.to_f64()));
    Complex::new(T::from_f64(w.re), T::from_f64(w.im))
}

/// The Fresnel integrals `(S(x), C(x))`, `S(x) = ∫₀^x sin(πt²/2) dt` and
/// `C(x) = ∫₀^x cos(πt²/2) dt`.
pub fn fresnel<T: Real>(x: T) -> (T, T) {
    let (s, c) = fresnel_f64(x.to_f64());
    (T::from_f64(s), T::from_f64(c))
}

/// Dawson's integral `F(x) = e^(−x²) ∫₀^x e^(t²) dt`.
pub fn dawson<T: Real>(x: T) -> T {
    via(x, dawson_f64)
}

/// Dawson's integral of an `f64`.
pub(crate) fn dawson_f64(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        let t = x * x;
        return (x * t).mul_add(poly(t, &DAWSON_SMALL), x);
    }
    let v = if a < 8.0 {
        let (i, r) = piece(a, B_HALF);
        head(&DAWSON_T[i], r).value()
    } else if a < f64::INFINITY {
        // x·F(x) = P(1/x²).
        let ix = recip_dd(a);
        mul_dd(ix, head(&DAWSON_ASYM, ix.hi * ix.hi)).value()
    } else if a.is_nan() {
        return x;
    } else {
        0.0
    };
    v.copysign(x)
}

/// `(sin(πx²/2), cos(πx²/2))`, with `x²` reduced modulo 4 exactly.
#[inline(always)]
fn sin_cos_half_pi_sq(x: f64) -> (f64, f64) {
    let (h, l) = two_prod(x, x);
    // x² = h + l; each modulo 4, then split into a quadrant and r in [−1/2, 1/2].
    let h4 = h - 4.0 * (h * 0.25).round();
    let l4 = l - 4.0 * (l * 0.25).round();
    let n1 = h4.round();
    let n2 = l4.round();
    let r = (h4 - n1) + (l4 - n2);
    let n3 = r.round();
    let r = r - n3;
    let q = ((n1 + n2 + n3) as i64) & 3;
    let s2 = r * r;
    let sin = r * head(&SINQ, s2).value();
    let cos = s2.mul_add(poly(s2, &COSQ), 1.0);
    match q {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}

/// The Fresnel integrals of an `f64`.
pub(crate) fn fresnel_f64(x: f64) -> (f64, f64) {
    let a = x.abs();
    if a < 1.0 {
        if a < 4.5e-100 {
            // Below 2⁻³³⁰ x³ would underflow: S = (π/6)x³ from x·2³³⁰, scaled back with one
            // rounding; C = x.
            if a < 4.4e-109 {
                return (0.0f64.copysign(x), x);
            }
            let b = a * pow2(330);
            let (b2, b2l) = two_prod(b, b);
            let (b3, b3l) = two_prod(b2, b);
            let p = mul_dd(
                Dd {
                    hi: b3,
                    lo: fma(b2l, b, b3l),
                },
                Dd {
                    hi: FRESNEL_S_SMALL[0],
                    lo: FRESNEL_S_SMALL[1],
                },
            );
            let (hi, lo) = fast_two_sum(p.hi, p.lo);
            return (scale_dd(Dd { hi, lo }, -990).copysign(x), x);
        }
        let (x2, x2l) = two_prod(a, a);
        let w = x2 * x2;
        // C = x + x·w·Q(w); S = x³·P(w) with x³ and P as double-doubles.
        let c = (a * w).mul_add(poly(w, &FRESNEL_C_SMALL), a);
        let (x3, x3l) = two_prod(x2, a);
        let x3 = Dd {
            hi: x3,
            lo: fma(x2l, a, x3l),
        };
        let p = mul_dd(x3, head(&FRESNEL_S_SMALL, w));
        let s = p.hi + p.lo;
        return (s.copysign(x), c.copysign(x));
    }
    if a.is_nan() || a >= 18_014_398_509_481_984.0 {
        // From 2⁵⁴, |C − 1/2| and |S − 1/2| < 1/(πx) < 2⁻⁵⁵.
        return if a.is_nan() {
            (x, x)
        } else {
            (0.5f64.copysign(x), 0.5f64.copysign(x))
        };
    }
    let (f, g) = if a < 8.0 {
        let (i, r) = piece(a, B_ONE);
        (head(&FRESNEL_F_T[i], r), head(&FRESNEL_G_T[i], r))
    } else {
        // x·f = P(1/x⁴), x³·g = Q(1/x⁴).
        let ix = 1.0 / a;
        let v = (ix * ix) * (ix * ix);
        let f = ix * poly(v, &FRESNEL_F_ASYM_1);
        let g = (ix * ix * ix) * poly(v, &FRESNEL_G_ASYM_1);
        (Dd::new(f), Dd::new(g))
    };
    let (sn, cs) = sin_cos_half_pi_sq(a);
    // C = 1/2 + f·sin − g·cos, S = 1/2 − f·cos − g·sin, with the products exact.
    let comb = |p: Dd, u: f64, qq: Dd, v: f64, sign: f64| {
        let (p1, e1) = two_prod(p.hi, u);
        let (p2, e2) = two_prod(qq.hi, v);
        let (d, ed) = two_sum(sign * p1, -p2);
        let (t, et) = two_sum(0.5, d);
        let lo = fma(sign * p.lo, u, sign * e1) - fma(qq.lo, v, e2);
        t + (et + ed + lo)
    };
    let c = comb(f, sn, g, cs, 1.0);
    let s = comb(f, cs, g, sn, -1.0);
    (s.copysign(x), c.copysign(x))
}

/// The asymptotic tables with the head folded: `[c0, c1, …]` as plain doubles.
static FRESNEL_F_ASYM_1: [f64; 5] = fold(&FRESNEL_F_ASYM);
static FRESNEL_G_ASYM_1: [f64; 6] = fold(&FRESNEL_G_ASYM);

/// `[hi, lo, c1, …]` to `[hi + lo, c1, …]`.
const fn fold<const N: usize, const M: usize>(c: &[f64; N]) -> [f64; M] {
    let mut out = [0.0; M];
    out[0] = c[0] + c[1];
    let mut i = 1;
    while i < M {
        out[i] = c[i + 1];
        i += 1;
    }
    out
}
