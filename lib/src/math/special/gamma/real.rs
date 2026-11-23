//! `Γ(x)` and `ln |Γ(x)|` of real arguments.
//!
//! * `Γ`: `1/Γ(1 + t)` on `|t| ≤ 1/2` by a minimax polynomial, carried in double-double, with the
//!   recurrence over exact factors `t + k` up to `x < 8` (and down to `x > −8`); Stirling's series
//!   in double-double with an accurate exponential beyond, and the reflection formula below `−8`;
//!   `1/x − γ` rounded once below `2⁻⁵⁴`.
//! * `ln Γ`: `ln Γ(2 + t) = t(1 + t)·F(t)` on `−1/2 ≤ t < 1` by three minimax pieces (the zeros at
//!   1 and 2 are the factors), the recurrences `ln Γ(x) = ln Γ(x + 1) − ln x` below and
//!   `ln Γ(x) = ln Γ(x − k) + ln Π(x − j)` above, Stirling's series from 8 on, and the reflection
//!   formula for `x < −1/2` (beside its zeros with the more accurate `ln Γ(−x)`).

use super::super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::ddm::*;
use super::tab::*;

/// Where Stirling's series takes over.
pub(super) const STIRLING_X: f64 = 8.0;

/// `1/Γ(1 + t)` for `|t| ≤ 1/2`, as a double-double: `1 + γt + c₂t² + t²R(t)`.
#[inline]
pub(super) fn rgamma1p(t: f64) -> Dd {
    let (th, tl) = two_prod(t, t);
    let r = th * poly(t, &RGAMMA);
    let (gh, gl) = two_prod(EULER_HI, t);
    let (ch, cl) = two_prod(RGAMMA_C2, th);
    let lo = fma(EULER_LO, t, gl) + fma(RGAMMA_C2, tl, cl) + r;
    let (h, e) = fast_two_sum(1.0, gh);
    let (h, e2) = two_sum(h, ch);
    let (hi, lo) = fast_two_sum(h, (e + e2) + lo);
    Dd { hi, lo }
}

/// The product `Π (t + k)` for `k` in `lo..hi`, compensated: `(p, e)` with `p + e` the product to
/// about `2⁻¹⁰⁰` (every factor is exact).
#[inline(always)]
fn product(t: f64, lo: i32, hi: i32) -> Dd {
    let mut p = 1.0;
    let mut e = 0.0;
    let mut k = lo;
    while k < hi {
        let a = t + k as f64;
        let q = p * a;
        e = fma(e, a, prod_err(p, a, q));
        p = q;
        k += 1;
    }
    Dd { hi: p, lo: e }
}

/// `(a.hi + a.lo)/(b.hi + b.lo)`, rounded once (nearly).
#[inline(always)]
fn div_f64(a: Dd, b: Dd) -> f64 {
    let q = a.hi / b.hi;
    let r = rem(q, b.hi, a.hi);
    let r = fma(-q, b.lo, r + a.lo);
    q + r / b.hi
}

/// `ln Γ(x) − ((x − 1/2) ln x − x + ln √(2π))` for `x ≥ 8`.
#[inline(always)]
fn stirling_tail(x: f64) -> f64 {
    let r = 1.0 / x;
    let u = r * r;
    let [c0, rest @ ..] = STIRLING;
    r * (fma(u, poly(u, &rest), c0) + STIRLING_C0_LO)
}

/// `ln Γ(x)` by Stirling's series in double-double, for `8 ≤ x < 2⁵²` (absolute error below
/// `2⁻⁶⁰` beside a value above 8).
#[inline]
pub(super) fn lgamma_stirling(x: f64) -> Dd {
    let l = ln_dd(x);
    // (x − 1/2) ln x − x + ln √(2π) + S(x); x − 1/2 is exact, and (x − 1/2) ln x > x.
    let h = x - 0.5;
    let (ph, pe) = two_prod(l.hi, h);
    let pl = fma(l.lo, h, pe);
    let (s, e1) = fast_two_sum(ph, -x);
    let (s, e2) = fast_two_sum(s, LN_SQRT_2PI_HI);
    let lo = (e1 + e2) + (pl + LN_SQRT_2PI_LO) + stirling_tail(x);
    Dd { hi: s, lo }
}

/// `Γ(x)` of an `f64`.
pub(crate) fn gamma_f64(x: f64) -> f64 {
    gamma_shift(x, 0)
}

/// `Γ(x + k)` for `k` 0 or 1, without rounding `x + k` (`x! = Γ(x + 1)`).
pub(super) fn gamma_shift(x: f64, k: i32) -> f64 {
    let kf = k as f64;
    let y = x + kf;
    if y.is_nan() || y.abs() >= STIRLING_X {
        if x.is_nan() {
            return x;
        }
        if y > 0.0 {
            if y > 171.625 {
                return f64::INFINITY;
            }
            if k == 0 {
                return exp_f64(lgamma_stirling(x));
            }
            // ln Γ(hi + lo) = ln Γ(hi) + lo·ψ(hi), ψ(hi) ≈ ln hi − 1/(2 hi) − 1/(12 hi²).
            let (yh, yl) = two_sum(x, kf);
            let e = lgamma_stirling(yh);
            let r = 1.0 / yh;
            let psi = ln_dd(yh).hi - r * (0.5 + r / 12.0);
            return exp_f64(e.add_f64(yl * psi));
        }
        return gamma_reflected(x, k);
    }
    if k == 0 && x.abs() < 5.551_115_123_125_783e-17 {
        // Γ(x) = 1/x − γ + O(x) (the next term is below 2⁻¹⁰⁸ of it), rounded once: 1/x with its
        // remainder (1 − x·q)/x. ±∞ at ±0, and where 1/x overflows.
        let q = 1.0 / x;
        if !q.is_finite() {
            return q;
        }
        return q + (rem(q, x, 1.0) * q - EULER_HI);
    }
    let r = round_int(x);
    let t = x - r;
    let n = r + kf;
    let d = rgamma1p(t);
    if n >= 1.0 {
        // Γ(n + t) = Γ(1 + t)·(t + 1)(t + 2)…(t + n − 1).
        div_f64(product(t, 1, n as i32), d)
    } else {
        if t == 0.0 {
            return f64::NAN;
        }
        // Γ(n + t) = Γ(1 + t)/(t(t − 1)…(t + n)).
        let p = product(t, n as i32, 1);
        let den = d.mul(p);
        div_f64(Dd::new(1.0), den)
    }
}

/// `Γ(x + k)` for `x + k ≤ −8`, `k` 0 or 1: `Γ(x) = −π/(x sin(πx) Γ(−x))` and
/// `Γ(x + 1) = x Γ(x)`.
fn gamma_reflected(x: f64, k: i32) -> f64 {
    if x == f64::NEG_INFINITY || x == x.floor() {
        return f64::NAN;
    }
    if x < -190.0 {
        // |Γ(x)| < 10⁻³⁴⁰ even beside a pole.
        let s = sin_pi_dd(x).hi * if k == 0 { 1.0 } else { -1.0 };
        return if s > 0.0 { 0.0 } else { -0.0 };
    }
    let e = Dd {
        hi: LN_PI_HI,
        lo: LN_PI_LO,
    }
    .sub(lgamma_stirling(-x));
    let (d, m) = exp_parts(e);
    let s = sin_pi_dd(x);
    let den = if k == 0 { s.mul_f64(-x) } else { s.neg() };
    let q = d.div(den);
    scale(q, m)
}

/// `ln Γ(2 + t)` for `−1/2 ≤ t < 1`, as a double-double.
#[inline]
pub(super) fn lgamma2p(t: f64) -> Dd {
    // F(t) = Y + c₁s + (c₀ + s²R(s)) on the piece of t, the first two terms without error (in
    // f64 the linear term would round to a sixth of an ulp of the result at the pieces' ends).
    let f = if t < 0.0 {
        let [y, c0, c1, r @ ..] = LGAM_A;
        piece_sum(y, c0, c1, t + 0.25, poly(t + 0.25, &r))
    } else if t < 0.5 {
        let [y, c0, c1, r @ ..] = LGAM_B;
        piece_sum(y, c0, c1, t - 0.25, poly(t - 0.25, &r))
    } else {
        let [y, c0, c1, r @ ..] = LGAM_C;
        piece_sum(y, c0, c1, t - 0.75, poly(t - 0.75, &r))
    };
    // t(1 + t) = t + t² (|t| ≥ t² on the pieces but t = −1/2… where both are 1/4: exact).
    f.mul(x_plus_x2(t))
}

/// `Y + c₁s + (c₀ + s²r)` with `r = R(s)`, the first two terms without error.
#[inline(always)]
fn piece_sum(y: f64, c0: f64, c1: f64, s: f64, r: f64) -> Dd {
    let (p, e) = two_prod(c1, s);
    let (h, e2) = fast_two_sum(y, p);
    let (hi, lo) = fast_two_sum(h, e2 + (e + fma(s * s, r, c0)));
    Dd { hi, lo }
}

/// `x + x²` as a double-double (`x(1 + x)` without rounding `1 + x`).
#[inline(always)]
fn x_plus_x2(x: f64) -> Dd {
    let (sh, sl) = two_prod(x, x);
    let (h, e) = fast_two_sum(x, sh);
    let (hi, lo) = fast_two_sum(h, e + sl);
    Dd { hi, lo }
}

/// `ln Γ(1 + t) = t·H(t)` for `−1/2 ≤ t < 1/2` (`t` exact), as a double-double: five minimax
/// pieces of `H`, the zero at `t = 0` a factor.
#[inline(always)]
fn lgamma1p(t: f64) -> Dd {
    // The piece without a branch: t ≥ −3/8, −1/4, 0, 1/4.
    let i =
        (t >= -0.375) as usize + (t >= -0.25) as usize + (t >= 0.0) as usize + (t >= 0.25) as usize;
    const C: [f64; 5] = [-0.4375, -0.3125, -0.125, 0.125, 0.375];
    // H = (Y + c₁s) + (c₀ + s²R(s)), the first part without error, then times t.
    let s = t - C[i];
    let [y, c0, c1, r @ ..] = LGAM_H[i];
    let (p, e) = two_prod(c1, s);
    let (u, ue) = fast_two_sum(y, p);
    let ul = ue + (e + fma(s * s, poly(s, &r), c0));
    let (q, qe) = two_prod(t, u);
    let (hi, lo) = fast_two_sum(q, fma(t, ul, qe));
    Dd { hi, lo }
}

/// `ln Γ(x)` for `0 < x < 2⁵²`, as a double-double (to about `2⁻⁶⁰`: the other functions build
/// on it).
pub(super) fn lgamma_pos_dd(x: f64) -> Dd {
    if x >= STIRLING_X {
        return lgamma_stirling(x);
    }
    if x >= 3.0 {
        // ln Γ(x) = ln Γ(2 + t) + ln((x − 1)(x − 2)…(2 + t)).
        let f = x.floor();
        let t = x - f;
        let p = product(t, 2, f as i32);
        return lgamma2p(t).add(ln_of_dd(p));
    }
    if x >= 1.5 {
        return lgamma2p(x - 2.0);
    }
    if x >= 0.5 {
        // ln Γ(x) = ln Γ(x + 1) − ln x.
        return lgamma2p(x - 1.0).sub(ln_dd(x));
    }
    // ln Γ(x) = ln Γ(x + 2) − ln(x(1 + x)).
    lgamma2p(x).sub(ln_of_dd(x_plus_x2(x)))
}

/// `ln Γ(x)` for `0 < x < 8` from the result's own pieces (`ln Γ(1 + t)/t` on `[−1/2, 1/2)`,
/// `ln Γ` on the units of `[3, 8)`), without the logarithms of `lgamma_pos_dd`: a
/// double-double to about `2⁻⁵⁸` relative.
#[inline]
fn lgamma_small_dd(x: f64) -> Dd {
    if x >= 3.0 {
        // Y + c₁s + (c₀ + c₂s² + s³R(s)), the first two terms without error and c₂s² with the
        // rounding of s² (rounded, c₂s² reaches a tenth of an ulp of the result on [3, 3.5)).
        let f = x.floor();
        let s = x - (f + 0.5);
        let [y, c0, c1, c2, r @ ..] = LGAM_MID[f as usize - 3];
        let (p, e) = two_prod(c1, s);
        let (h, e2) = fast_two_sum(y, p);
        let (sh, sl) = two_prod(s, s);
        let q = c2 * sh;
        let ql = fma(c2, sl, prod_err(c2, sh, q));
        let (u, e3) = fast_two_sum(h, q);
        let lo = (e2 + e3) + (e + (ql + fma(sh * s, poly(s, &r), c0)));
        let (hi, lo) = fast_two_sum(u, lo);
        return Dd { hi, lo };
    }
    if x >= 1.5 {
        return lgamma2p(x - 2.0);
    }
    if x >= 0.5 {
        return lgamma1p(x - 1.0);
    }
    // ln Γ(x) = ln Γ(1 + x) − ln x.
    lgamma1p(x).sub(ln_dd(x))
}

/// `ln |Γ(x)|` of an `f64`.
pub(crate) fn lgamma_f64(x: f64) -> f64 {
    if x > 0.0 {
        if x < STIRLING_X {
            return lgamma_small_dd(x).value();
        }
        if x < 4_503_599_627_370_496.0 {
            return lgamma_stirling(x).value();
        }
        if x == f64::INFINITY {
            return x;
        }
        // ln Γ(x) = x(ln x − 1) − (ln x)/2 + O(1), rounded once.
        let l = ln_dd(x);
        let d = l.add_f64(-1.0);
        if FMA {
            return x.mul_add(d.hi, fma(x, d.lo, -0.5 * l.hi));
        }
        // Dekker's product needs its factors below 2⁹⁹⁶: x scaled by 2⁻⁶⁴, rounded once.
        let xs = x * 5.421_010_862_427_522e-20;
        let (p, e) = two_prod(xs, d.hi);
        let c = fma(xs, d.lo, -0.5 * l.hi * 5.421_010_862_427_522e-20);
        return (p + (e + c)) * 18_446_744_073_709_551_616.0;
    }
    if x.is_nan() {
        return x;
    }
    if x == x.floor() {
        // The poles, ±0 and −∞.
        return f64::INFINITY;
    }
    lgamma_neg_dd(x, true).value()
}

/// `ln |Γ(x)|` for a negative non-integer `x`, as a double-double (`fast`: `ln Γ(−x)` from the
/// result's pieces, to about `2⁻⁵⁸` relative, enough for `lgamma` itself but where the result is
/// a small difference of its terms).
pub(super) fn lgamma_neg_dd(x: f64, fast: bool) -> Dd {
    if x > -0.5 {
        // ln |Γ(x)| = ln Γ(1 + x) − ln(−x).
        return lgamma1p(x).sub(ln_dd(-x));
    }
    // ln |Γ(x)| = ln π − ln |x sin(πx)| − ln Γ(−x).
    let s = sin_pi_dd(x).mul_f64(x);
    let s = if s.hi < 0.0 { s.neg() } else { s };
    let head = Dd {
        hi: LN_PI_HI,
        lo: LN_PI_LO,
    }
    .sub(ln_of_dd(s));
    if fast && x > -STIRLING_X {
        let g = lgamma_small_dd(-x);
        let r = head.sub(g);
        // The pieces' absolute error, about 2.5·10⁻¹⁸ ln Γ(−x) + 1.2·10⁻¹⁸, is below a tenth
        // of an ulp of a result above 1/8 + |ln Γ(−x)|/4. Beside the zeros, where the result is
        // a small difference of its terms, ln Γ(−x) is taken instead from the logarithm of the
        // exact product and ln Γ(2 + t) (at most 0.7), better absolutely.
        if r.hi.abs() >= 0.125 + 0.25 * g.hi.abs() {
            return r;
        }
    }
    head.sub(lgamma_pos_dd(-x))
}

/// `sin(πx)`, exact at the integers and correctly rounded nearly always (for the other areas).
#[allow(dead_code)]
pub(crate) fn sin_pi(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    sin_pi_dd(x).hi
}
