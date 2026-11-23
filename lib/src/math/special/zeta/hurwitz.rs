//! The Hurwitz zeta function of real arguments, by the Euler–Maclaurin formula
//! `ζ(s, a) = Σ_{n<N} (n + a)^(−s) + A^(1−s)/(s − 1) + A^(−s)/2
//! + Σ_{k≤24} B₂ₖ/(2k)! (s)₂ₖ₋₁ A^(−s−2k+1)`, `A = a + N`, with `N` the fewest terms that bring
//! `A` to `7 + s/2` (where the remainder is below `2⁻⁵⁶`); or, for a small `a`, the direct terms
//! alone once they and everything past them (at most `(m + a)^(1−s)/(s − 1) + (m + a)^(−s)` past
//! the `m`-th) fall below `2⁻⁵⁶` of the first. Every term is positive; they are added from the
//! smallest, each `(n + a)^(−s)` of the exact `n + a` (a double-double, its low part to first
//! order). The remainder is `A^(1−s) (1/(s − 1) + (1/2 + Σ…)/A)` with `A` a double-double: one
//! power, so that it does not leave the range where the result does not, and the rounding of
//! `a + N` not multiplied by `s`.
//!
//! The tables are printed by `gen_util.py` in this directory (with `gen_fit.py`).

use super::real::zeta_f64;
use super::util::{exp_mul, exp_parts, ln_dd, residual, scale, two_prod, two_sum};

/// `B₂ₖ/(2k)!` for `k = 1, …, 24`.
pub(super) const BERNOULLI_2K: [f64; 24] = [
    0.08333333333333333,
    -0.001388888888888889,
    3.306878306878307e-05,
    -8.267195767195768e-07,
    2.08767569878681e-08,
    -5.284190138687493e-10,
    1.3382536530684679e-11,
    -3.3896802963225827e-13,
    8.586062056277845e-15,
    -2.174868698558062e-16,
    5.5090028283602295e-18,
    -1.3954464685812522e-19,
    3.534707039629467e-21,
    -8.953517427037546e-23,
    2.267952452337683e-24,
    -5.744790668872202e-26,
    1.455172475614865e-27,
    -3.6859949406653103e-29,
    9.336734257095045e-31,
    -2.36502241570063e-32,
    5.990671762482134e-34,
    -1.5174548844682903e-35,
    3.843758125454189e-37,
    -9.736353072646691e-39,
];

/// `(xh + xl)^y` as a double-double for a double-double `x > 0`: `y ln x` as a double-double
/// (`ln(xh + xl) = ln xh + xl/xh`) and `e^(t + tl) = 2^q (e + el)(1 + tl)`; past the normal range
/// (`|y ln x| > 700`) rounded once, with a zero low part.
#[inline]
fn pow_dd(xh: f64, xl: f64, y: f64) -> (f64, f64) {
    let (l, ll) = ln_dd(xh);
    let (t, te) = two_prod(y, l);
    let tl = te + y * (ll + xl / xh);
    if t.abs() > 700.0 || t.is_nan() {
        return (exp_mul(t, 1.0, tl), 0.0);
    }
    let (e, el, q) = exp_parts(t);
    let (h, hl) = two_sum(e, el + e * tl);
    (scale(h, q), scale(hl, q))
}

/// `(n + a)^(−s)` of the exact sum, as a double-double.
#[inline(always)]
fn term(n: f64, a: f64, s: f64) -> (f64, f64) {
    let (h, l) = two_sum(n, a);
    let p = h.powf(-s);
    (if l == 0.0 { p } else { p * (1.0 - s * (l / h)) }, 0.0)
}

/// `(xh + xl)^y · (g + gl)` for a double-double `x > 0`, rounded once: `y ln x` as a
/// double-double, `ln(xh + xl) = ln xh + xl/xh`.
#[inline]
fn pow_mul_dd(xh: f64, xl: f64, y: f64, g: f64, gl: f64) -> f64 {
    let (l, ll) = ln_dd(xh);
    let (t, te) = two_prod(y, l);
    let tl = te + y * (ll + xl / xh);
    exp_mul(t, g, gl + g * tl)
}

/// The Euler–Maclaurin remainder at the double-double `A = ah + al`:
/// `A^(1−s)/(s − 1) + A^(−s)/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ A^(−s−2k+1)`
/// `= A^(1−s) (1/(s − 1) + (1/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ A^(1−2k))/A)`, rounded once.
#[inline]
fn tail(ah: f64, al: f64, s: f64) -> f64 {
    let r = 1.0 / ah;
    // (s)₂ₖ₋₁ A^(1−2k), k = 1, 2, … (each factor (s + j)/A apart, which stays near 2 at most)
    let mut fac = s * r;
    let mut sum = 0.0;
    let mut k2 = 1.0;
    for &b in BERNOULLI_2K.iter() {
        let t = b * fac;
        sum += t;
        if t.abs() <= 1e-18 * (ah / (s - 1.0)) {
            break;
        }
        fac *= ((s + k2) * r) * ((s + k2 + 1.0) * r);
        k2 += 2.0;
    }
    // 1/(s − 1) and (1/2 + sum)/A as double-doubles (s − 1 is exact)
    let sm1 = s - 1.0;
    let q = 1.0 / sm1;
    let ql = residual(q, sm1, 1.0) / sm1;
    let (u, ul) = two_sum(0.5, sum);
    let v = u / ah;
    let vl = (residual(v, ah, u) + ul - v * al) / ah;
    let (h, hl) = two_sum(q, v);
    pow_mul_dd(ah, al, 1.0 - s, h, hl + (ql + vl))
}

/// Whether the series past its `m`-th term is below `2⁻⁵⁶` of the first: the rest is at most
/// `(m + a)^(−s) (1 + (m + a)/(s − 1))`.
#[inline]
fn rest_negligible(s: f64, a: f64, m: f64) -> bool {
    let ma = m + a;
    -s * (ma / a).ln() + (1.0 + ma / (s - 1.0)).ln() < -56.0 * core::f64::consts::LN_2
}

/// `ζ(s, a)` of `f64`s.
pub(crate) fn hurwitz_f64(s: f64, a: f64) -> f64 {
    if s.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if s < 1.0 || a <= 0.0 {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    if a == 1.0 {
        return zeta_f64(s);
    }
    if a == f64::INFINITY {
        return 0.0;
    }
    if s == f64::INFINITY {
        return if a < 1.0 { f64::INFINITY } else { 0.0 };
    }
    // The direct terms: to A = a + N ≥ 7 + s/2, or while (n + a)^(−s) ≥ 2⁻⁵⁶ a^(−s) when the
    // series past them is negligible (near s = 1 it is about (m + a)^(1−s)/(s − 1), which a tiny
    // a^(−s) need not dwarf).
    let need = 7.0 + 0.5 * s;
    let mut n = if a >= need { 0.0 } else { (need - a).ceil() };
    let mut with_tail = true;
    if n > 0.0 {
        // (1 + n/a)^(−s) < 2⁻⁵⁶ once n ≥ a (2^(56/s) − 1).
        let m = (a * ((56.0 / s).exp2() - 1.0)).ceil() + 1.0;
        if m < n && rest_negligible(s, a, m) {
            n = m;
            with_tail = false;
        }
    }
    // From the smallest term up, in double-double; the first term a^(−s) rounded once with the
    // rest.
    let (mut hi, mut lo) = if with_tail {
        // A = a + n exactly, as a double-double
        let (ah, al) = two_sum(a, n);
        (tail(ah, al, s), 0.0)
    } else {
        (0.0, 0.0)
    };
    while n > 1.0 {
        n -= 1.0;
        let (th, tl) = term(n, a, s);
        let (h, e) = two_sum(hi, th);
        hi = h;
        lo += e + tl;
    }
    if n == 1.0 {
        // a^(−s) + (hi + lo), rounded once
        let (th, tl) = pow_dd(a, 0.0, -s);
        if !th.is_finite() {
            return th;
        }
        let (h, e) = two_sum(th, hi);
        h + (e + (tl + lo))
    } else {
        hi + lo
    }
}
