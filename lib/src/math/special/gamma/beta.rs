//! The beta function and its logarithm, and factorials, binomial coefficients and Pochhammer's
//! symbol of floats.
//!
//! * `ln B(a, b)` for `a, b > 0`: `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` in double-double below 8
//!   (`a + b` as a double-double); `ln Γ(a) + [ln Γ(b) − ln Γ(a + b)]` by the Stirling difference
//!   when only `b ≥ 8`; and for both at least 8 the symmetric form
//!   `ln √(2π) + (a − ½) ln(a/c) + (b − ½) ln(b/c) − ½ ln c + S(a) + S(b) − S(c)` with
//!   `ln(b/c) = −ln(1 + a/b)` taken directly, so no term cancels another.
//! * Other real arguments by the reflection formula `Γ(x) = π/(sin(πx) Γ(1 − x))` turned into
//!   identities between beta functions (or Pochhammer symbols) of positive arguments, with the
//!   sines of the exact arguments (`a + b` as a double-double, so that a sum beside a pole of `Γ`
//!   keeps its distance to it) and the ratios of gamma functions of large arguments as Stirling
//!   differences, never as differences of their logarithms.
//! * `x! = Γ(x + 1)` without rounding `x + 1`; `C(n, k)` and `(x)ₙ` by exact double-double
//!   products for small integer `k` or `n`, else by `ln B` and the Stirling difference.

use super::super::dd::{Dd, fast_two_sum, two_sum};
use super::ddm::*;
use super::ibeta::{
    lgamma_delta, lgamma_delta_dd, lgamma_of_dd, ln1p_of_dd, rgamma_dd, stirling_delta,
    stirling_rem,
};
use super::psi::digamma_f64;
use super::real::{gamma_shift, lgamma_f64, lgamma_pos_dd};
use super::tab::*;

const LN_SQRT_2PI: Dd = Dd {
    hi: LN_SQRT_2PI_HI,
    lo: LN_SQRT_2PI_LO,
};

const LN_PI: Dd = Dd {
    hi: LN_PI_HI,
    lo: LN_PI_LO,
};

/// `2⁵²`: every float beyond it is an integer.
const INT: f64 = 4_503_599_627_370_496.0;

/// `x` is an integer.
#[inline(always)]
fn is_int(x: f64) -> bool {
    x == x.floor()
}

/// `B(a, b)` for `10⁻¹⁰⁰ < a, b < 8` as a double-double: `R(a + b)/(R(a)R(b))`, `R = 1/Γ` in
/// double-double, the low part of `a + b` through `ψ`.
#[inline]
fn beta_small_dd(a: f64, b: f64) -> Dd {
    let (ch, cl) = two_sum(a, b);
    let mut rc = rgamma_dd(ch);
    if cl != 0.0 {
        rc = rc.add_f64(-rc.hi * cl * digamma_f64(ch));
    }
    rc.div(rgamma_dd(a).mul(rgamma_dd(b)))
}

/// `ln B(a, b)` for finite `a, b > 0`, as a double-double.
pub(super) fn lbeta_pos_dd(a: f64, b: f64) -> Dd {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    if b < STIRLING_HALF {
        let (ch, cl) = two_sum(a, b);
        return lgamma_pos_dd(a)
            .add(lgamma_pos_dd(b))
            .sub(lgamma_of_dd(Dd { hi: ch, lo: cl }));
    }
    if a < STIRLING_HALF {
        // ln Γ(a) + (ln Γ(b) − ln Γ(b + a)).
        return lgamma_pos_dd(a).add(lgamma_delta(b, a));
    }
    // q = a/b ≤ 1, L₁ = ln(1 + q) = ln(c/b), ln(a/c) = ln q − L₁.
    let q = Dd::new(a).div(Dd::new(b));
    let l1 = ln1p_of_dd(q);
    let la_c = ln_of_dd(q).sub(l1);
    let lc = ln_dd(b).add(l1);
    if (a + b).is_finite() {
        // (a − ½) ln(a/c) − (b − ½) L₁ − ½ ln c + ln √(2π) + S(a) + S(b) − S(c).
        return la_c
            .mul_f64(a)
            .sub(la_c.mul_f64(0.5))
            .sub(l1.mul_f64(b))
            .add(l1.mul_f64(0.5))
            .sub(lc.mul_f64(0.5))
            .add(LN_SQRT_2PI)
            .add(stirling_delta(a, b).neg());
    }
    // a + b overflows (S(c) is then below every ulp), and so may the sum, whose first terms reach
    // (a + b) ln 2 together: at half scale, −∞ where twice it is beyond the range.
    let s = stirling_rem(a).add(stirling_rem(b));
    let h = la_c
        .mul_f64(0.5 * a)
        .sub(la_c.mul_f64(0.25))
        .sub(l1.mul_f64(0.5 * b))
        .add(l1.mul_f64(0.25))
        .sub(lc.mul_f64(0.25))
        .add(LN_SQRT_2PI.add(s).mul_f64(0.5));
    if h.hi < -0.5 * f64::MAX {
        return Dd::new(f64::NEG_INFINITY);
    }
    h.mul_f64(2.0)
}

/// Where `lbeta_pos_dd` changes form (`ibeta`'s Stirling functions need 8).
const STIRLING_HALF: f64 = 8.0;

/// The sign of `Γ(x)` of a non-pole `x`.
#[inline]
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 || (x.floor() as i64) % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// `x + y` as a double-double.
#[inline(always)]
fn sum(x: f64, y: f64) -> Dd {
    let (hi, lo) = two_sum(x, y);
    Dd { hi, lo }
}

/// A double-double is a pole of `Γ` (a non-positive integer, exactly).
#[inline]
fn is_pole(d: Dd) -> bool {
    d.lo == 0.0 && d.hi <= 0.0 && is_int(d.hi)
}

/// `1 − x` of a double-double.
#[inline]
fn one_minus(x: Dd) -> Dd {
    let (hi, lo) = two_sum(1.0, -x.hi);
    Dd { hi, lo }.add_f64(-x.lo)
}

/// `ln |sin(πx)|` and the sign of `sin(πx)`, for a double-double `x` that is not an integer.
#[inline]
fn ln_sin_pi(x: Dd) -> (Dd, f64) {
    let s = sin_pi_of_dd(x);
    if s.hi < 0.0 {
        (ln_of_dd(s.neg()), -1.0)
    } else {
        (ln_of_dd(s), 1.0)
    }
}

/// The sign of `Γ(x)` of a non-pole double-double (that of `sin(πx)` for `x < 0`).
#[inline]
fn gamma_sign_of_dd(x: Dd) -> f64 {
    if x.hi > 0.0 || x.lo == 0.0 {
        gamma_sign(x.hi)
    } else {
        ln_sin_pi(x).1
    }
}

/// `ln Γ(x)` of a positive double-double, the low part through `ψ` (to first order: `x.lo` is
/// below half an ulp of `x.hi`, so the second order is below `2⁻¹⁰⁷(1 + x)`).
fn lgamma_pos_of_dd(x: Dd) -> Dd {
    if x.hi >= INT {
        // (ψ = ln x to far below an ulp here.)
        return Dd::new(lgamma_f64(x.hi)).add_f64(x.lo * ln_dd(x.hi).hi);
    }
    let l = lgamma_pos_dd(x.hi);
    if x.lo == 0.0 {
        l
    } else {
        l.add_f64(x.lo * digamma_f64(x.hi))
    }
}

/// `ln Γ(x) − ln Γ(x + d)` for positive double-doubles `x` and `x + d`: a Stirling difference
/// where either is at least 8 (the difference of two logarithms of gamma functions would lose
/// their size in absolute error), else the difference of the two.
fn lgamma_ratio(x: Dd, d: Dd) -> Dd {
    let y = x.add(d);
    if x.hi >= STIRLING_HALF {
        lgamma_delta_dd(x, d)
    } else if y.hi >= STIRLING_HALF {
        lgamma_delta_dd(y, d.neg()).neg()
    } else {
        lgamma_pos_of_dd(x).sub(lgamma_pos_of_dd(y))
    }
}

/// `ln B(p, q)` of positive double-doubles (`lbeta_pos_dd` where both are `f64`s).
fn lbeta_pos_of_dd(p: Dd, q: Dd) -> Dd {
    if p.lo == 0.0 && q.lo == 0.0 {
        return lbeta_pos_dd(p.hi, q.hi);
    }
    if p.hi.max(q.hi) < 1_099_511_627_776.0 {
        // The low parts by the derivatives of ln B, ψ(p) − ψ(p + q) and the like: in f64 their
        // error, about 2⁻⁵³ ln(p + q), times a low part of at most 2⁻⁵³ p is below 2⁻⁶⁰ here.
        let d = digamma_f64(p.hi + q.hi);
        return lbeta_pos_dd(p.hi, q.hi)
            .add_f64(p.lo * (digamma_f64(p.hi) - d) + q.lo * (digamma_f64(q.hi) - d));
    }
    // Beyond 2⁴⁰ as a Stirling difference of the double-doubles: ln Γ(p) + (ln Γ(q) − ln Γ(q + p)).
    let (p, q) = if p.hi <= q.hi { (p, q) } else { (q, p) };
    lgamma_pos_of_dd(p).add(lgamma_ratio(q, p))
}

/// `ln |B(a, b)|` and the sign of `B(a, b)` for double-doubles `a`, `b` and `c = a + b` none of
/// which is a pole, `a` or `b` negative: the reflection formula makes of `B` a beta function of
/// positive arguments and sines (with `a ≤ b`),
///
/// * `a, b < 0`: `B = π sin(πc)/(sin(πa) sin(πb)) / ((1 − c) B(1 − a, 1 − b))`,
/// * `a < 0 < b`, `c < 0`: `B = (sin(πc)/sin(πa)) B(b, 1 − c)`,
/// * `a < 0 < b`, `c > 0`: `B = π/(sin(πa) · b · B(c, 1 − a))`.
fn lbeta_reflected(a: Dd, b: Dd, c: Dd) -> (Dd, f64) {
    let (a, b) = if a.hi <= b.hi { (a, b) } else { (b, a) };
    let (la, sa) = ln_sin_pi(a);
    if b.hi < 0.0 {
        let (lb, sb) = ln_sin_pi(b);
        let (lc, sc) = ln_sin_pi(c);
        let l = LN_PI
            .add(lc)
            .sub(la)
            .sub(lb)
            .sub(ln_of_dd(one_minus(c)))
            .sub(lbeta_pos_of_dd(one_minus(a), one_minus(b)));
        return (l, sa * sb * sc);
    }
    if c.hi < 0.0 {
        let (lc, sc) = ln_sin_pi(c);
        return (lc.sub(la).add(lbeta_pos_of_dd(b, one_minus(c))), sa * sc);
    }
    let l = LN_PI
        .sub(la)
        .sub(ln_of_dd(b))
        .sub(lbeta_pos_of_dd(c, one_minus(a)));
    (l, sa)
}

/// `ln |B(a, b)|` and its sign for non-pole double-doubles, `c = a + b` not a pole either.
#[inline]
fn lbeta_signed(a: Dd, b: Dd, c: Dd) -> (Dd, f64) {
    if a.hi > 0.0 && b.hi > 0.0 {
        (lbeta_pos_of_dd(a, b), 1.0)
    } else {
        lbeta_reflected(a, b, c)
    }
}

/// `B(a, b)` of `f64`s.
pub(super) fn beta_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return a + b;
    }
    if a > 0.0 && b > 0.0 {
        if a == f64::INFINITY || b == f64::INFINITY {
            return 0.0;
        }
        if a.max(b) < STIRLING_HALF && a.min(b) > 1e-100 {
            return beta_small_dd(a, b).value();
        }
        return exp_f64(lbeta_pos_dd(a, b));
    }
    if a.is_infinite() || b.is_infinite() {
        return f64::NAN;
    }
    if a == 0.0 || b == 0.0 {
        // Γ(±0) = ±∞ over a finite ratio: ±∞; B(0, 0) has no value.
        let (z, o) = if a == 0.0 { (a, b) } else { (b, a) };
        if o == 0.0 || (o < 0.0 && is_int(o)) {
            return f64::NAN;
        }
        return 1.0 / z;
    }
    if (a < 0.0 && is_int(a)) || (b < 0.0 && is_int(b)) {
        // Γ(a) or Γ(b) infinite: no limit, with or without a pole of Γ(a + b).
        return f64::NAN;
    }
    let c = sum(a, b);
    if is_pole(c) {
        return 0.0 * gamma_sign(a) * gamma_sign(b);
    }
    let (l, s) = lbeta_reflected(Dd::new(a), Dd::new(b), c);
    s * exp_f64(l)
}

/// `ln |B(a, b)|` of `f64`s.
pub(super) fn lbeta_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return a + b;
    }
    if a > 0.0 && b > 0.0 {
        if a == f64::INFINITY || b == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        if a.max(b) < STIRLING_HALF && a.min(b) > 1e-100 {
            return ln_of_dd(beta_small_dd(a, b)).value();
        }
        return lbeta_pos_dd(a, b).value();
    }
    if a.is_infinite() || b.is_infinite() {
        return f64::NAN;
    }
    let pole = |x: f64| x <= 0.0 && is_int(x);
    let c = sum(a, b);
    match (pole(a) || pole(b), is_pole(c)) {
        (true, true) => f64::NAN,
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => lbeta_reflected(Dd::new(a), Dd::new(b), c).0.value(),
    }
}

/// `x! = Γ(x + 1)` of an `f64` (NaN at the negative integers).
pub(super) fn factorial_f64(x: f64) -> f64 {
    if (0.0..=170.0).contains(&x) && is_int(x) {
        return FACT_HI[x as usize];
    }
    gamma_shift(x, 1)
}

/// `n!` of an integer `0 ≤ n ≤ 170` as a double-double.
#[inline(always)]
fn fact_dd(n: f64) -> Dd {
    let i = n as usize;
    Dd {
        hi: FACT_HI[i],
        lo: FACT_LO[i],
    }
}

/// `Π (x + start + step·i)` for `i` in `0..m`, every factor an exact double-double `h + l`,
/// compensated: `p` with the sum `e` of the rounding errors (`p·l` per factor, and the error of
/// each product).
#[inline]
fn product(x: f64, start: f64, step: f64, m: u32) -> Dd {
    let mut p = 1.0;
    let mut e = 0.0;
    for i in 0..m {
        let (h, l) = two_sum(x, start + step * i as f64);
        let q = p * h;
        e = fma(e, h, fma(p, l, prod_err(p, h, q)));
        p = q;
    }
    let (hi, lo) = fast_two_sum(p, e);
    Dd { hi, lo }
}

/// `m!` as a double-double for `m ≤ 32`.
#[inline]
fn factorial_dd(m: u32) -> Dd {
    let mut p = Dd::new(1.0);
    for i in 2..=m {
        p = p.mul_f64(i as f64);
    }
    p
}

/// `C(n, k)` of `f64`s.
pub(super) fn binomial_f64(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() {
        return n + k;
    }
    if !(n.is_finite() && k.is_finite()) {
        return f64::NAN;
    }
    if is_int(k) {
        if k < 0.0 {
            return 0.0;
        }
        let mut k = k;
        if is_int(n) {
            if n >= 0.0 {
                if k > n {
                    return 0.0;
                }
                k = k.min(n - k);
                if n <= 170.0 {
                    // n!/(k!(n − k)!) from the table, in double-double.
                    return fact_dd(n).div(fact_dd(k).mul(fact_dd(n - k))).value();
                }
            } else if k < INT && n > -INT {
                // C(n, k) = (−1)^k C(k − n − 1, k).
                let s = if k % 2.0 == 0.0 { 1.0 } else { -1.0 };
                return s * binomial_f64(k - n - 1.0, k);
            }
        }
        if k == 0.0 {
            return 1.0;
        }
        if k <= 32.0 {
            // n(n − 1)…(n − k + 1)/k!, the factors exact as double-doubles.
            let m = k as u32;
            let p = product(n, 0.0, -1.0, m);
            if p.hi.is_finite() {
                return p.div(factorial_dd(m)).value();
            }
        }
    }
    // Γ(n + 1)/(Γ(k + 1) Γ(n − k + 1)) = 1/((n + 1) B(n − k + 1, k + 1)), every argument an exact
    // double-double (v + w = n + 2).
    let u = sum(n, 1.0);
    let v = sum(k, 1.0);
    let (dh, dl) = two_sum(n, -k);
    let w = sum(dh, 1.0).add_f64(dl);
    if u.hi > 0.0 && v.hi > 0.0 && w.hi > 0.0 {
        return exp_f64(ln_of_dd(u).add(lbeta_pos_of_dd(w, v)).neg());
    }
    if is_pole(u) {
        // Γ(n + 1) infinite: over a finite denominator (or with a pole there too), no limit.
        return f64::NAN;
    }
    if is_pole(v) || is_pole(w) {
        return 0.0 * gamma_sign_of_dd(u);
    }
    let (lb, sb) = lbeta_signed(w, v, sum(n, 2.0));
    let (su, au) = if u.hi < 0.0 {
        (-1.0, u.neg())
    } else {
        (1.0, u)
    };
    su * sb * exp_f64(ln_of_dd(au).add(lb).neg())
}

/// `(x)ₙ = Γ(x + n)/Γ(x)` of `f64`s.
pub(super) fn pochhammer_f64(x: f64, n: f64) -> f64 {
    if x.is_nan() || n.is_nan() {
        return x + n;
    }
    if n == 0.0 {
        return 1.0;
    }
    if !(x.is_finite() && n.is_finite()) {
        return f64::NAN;
    }
    if is_int(n) && n.abs() <= 32.0 {
        let m = n.abs() as u32;
        if n > 0.0 {
            return product(x, 0.0, 1.0, m).value();
        }
        // (x)₋ₘ = 1/((x − 1)(x − 2)…(x − m)), a pole where a factor is zero.
        let d = product(x, -1.0, -1.0, m);
        if d.hi == 0.0 {
            return f64::NAN;
        }
        return Dd::new(1.0).div(d).value();
    }
    let c = sum(x, n);
    if x > 0.0 && c.hi > 0.0 {
        let l = if x >= STIRLING_HALF {
            lgamma_delta(x, n).neg()
        } else if c.hi < INT {
            lgamma_of_dd(c).sub(lgamma_pos_dd(x))
        } else {
            Dd::new(lgamma_f64(c.hi)).sub(lgamma_pos_dd(x))
        };
        return exp_f64(l);
    }
    if x <= 0.0 && is_int(x) && is_int(n) {
        // Both poles: (x)ₙ = (−1)ⁿ Γ(1 − x)/Γ(1 − x − n), zero past the end of the product.
        let s = if n % 2.0 == 0.0 { 1.0 } else { -1.0 };
        if n > -x {
            return 0.0;
        }
        return s * pochhammer_f64(1.0 - x - n, n);
    }
    if is_pole(c) {
        // Γ(x + n) infinite (over a finite Γ(x), or with Γ(x) infinite too): no limit.
        return f64::NAN;
    }
    if x <= 0.0 && is_int(x) {
        return 0.0 * gamma_sign_of_dd(c);
    }
    let (l, s) = lpoch_reflected(x, n, c);
    s * exp_f64(l)
}

/// `ln |(x)ₙ|` and the sign of `(x)ₙ` for `x` or `c = x + n` negative (neither a pole), by the
/// reflection formula: Pochhammer symbols (or products) of positive arguments and sines,
///
/// * `x, c < 0`: `(x)ₙ = (sin(πx)/sin(πc)) Γ(1 − x)/Γ(1 − c)`, with `1 − c = (1 − x) − n`,
/// * `x < 0 < c`: `(x)ₙ = Γ(c) Γ(1 − x) sin(πx)/π`,
/// * `c < 0 < x`: `(x)ₙ = π/(sin(πc) Γ(1 − c) Γ(x))`.
fn lpoch_reflected(x: f64, n: f64, c: Dd) -> (Dd, f64) {
    let xd = Dd::new(x);
    if x < 0.0 && c.hi < 0.0 {
        let (lx, sx) = ln_sin_pi(xd);
        let (lc, sc) = ln_sin_pi(c);
        let r = lgamma_ratio(one_minus(xd), Dd::new(-n));
        return (lx.sub(lc).add(r), sx * sc);
    }
    if x < 0.0 {
        let (lx, sx) = ln_sin_pi(xd);
        let l = lgamma_pos_of_dd(c)
            .add(lgamma_pos_of_dd(one_minus(xd)))
            .add(lx)
            .sub(LN_PI);
        return (l, sx);
    }
    let (lc, sc) = ln_sin_pi(c);
    let l = LN_PI
        .sub(lc)
        .sub(lgamma_pos_of_dd(one_minus(c)))
        .sub(lgamma_pos_of_dd(xd));
    (l, sc)
}
