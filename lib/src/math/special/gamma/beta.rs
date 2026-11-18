//! The beta function and its logarithm, and factorials, binomial coefficients and Pochhammer's
//! symbol of floats.
//!
//! * `ln B(a, b)` for `a, b > 0`: `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` in double-double below 8
//!   (`a + b` as a double-double); `ln Γ(a) + [ln Γ(b) − ln Γ(a + b)]` by the Stirling difference
//!   when only `b ≥ 8`; and for both at least 8 the symmetric form
//!   `ln √(2π) + (a − ½) ln(a/c) + (b − ½) ln(b/c) − ½ ln c + S(a) + S(b) − S(c)` with
//!   `ln(b/c) = −ln(1 + a/b)` taken directly, so no term cancels another.
//! * Other real arguments through the signs and logarithms of the three gamma functions.
//! * `x! = Γ(x + 1)` without rounding `x + 1`; `C(n, k)` and `(x)ₙ` by exact double-double
//!   products for small integer `k` or `n`, else by `ln B` and the Stirling difference.

use super::super::dd::{Dd, fast_two_sum, two_sum};
use super::ddm::*;
use super::ibeta::{lgamma_delta, lgamma_of_dd, phi, rgamma_dd, stirling_delta, stirling_rem};
use super::psi::digamma_f64;
use super::real::{gamma_shift, lgamma_f64, lgamma_neg_dd, lgamma_pos_dd};
use super::tab::*;

const LN_SQRT_2PI: Dd = Dd {
    hi: LN_SQRT_2PI_HI,
    lo: LN_SQRT_2PI_LO,
};

/// `2⁵²`: every float beyond it is an integer.
const INT: f64 = 4_503_599_627_370_496.0;

/// `x` is an integer.
#[inline(always)]
fn is_int(x: f64) -> bool {
    x == x.floor()
}

/// `ln(1 + q)` of a double-double `q ≥ 0`, accurate relative to its value.
#[inline]
fn ln1p_of_dd(q: Dd) -> Dd {
    phi(q.add_f64(1.0), q).add(q)
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
    let c = a + b;
    let s = if c.is_finite() {
        stirling_delta(a, b).neg()
    } else {
        stirling_rem(a).add(stirling_rem(b))
    };
    // (a − ½) ln(a/c) − (b − ½) L₁ − ½ ln c + ln √(2π) + S(a) + S(b) − S(c).
    la_c.mul_f64(a)
        .sub(la_c.mul_f64(0.5))
        .sub(l1.mul_f64(b))
        .add(l1.mul_f64(0.5))
        .sub(lc.mul_f64(0.5))
        .add(LN_SQRT_2PI)
        .add(s)
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

/// `ln |Γ(x)|` of a non-pole `x`, as a double-double.
#[inline]
fn lgamma_any_dd(x: f64) -> Dd {
    if x > 0.0 {
        if x < INT {
            lgamma_pos_dd(x)
        } else {
            Dd::new(lgamma_f64(x))
        }
    } else {
        lgamma_neg_dd(x, false)
    }
}

/// `ln |Γ(hi + lo)|` of a non-pole double-double, by `ln |Γ(hi)| + lo·ψ(hi)`.
#[inline]
fn lgamma_any_of_dd(c: Dd) -> Dd {
    let l = lgamma_any_dd(c.hi);
    if c.lo == 0.0 {
        l
    } else {
        l.add_f64(c.lo * digamma_f64(c.hi))
    }
}

/// How the gamma functions of a ratio meet their poles.
enum Poles {
    /// None of them: the ratio's logarithm and sign.
    Finite(Dd, f64),
    /// The ratio is zero (a pole below only), with its sign.
    Zero(f64),
    /// The ratio is infinite (a pole above only).
    Infinite,
    /// Poles above and below: no limit.
    Undefined,
}

/// `Γ(u)/(Γ(v)Γ(w))` (`w` absent when `None`), the arguments double-doubles, by their poles or
/// their logarithms and signs.
fn gamma_ratio(u: Dd, v: Dd, w: Option<Dd>) -> Poles {
    let pole = |d: Dd| d.lo == 0.0 && d.hi <= 0.0 && is_int(d.hi);
    let (pu, pv, pw) = (pole(u), pole(v), w.is_some_and(pole));
    if pu {
        return if pv || pw {
            Poles::Undefined
        } else {
            Poles::Infinite
        };
    }
    let su = gamma_sign(u.hi);
    if pv || pw {
        return Poles::Zero(su);
    }
    let mut l = lgamma_any_of_dd(u).sub(lgamma_any_of_dd(v));
    let mut s = su * gamma_sign(v.hi);
    if let Some(w) = w {
        l = l.sub(lgamma_any_of_dd(w));
        s *= gamma_sign(w.hi);
    }
    Poles::Finite(l, s)
}

/// `x + y` as a double-double.
#[inline(always)]
fn sum(x: f64, y: f64) -> Dd {
    let (hi, lo) = two_sum(x, y);
    Dd { hi, lo }
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
    match gamma_ratio(Dd::new(a), sum(a, b), None) {
        // Γ(a)Γ(b)/Γ(a + b) as Γ(a)/Γ(a + b) · Γ(b).
        Poles::Finite(l, s) => {
            if b < 0.0 && is_int(b) {
                return f64::NAN;
            }
            s * gamma_sign(b) * exp_f64(l.add(lgamma_any_dd(b)))
        }
        Poles::Zero(s) => {
            if b < 0.0 && is_int(b) {
                return f64::NAN;
            }
            0.0 * s * gamma_sign(b)
        }
        Poles::Infinite | Poles::Undefined => f64::NAN,
    }
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
    let pc = c.lo == 0.0 && pole(c.hi);
    match (pole(a) || pole(b), pc) {
        (true, true) => f64::NAN,
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => lgamma_any_dd(a)
            .add(lgamma_any_dd(b))
            .sub(lgamma_any_of_dd(c))
            .value(),
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
    // Γ(n + 1)/(Γ(k + 1) Γ(n − k + 1)) = 1/((n + 1) B(n − k + 1, k + 1)) for positive arguments.
    let u = sum(n, 1.0);
    let v = sum(k, 1.0);
    // n − k + 1 as a double-double.
    let (dh, dl) = two_sum(n, -k);
    let w = sum(dh, 1.0).add_f64(dl);
    if u.hi > 0.0 && v.hi > 0.0 && w.hi > 0.0 {
        // The low parts of v and w by the derivatives of ln B: ψ(v) − ψ(v + w) and the like.
        let mut lb = lbeta_pos_dd(w.hi, v.hi);
        if v.lo != 0.0 || w.lo != 0.0 {
            let d = digamma_f64(v.hi + w.hi);
            lb = lb.add_f64(v.lo * (digamma_f64(v.hi) - d) + w.lo * (digamma_f64(w.hi) - d));
        }
        return exp_f64(ln_of_dd(u).add(lb).neg());
    }
    match gamma_ratio(u, v, Some(w)) {
        Poles::Finite(l, s) => s * exp_f64(l),
        Poles::Zero(s) => 0.0 * s,
        Poles::Infinite | Poles::Undefined => f64::NAN,
    }
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
    match gamma_ratio(c, Dd::new(x), None) {
        Poles::Finite(l, s) => s * exp_f64(l),
        Poles::Zero(s) => 0.0 * s,
        Poles::Infinite | Poles::Undefined => f64::NAN,
    }
}
