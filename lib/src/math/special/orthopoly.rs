//! The classical orthogonal polynomials.
//!
//! Each polynomial comes from its three-term recurrence, run compensated: the value follows the
//! plain recurrence and a second value carries its error, from the error-free transformations
//! (`two_prod`, `two_sum`) of every step, so that the result is rounded about once. Where the
//! recurrence divides by an integer `c_k` (Legendre, Laguerre, Gegenbauer, associated Legendre
//! and Laguerre) it runs divided by none: `Q_k = p_k Π c_j` and the product beside it, one
//! division at the end. Parameters `α`, `β` on the grid `2^−20` make the coefficients exact sums
//! and their products exact by `two_prod`, a shorter path than double-double arithmetic. The
//! values are rescaled by powers of two as they grow or shrink, so that no intermediate step
//! overflows or underflows (on the size of the value, which lives in the low words where the plain
//! recurrence meets an exact zero). The Chebyshev polynomials of degree above 16 are computed in
//! `O(log n)` by binary powering of `(T₁ + i sin θ U₀)` in double-double.
//!
//! Where a recurrence coefficient vanishes the recurrence decouples or divides by zero, and the
//! values next to `±1` or `0` are small differences of large terms: those parameters go by
//! identities that keep the relative accuracy. Laguerre's `α = −l` by Szegő (5.2.1); Gegenbauer's
//! `α = −l` (zero above degree `2l`) and `α = ½ − l` by the reflection `λ → 1 − λ` of its
//! differential equation; Jacobi's `α` or `β` a negative integer by Szegő (4.22.2), and `α + β`
//! next to `−l`, `2 ≤ l ≤ 30` (where two coefficients of the recurrence are small or zero), by
//! the explicit sum in double-double up to the degree `l` and the recurrence from there. For
//! `|x| > 2^400` a polynomial is its leading term to `2^−336`, or where that is small (Jacobi's
//! `α + β` next to `−l` with `n < l ≤ 2n`) the sum of the two largest terms of the explicit sum.
//!
//! The recurrences take `O(n)` steps; the asymptotic forms for large degrees are left to the
//! release optimisation pass.

use super::Real;
use super::dd::Dd;
use super::elliptic::eft::{mla, res, two_prod, two_sum};

/// `2^e` scaling of an `f64` for any `e`, by steps that cannot overflow on the way.
#[inline]
fn scalbn(mut x: f64, mut e: i32) -> f64 {
    if e == 0 {
        return x;
    }
    let p2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    while e > 1000 {
        x *= p2(1000);
        e -= 1000;
    }
    while e < -1000 {
        x *= p2(-1000);
        e += 1000;
    }
    // Two steps below the subnormal range so that the last multiplication rounds once.
    if e < -1020 {
        x *= p2(-500);
        e += 500;
    }
    x * p2(e)
}

const BIG: f64 = 3.273_390_607_896_142e150; // 2^500
const SMALL: f64 = 3.054_936_363_499_605e-151; // 2^−500

/// The `|x|` above which a polynomial of degree `n ≥ 1` is its leading term `cₙxⁿ` to `2^−336`
/// (its other terms are smaller by `n²/|x|` or less) and goes by `huge_x`; below it no step of a
/// recurrence overflows (`|a_k p_k| < 2^933`).
const HUGE_X: f64 = 2.582_249_878_086_908_6e120; // 2^400

#[inline(always)]
fn dd(x: f64) -> Dd {
    Dd::new(x)
}

#[inline(always)]
fn sum(a: f64, b: f64) -> Dd {
    let (hi, lo) = two_sum(a, b);
    Dd { hi, lo }
}

#[inline(always)]
fn prod(a: f64, b: f64) -> Dd {
    let (hi, lo) = two_prod(a, b);
    Dd { hi, lo }
}

/// `v` brought into `[2^−500, 2^500]` by a power of two, which `e` keeps (for `v` within `2^500`
/// of that range).
#[inline]
fn renorm(v: Dd, e: &mut i32) -> Dd {
    let a = v.hi.abs();
    if a > BIG {
        *e = e.saturating_add(500);
        Dd {
            hi: v.hi * SMALL,
            lo: v.lo * SMALL,
        }
    } else if a < SMALL && a != 0.0 {
        *e = e.saturating_sub(500);
        Dd {
            hi: v.hi * BIG,
            lo: v.lo * BIG,
        }
    } else {
        v
    }
}

/// A double-double times `2^e`, rounded to an `f64`.
#[inline]
fn round_e((v, e): (Dd, i32)) -> f64 {
    scalbn(v.value(), e.clamp(-3000, 3000))
}

/// `(a·p − b·q)/c` as a compensated step: the value `p` follows the plain recurrence and `pl`
/// carries its error, so that only the plain operations are on the chain from step to step and
/// the error terms (exact by `two_prod`, `two_sum` and the residual of the quotient) run beside
/// it. `ic` is `1/c.hi`, computed off the chain.
#[inline(always)]
fn step(a: Dd, (p, pl): (f64, f64), b: Dd, (q, ql): (f64, f64), c: Dd, ic: f64) -> (f64, f64) {
    let (u, e2) = two_prod(a.hi, p);
    let (v, e3) = two_prod(b.hi, q);
    let (w, e4) = two_sum(u, -v);
    let lo = e4 + (e2 - e3) + mla(a.hi, pl, a.lo * p) - mla(b.hi, ql, b.lo * q);
    if c.hi == 1.0 && c.lo == 0.0 {
        (w, lo)
    } else {
        let qn = w * ic;
        let r = res(qn, c.hi, w) - qn * c.lo;
        (qn, (r + lo) * ic)
    }
}

/// The last two values `(p, q)` of a recurrence (each `hi + lo`) rescaled by `2^∓500` when they
/// leave `[2^−500, 2^500]`, the exponent `e` with them (by value: references to them would keep
/// them out of registers in the loops).
#[inline(always)]
fn rescale(p: (f64, f64), q: (f64, f64), e: i32) -> ((f64, f64), (f64, f64), i32) {
    let big = p.0.abs().max(q.0.abs());
    if big > BIG {
        (
            (p.0 * SMALL, p.1 * SMALL),
            (q.0 * SMALL, q.1 * SMALL),
            e + 500,
        )
    } else if big < SMALL {
        rescale_small(p, q, e)
    } else {
        (p, q, e)
    }
}

/// The rare half of `rescale`, the high words small or zero. Where the plain recurrence meets an
/// exact zero (a vanishing coefficient decouples it, or its terms cancel exactly) the value lives
/// in the low words, which would grow unscaled to ∞ and NaN: each pair is renormalised first, so
/// that its high word is its value, and the pair is scaled on that.
#[cold]
#[inline(never)]
fn rescale_small(p: (f64, f64), q: (f64, f64), e: i32) -> ((f64, f64), (f64, f64), i32) {
    let p = two_sum(p.0, p.1);
    let q = two_sum(q.0, q.1);
    let big = p.0.abs().max(q.0.abs());
    if big > BIG {
        (
            (p.0 * SMALL, p.1 * SMALL),
            (q.0 * SMALL, q.1 * SMALL),
            e + 500,
        )
    } else if big < SMALL && big != 0.0 {
        ((p.0 * BIG, p.1 * BIG), (q.0 * BIG, q.1 * BIG), e - 500)
    } else {
        (p, q, e)
    }
}

/// `p_n` of the recurrence `p_(k+1) = (a_k p_k − b_k p_(k−1)) / c_k` from `p₀`, `p₁` and the
/// coefficients `coef(k) = (a_k, b_k, c_k)`, compensated and rescaled: `p_n · 2^(e0 − e)` as the
/// double-double and the exponent `(p, e)`.
#[inline(always)]
fn recur_dd(
    n: u32,
    p0: Dd,
    p1: Dd,
    e0: i32,
    mut coef: impl FnMut(u32) -> (Dd, Dd, Dd),
) -> (Dd, i32) {
    if n == 0 {
        return (p0, e0);
    }
    let (mut q, mut p) = ((p0.hi, p0.lo), (p1.hi, p1.lo));
    let mut e = e0;
    for k in 1..n {
        let (a, b, c) = coef(k);
        let r = step(a, p, b, q, c, 1.0 / c.hi);
        q = p;
        p = r;
        (p, q, e) = rescale(p, q, e);
    }
    let (hi, lo) = two_sum(p.0, p.1);
    (Dd { hi, lo }, e)
}

/// `a·p − b·q` as a compensated step (the division-free form of `step`).
#[inline(always)]
fn step1(a: Dd, (p, pl): (f64, f64), b: Dd, (q, ql): (f64, f64)) -> (f64, f64) {
    let (u, e2) = two_prod(a.hi, p);
    let (v, e3) = two_prod(b.hi, q);
    let (w, e4) = two_sum(u, -v);
    (
        w,
        e4 + (e2 - e3) + mla(a.hi, pl, a.lo * p) - mla(b.hi, ql, b.lo * q),
    )
}

/// `a·p − b·q` for exact `f64` coefficients, compensated.
#[inline(always)]
fn step_ex(a: f64, (p, pl): (f64, f64), b: f64, (q, ql): (f64, f64)) -> (f64, f64) {
    let (u, e2) = two_prod(a, p);
    let (v, e3) = two_prod(b, q);
    let (w, e4) = two_sum(u, -v);
    (w, e4 + (e2 - e3) + mla(a, pl, -b * ql))
}

/// `p_n` of `p_(k+1) = a_k p_k − b_k p_(k−1)`, `p_0 = 1`, with exact `f64` coefficients
/// `coef(k)`, compensated and rescaled.
#[inline(always)]
fn recur_ex(n: u32, p1: f64, mut coef: impl FnMut(u32) -> (f64, f64)) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let (mut q, mut p) = ((1.0, 0.0), (p1, 0.0));
    let mut e = 0;
    for k in 1..n {
        let (a, b) = coef(k);
        let r = step_ex(a, p, b, q);
        q = p;
        p = r;
        (p, q, e) = rescale(p, q, e);
    }
    scalbn(p.0 + p.1, e)
}

/// `p_n` of `c_k p_(k+1) = a_k p_k − b_k p_(k−1)` without its divisions: `Q_k = p_k Π_(j<k) c_j`
/// follows `Q_(k+1) = a_k Q_k − b_k c_(k−1) Q_(k−1)` and the product `D` of the `c_j` is kept
/// beside it, both rescaled; `coef(k) = (a_k, b_k c_(k−1), c_k)` (with `c_0 = 1`) and `c_k` exact.
/// The result is `Q_n / D · 2^(e0 − e)`, as the double-double and the exponent.
#[inline(always)]
fn recur_q_dd(
    n: u32,
    p0: Dd,
    p1: Dd,
    e0: i32,
    mut coef: impl FnMut(u32) -> (Dd, Dd, f64),
) -> (Dd, i32) {
    if n == 0 {
        return (p0, e0);
    }
    let (mut q, mut p) = ((p0.hi, p0.lo), (p1.hi, p1.lo));
    let (mut dh, mut dl) = (1.0f64, 0.0f64);
    let mut e = e0;
    for k in 1..n {
        let (a, b, c) = coef(k);
        let r = step1(a, p, b, q);
        q = p;
        p = r;
        let (h, f) = two_prod(dh, c);
        dl = mla(dl, c, f);
        dh = h;
        (p, q, e) = rescale(p, q, e);
        if dh > BIG {
            dh *= SMALL;
            dl *= SMALL;
            e -= 500;
        }
    }
    // (p + pl)/(dh + dl), rounded once by the caller.
    let q0 = p.0 / dh;
    let r = res(q0, dh, p.0) + p.1 - q0 * dl;
    let (hi, lo) = two_sum(q0, r / dh);
    (Dd { hi, lo }, e)
}

/// `recur_q_dd` rounded to an `f64`.
#[inline(always)]
fn recur_q(n: u32, p0: Dd, p1: Dd, e0: i32, coef: impl FnMut(u32) -> (Dd, Dd, f64)) -> f64 {
    round_e(recur_q_dd(n, p0, p1, e0, coef))
}

/// `±∞` with the sign of `lead · x^n` for an infinite `x`.
#[inline(always)]
fn at_infinity(n: u32, lead_sign: f64, x: f64) -> f64 {
    let s = if x < 0.0 && n % 2 == 1 {
        -lead_sign
    } else {
        lead_sign
    };
    f64::INFINITY * s
}

/// `cₙxⁿ` for `|x| > 2^400` and `n ≥ 1`, the leading coefficient `cₙ = Π_(i=1)^n factor(i)` (the
/// factors nonzero), as a double-double and an exponent; the polynomial is its leading term to
/// `2^−336` there. Above the degree 8 the value is beyond the range of `f64` whatever the leading
/// coefficient of these families (`|x|ⁿ > 2^3600` and `|cₙ| > 2^−(52 + 3n)`): an infinity of its
/// sign.
fn huge_x(n: u32, x: f64, factor: impl Fn(u32) -> Dd) -> (Dd, i32) {
    if n > 8 {
        let mut neg = x < 0.0 && n % 2 == 1;
        for i in 1..=n {
            if factor(i).hi < 0.0 {
                neg = !neg;
            }
        }
        let inf = if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        return (dd(inf), 0);
    }
    // x = xm · 2^ex, 1 ≤ |xm| < 2.
    let ex = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let xm = scalbn(x, -ex);
    let mut v = dd(1.0);
    let mut e = 0i32;
    for i in 1..=n {
        v = renorm(v.mul(factor(i)).mul_f64(xm), &mut e);
        e += ex;
    }
    (v, e)
}

pub(crate) fn legendre_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, 1.0, x);
    }
    if x.abs() > HUGE_X {
        // cₙ = Π (2i − 1)/i.
        return round_e(huge_x(n, x, |i| {
            let i = i as f64;
            dd(2.0 * i - 1.0).div(dd(i))
        }));
    }
    // (k + 1) P_(k+1) = (2k + 1) x P_k − k P_(k−1).
    recur_q(n, dd(1.0), dd(x), 0, |k| {
        let kf = k as f64;
        (prod(2.0 * kf + 1.0, x), prod(kf, kf), kf + 1.0)
    })
}

pub(crate) fn assoc_legendre_f64(n: u32, m: u32, x: f64) -> f64 {
    if x.is_nan() || x.abs() > 1.0 {
        return f64::NAN;
    }
    if m > n {
        return 0.0;
    }
    if m == 0 {
        return legendre_f64(n, x);
    }
    if x.abs() == 1.0 {
        return 0.0;
    }
    // P_m^m = (−1)^m (2m − 1)!! (1 − x²)^(m/2), a product of m factors (2i − 1)√(1 − x²) in
    // double-double, rescaled.
    let x2 = prod(x, x);
    let s = dd(1.0).sub(x2).sqrt();
    let mut v = dd(1.0);
    let mut e = 0i32;
    for i in 1..=m {
        v = s.mul_f64((2 * i - 1) as f64).mul(v);
        let a = v.hi.abs();
        if a > BIG {
            v = Dd {
                hi: v.hi * SMALL,
                lo: v.lo * SMALL,
            };
            e += 500;
        } else if a < SMALL {
            v = Dd {
                hi: v.hi * BIG,
                lo: v.lo * BIG,
            };
            e -= 500;
        }
    }
    if m % 2 == 1 {
        v = v.neg();
    }
    if n == m {
        return scalbn(v.value(), e);
    }
    // P_(m+1)^m = (2m + 1) x P_m^m; (k − m + 1) P_(k+1)^m = (2k + 1) x P_k^m − (k + m) P_(k−1)^m.
    let v1 = prod((2 * m + 1) as f64, x).mul(v);
    recur_q(n - m, v, v1, e, |j| {
        let k = (j + m) as f64;
        let jf = j as f64;
        (prod(2.0 * k + 1.0, x), prod(k + m as f64, jf), jf + 1.0)
    })
}

pub(crate) fn hermite_h_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, 1.0, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, |_| dd(2.0)));
    }
    // H_(k+1) = 2x H_k − 2k H_(k−1).
    let x2 = 2.0 * x;
    recur_ex(n, x2, |k| (x2, (2 * k) as f64))
}

pub(crate) fn hermite_he_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, 1.0, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, |_| dd(1.0)));
    }
    // He_(k+1) = x He_k − k He_(k−1).
    recur_ex(n, x, |k| (x, k as f64))
}

/// The leading-coefficient ratio `c_i/c_(i−1) = −1/i` of the Laguerre polynomials.
#[inline]
fn laguerre_factor(i: u32) -> Dd {
    dd(-1.0).div(dd(i as f64))
}

pub(crate) fn laguerre_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, if n % 2 == 1 { -1.0 } else { 1.0 }, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, laguerre_factor));
    }
    // (k + 1) L_(k+1) = (2k + 1 − x) L_k − k L_(k−1).
    recur_q(n, dd(1.0), sum(1.0, -x), 0, |k| {
        let kf = k as f64;
        (sum(2.0 * kf + 1.0, -x), prod(kf, kf), kf + 1.0)
    })
}

pub(crate) fn assoc_laguerre_f64(n: u32, alpha: f64, x: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if alpha.is_infinite() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return at_infinity(n, if n % 2 == 1 { -1.0 } else { 1.0 }, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, laguerre_factor));
    }
    if alpha < 0.0 && alpha == alpha.round() && -alpha <= n as f64 {
        // α = −l, n ≥ l: L_n^(−l)(x) = (−x)^l (n − l)!/n! L_(n−l)^(l)(x) (Szegő (5.2.1)). The
        // recurrence of α itself decouples at k = l (its coefficient k + α vanishes), and its
        // values next to x = 0 are small differences of large terms.
        let l = (-alpha) as u32;
        let (mut v, mut e) = assoc_laguerre_dd(n - l, l as f64, x);
        for i in 0..l {
            v = renorm(v.mul_f64(-x).div(dd((n - i) as f64)), &mut e);
        }
        return round_e((v, e));
    }
    round_e(assoc_laguerre_dd(n, alpha, x))
}

/// `L_n^(α)(x)` by its recurrence, as a double-double and an exponent (`|x| ≤ 2^400`).
fn assoc_laguerre_dd(n: u32, alpha: f64, x: f64) -> (Dd, i32) {
    // (k + 1) L_(k+1) = (2k + 1 + α − x) L_k − (k + α) L_(k−1), L₁ = 1 + α − x.
    let l1 = sum(1.0, alpha).add_f64(-x);
    if on_grid(alpha) {
        // k + α exact: the coefficients by two_sum and two_prod of exact values.
        return recur_q_dd(n, dd(1.0), l1, 0, |k| {
            let kf = k as f64;
            (
                sum((2.0 * kf + 1.0) + alpha, -x),
                prod(kf + alpha, kf),
                kf + 1.0,
            )
        });
    }
    recur_q_dd(n, dd(1.0), l1, 0, |k| {
        let kf = k as f64;
        (
            sum(2.0 * kf + 1.0, alpha).add_f64(-x),
            sum(kf, alpha).mul_f64(kf),
            kf + 1.0,
        )
    })
}

/// `(T_n(x), U_(n−1)(x), e)` with the values scaled by `2^−e`, by binary powering of
/// `z = T₁ + i s U₀` (`s² = 1 − x²`) in double-double: `T_(a+b) = T_a T_b − s² U_(a−1) U_(b−1)`,
/// `U_(a+b−1) = T_a U_(b−1) + U_(a−1) T_b`.
fn cheb_ladder(n: u32, x: f64) -> (Dd, Dd, i32) {
    let w = dd(1.0).sub(prod(x, x)); // 1 − x²
    let (mut t, mut u) = (dd(x), dd(1.0));
    let mut e = 0i32;
    // The values stay below 2^200 (times |x| ≤ 2^66, by the guard of the callers).
    let rescale = |t: &mut Dd, u: &mut Dd, e: &mut i32| {
        if t.hi.abs().max(u.hi.abs()) > 1.606_938_044_258_990_3e60 {
            let s = 6.223_015_277_861_142e-61; // 2^−200
            *t = Dd {
                hi: t.hi * s,
                lo: t.lo * s,
            };
            *u = Dd {
                hi: u.hi * s,
                lo: u.lo * s,
            };
            *e += 200;
        }
    };
    let top = 31 - n.leading_zeros();
    for bit in (0..top).rev() {
        // Square: the scale 2^e squares with the values.
        e = e.saturating_mul(2).min(1 << 20);
        let t2 = t.mul(t).sub(w.mul(u.mul(u)));
        let u2 = t.mul(u).mul_f64(2.0);
        t = t2;
        u = u2;
        rescale(&mut t, &mut u, &mut e);
        if n >> bit & 1 == 1 {
            // Times z₁ = (x, 1).
            let t3 = t.mul_f64(x).sub(w.mul(u));
            let u3 = t.add(u.mul_f64(x));
            t = t3;
            u = u3;
            rescale(&mut t, &mut u, &mut e);
        }
    }
    (t, u, e)
}

/// Whether `T_n(x)` and `U_n(x)` overflow for `|x| > 1`: `n log₂(|x| + √(x² − 1)) > 1100`
/// (`log₂ 2|x|` to `2^−200` for `|x| > 2^100`, where `x²` would overflow).
#[inline]
fn cheb_overflows(n: u32, x: f64) -> bool {
    let a = x.abs();
    if a <= 1.0 {
        return false;
    }
    let l = if a < 1.267_650_600_228_229_4e30 {
        (a + ((a - 1.0) * (a + 1.0)).sqrt()).log2()
    } else {
        1.0 + a.log2()
    };
    n as f64 * l > 1100.0
}

/// The degree above which the Chebyshev polynomials go by binary powering.
const CHEB_LADDER: u32 = 16;

pub(crate) fn chebyshev_t_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, 1.0, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, |i| dd(if i == 1 { 1.0 } else { 2.0 })));
    }
    if cheb_overflows(n, x) {
        return at_infinity(n, 1.0, x);
    }
    if n > CHEB_LADDER {
        let (t, _, e) = cheb_ladder(n, x);
        return scalbn(t.value(), e);
    }
    // T_(k+1) = 2x T_k − T_(k−1).
    let x2 = 2.0 * x;
    recur_ex(n, x, |_| (x2, 1.0))
}

pub(crate) fn chebyshev_u_f64(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return at_infinity(n, 1.0, x);
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, |_| dd(2.0)));
    }
    if cheb_overflows(n, x) {
        return at_infinity(n, 1.0, x);
    }
    if n > CHEB_LADDER {
        // U_n = x U_(n−1) + T_n.
        let (t, u, e) = cheb_ladder(n, x);
        return scalbn(u.mul_f64(x).add(t).value(), e);
    }
    let x2 = 2.0 * x;
    recur_ex(n, x2, |_| (x2, 1.0))
}

/// Whether a parameter is on the grid `2^−20` with magnitude at most `2^28`: then every sum of it
/// with integers up to `2^31` (and of two such parameters) in a recurrence coefficient is exact,
/// and the products of two such sums are exact double-doubles by `two_prod`.
#[inline(always)]
fn on_grid(v: f64) -> bool {
    v.abs() <= 268_435_456.0 && (v * 1_048_576.0).fract() == 0.0
}

/// The sign of `Π_(i=lo)^(hi) (a + i)`, 0 if a factor vanishes.
fn product_sign(a: f64, lo: u32, hi: u32) -> f64 {
    let mut s = 1.0;
    for i in lo..=hi {
        let f = a + i as f64;
        if f == 0.0 {
            return 0.0;
        }
        if f < 0.0 {
            s = -s;
        }
    }
    s
}

/// Whether `v` is a negative integer of magnitude at most `n`.
#[inline]
fn neg_int_upto(v: f64, n: u32) -> bool {
    v < 0.0 && v == v.round() && -v <= n as f64
}

pub(crate) fn jacobi_p_f64(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() || beta.is_nan() {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if alpha.is_infinite() || beta.is_infinite() {
        return f64::NAN;
    }
    if x.is_infinite() {
        // The leading coefficient (n + α + β + 1)_n / (2^n n!).
        let s = product_sign(n as f64 + (alpha + beta), 1, n);
        return if s == 0.0 {
            f64::NAN
        } else {
            at_infinity(n, s, x)
        };
    }
    round_e(jacobi_dd(n, alpha, beta, x))
}

/// `P_n^(α,β)(x)` as a double-double and an exponent: by the identities of the parameters at
/// which the recurrence decouples or divides by zero, and the recurrence elsewhere.
fn jacobi_dd(n: u32, alpha: f64, beta: f64, x: f64) -> (Dd, i32) {
    if n == 0 {
        return (dd(1.0), 0);
    }
    if neg_int_upto(alpha, n) {
        // α = −l, n ≥ l: Szegő (4.22.2), P_n^(−l,β)(x) = [C(n + β, l)/C(n, l)] ((x − 1)/2)^l
        // P_(n−l)^(l,β)(x); the recurrence decouples at k = l + 1 (its b_k vanishes), and its
        // values next to x = 1 are small differences of large terms.
        let l = (-alpha) as u32;
        let mut r = dd(1.0);
        for i in 0..l {
            r = r
                .mul(sum(n as f64 - i as f64, beta))
                .div(dd((n - i) as f64));
        }
        if r.hi == 0.0 {
            return (dd(0.0), 0);
        }
        let (mut v, mut e) = jacobi_dd(n - l, l as f64, beta, x);
        let y = sum(x, -1.0).mul_f64(0.5);
        if v.hi.is_infinite() {
            // Past the range (|x| > 2^400): the sign of the product.
            let s = if y.hi < 0.0 && l % 2 == 1 { -1.0 } else { 1.0 };
            return (dd(v.hi * r.hi.signum() * s), 0);
        }
        // y = y_m·2^(e_y), so that v·y cannot overflow for |x| up to f64::MAX (exactly: a power
        // of two apart).
        let ey = if y.hi == 0.0 {
            0
        } else {
            ((y.hi.to_bits() >> 52) & 0x7ff) as i32 - 1023
        };
        let s = scalbn(1.0, -ey);
        let ym = Dd {
            hi: y.hi * s,
            lo: y.lo * s,
        };
        v = renorm(v.mul(r), &mut e);
        for _ in 0..l {
            v = renorm(v.mul(ym), &mut e);
            e = e.saturating_add(ey);
        }
        return (v, e);
    }
    if neg_int_upto(beta, n) {
        // P_n^(α,β)(x) = (−1)^n P_n^(β,α)(−x) (DLMF 18.6.1).
        let (v, e) = jacobi_dd(n, beta, alpha, -x);
        return (if n % 2 == 1 { v.neg() } else { v }, e);
    }
    let abd = sum(alpha, beta);
    let lead = |i: u32| abd.add_f64(n as f64 + i as f64).div(dd(2.0 * i as f64));
    // α + β = −l + δ, l ≥ 2: the coefficient c_k = 2k(k + α + β)(2k + α + β − 2) of the
    // recurrence is δ times about its usual size, or zero, at k = l and k = (l + 2)/2, and the
    // recurrence through it loses more as δ shrinks or l grows (10⁵ ulps of S at l = 4,
    // δ = 10⁻⁶; 10 at l = 8 even for δ = 0.32): for 5 ≤ l ≤ 30, where the explicit sum to the
    // degree l is accurate, the sum and the recurrence beyond it for every δ; for l = 2..4 where
    // |δ| < 1/64, and from the degree 16 on where |δ| < 1/16 (the recurrence was within 0.5 ulps
    // of S below the degree 16 for 1/64 ≤ |δ| ≤ 1/2, and from it for 1/16 ≤ |δ| ≤ 1/4 at and
    // next to x = ±1, in the range α, β ∈ (−5, 10), where the sum costs up to 8 times as much
    // at low degrees and twice at the degree 20; between, it reached 1.35 ulps of S just past
    // 1/64, at x = −1 and n = 232); for larger l only within 2^−40 l (where the recurrence is
    // hopeless).
    let l = (-abd.hi).round();
    let delta = (abd.hi + l) + abd.lo;
    if x.abs() > HUGE_X {
        if l >= 2.0 && l > n as f64 && l <= 2.0 * n as f64 {
            // The leading coefficient (n + α + β + 1)_n/(2^n n!) has the factor δ.
            return jacobi_huge_near(n, alpha, beta, x, l as u32);
        }
        return huge_x(n, x, lead);
    }
    let near = delta.abs() <= l * 9.094_947_017_729_282e-13;
    let sum_l = l >= 5.0 || delta.abs() < if n >= 16 { 0.0625 } else { 0.015_625 };
    if l >= 2.0 && ((l <= 30.0 && sum_l) || near) {
        if n as f64 >= l {
            // The recurrence from P_(l−1) and P_l, by the explicit sum.
            let l = l as u32;
            let (p0, e0) = jacobi_sum_dd(l - 1, alpha, beta, x);
            let (p1, e1) = jacobi_sum_dd(l, alpha, beta, x);
            // Both to the exponent of the larger (P_(l−1) is a constant when α + β = −l).
            let e = e0.max(e1);
            let at = |p: Dd, ep: i32| {
                let s = scalbn(1.0, ep - e);
                Dd {
                    hi: p.hi * s,
                    lo: p.lo * s,
                }
            };
            let (v, er) = jacobi_recur(n - l + 1, l - 1, alpha, beta, x, at(p0, e0), at(p1, e1));
            return (v, er.saturating_add(e));
        }
        if l % 2.0 == 0.0 && n as f64 >= (l + 2.0) * 0.5 {
            return jacobi_sum_dd(n, alpha, beta, x);
        }
        // Below both: the recurrence reaches neither.
    }
    // P₁ = (α + 1) + (α + β + 2)(x − 1)/2.
    let xm1 = sum(x, -1.0);
    let p1 = sum(alpha, 1.0).add(abd.add_f64(2.0).mul(xm1).mul_f64(0.5));
    jacobi_recur(n, 0, alpha, beta, x, dd(1.0), p1)
}

/// `P_(off+N)^(α,β)(x)` by the recurrence from `p₀ = P_off` and `p₁ = P_(off+1)`, as a
/// double-double and an exponent.
fn jacobi_recur(steps: u32, off: u32, alpha: f64, beta: f64, x: f64, p0: Dd, p1: Dd) -> (Dd, i32) {
    let ab = alpha + beta;
    let abd = sum(alpha, beta);
    if on_grid(alpha) && on_grid(beta) {
        let (h, l) = two_prod(alpha - beta, ab);
        let a2b2 = Dd { hi: h, lo: l };
        return recur_dd(steps, p0, p1, 0, |k| {
            let k1 = (k + 1 + off) as f64;
            let s = 2.0 * k1 + ab; // 2k + α + β
            let (s1, s2) = (s - 1.0, s - 2.0);
            let c = prod(k1 + ab, s2).mul_f64(2.0 * k1);
            let a = prod(s, s2).mul_f64(x).add(a2b2).mul_f64(s1);
            let b = prod(k1 - 1.0 + alpha, k1 - 1.0 + beta).mul_f64(2.0 * s);
            (a, b, c)
        });
    }
    let a2b2 = sum(alpha, -beta).mul(abd); // α² − β²
    // 2k(k+α+β)(2k+α+β−2) P_k = (2k+α+β−1)[(2k+α+β)(2k+α+β−2)x + α²−β²] P_(k−1)
    //                            − 2(k+α−1)(k+β−1)(2k+α+β) P_(k−2), here with k → k + 1 + off.
    recur_dd(steps, p0, p1, 0, |k| {
        let k1 = (k + 1 + off) as f64;
        let s = abd.add_f64(2.0 * k1); // 2k + α + β
        let s2 = s.add_f64(-2.0);
        let s1 = s.add_f64(-1.0);
        let c = abd.add_f64(k1).mul(s2).mul_f64(2.0 * k1);
        let a = s1.mul(s.mul(s2).mul_f64(x).add(a2b2));
        let b = sum(k1 - 1.0, alpha)
            .mul(sum(k1 - 1.0, beta))
            .mul(s)
            .mul_f64(2.0);
        (a, b, c)
    })
}

/// `P_n^(α,β)(x) = Σ_j d_j yʲ`, `y = (x − 1)/2`, `d₀ = (α + 1)_n/n!`,
/// `d_j/d_(j−1) = (n + α + β + j)(n − j + 1)/(j (α + j))` (DLMF 18.5.7), in double-double by
/// Horner's rule from the last nonzero coefficient, for the low degrees at which the recurrence
/// would divide by zero or nearly (`α + β` next to `−l`, degree up to `l`; `α` not a negative
/// integer, which goes by Szegő (4.22.2) first), `|x| ≤ 2^400`. The terms alternate on `[−1, 1]`
/// and grow with the degree: accurate for degrees up to about 30.
fn jacobi_sum_dd(n: u32, alpha: f64, beta: f64, x: f64) -> (Dd, i32) {
    let nab = sum(alpha, beta).add_f64(n as f64); // n + α + β
    // d_j vanishes from the first j with n + α + β + j = 0.
    let mut top = n;
    for j in 1..=n {
        if nab.add_f64(j as f64).hi == 0.0 {
            top = j - 1;
            break;
        }
    }
    let mut d = dd(1.0);
    for i in 1..=n {
        d = d.mul(sum(alpha, i as f64)).div(dd(i as f64));
    }
    for j in 1..=top {
        let jf = j as f64;
        d = d
            .mul(nab.add_f64(jf))
            .mul_f64((n - j + 1) as f64)
            .div(sum(alpha, jf).mul_f64(jf));
    }
    let y = sum(x, -1.0).mul_f64(0.5);
    // Horner's rule with the running value p·2^e (it grows as yʲ for |x| > 3).
    let (mut p, mut e) = (d, 0i32);
    for j in (1..=top).rev() {
        let jf = j as f64;
        d = d
            .mul(sum(alpha, jf).mul_f64(jf))
            .div(nab.add_f64(jf).mul_f64((n - j + 1) as f64));
        p = renorm(p.mul(y), &mut e);
        let s = if e == 0 { 1.0 } else { scalbn(1.0, -e) };
        p = p.add(Dd {
            hi: d.hi * s,
            lo: d.lo * s,
        });
    }
    (p, e)
}

/// `P_n^(α,β)(x)` for `|x| > 2^400` and `α + β = −l + δ` (`l` the nearest integer) with
/// `n < l ≤ 2n`, where the leading coefficient `(n + α + β + 1)_n/(2^n n!)` has the factor
/// `n + α + β + (l − n) = δ`, small or zero: the coefficients `d_j` of DLMF 18.5.7 (as in
/// `jacobi_sum_dd`) carry it from `j = l − n` on, and the polynomial is `d_t yᵗ + d_n yⁿ`,
/// `t = l − n − 1`, `y = (x − 1)/2`, the other terms smaller than one of these two by `|y|` over a
/// ratio of coefficients (`2^−280` or less of it in the documented range). Each term with its
/// exponent apart; `α` is not a negative integer (Szegő (4.22.2) goes first).
fn jacobi_huge_near(n: u32, alpha: f64, beta: f64, x: f64, l: u32) -> (Dd, i32) {
    let nab = sum(alpha, beta).add_f64(n as f64); // n + α + β
    let t = l - n - 1;
    // d_0 = (α + 1)_n/n!, then d_j = d_(j−1) (n + α + β + j)(n − j + 1)/(j (α + j)).
    let (mut d, mut e) = (dd(1.0), 0i32);
    for i in 1..=n {
        d = renorm(d.mul(sum(alpha, i as f64)).div(dd(i as f64)), &mut e);
    }
    let mut dt = (d, e);
    for j in 1..=n {
        let jf = j as f64;
        d = d
            .mul(nab.add_f64(jf))
            .mul_f64((n - j + 1) as f64)
            .div(sum(alpha, jf).mul_f64(jf));
        d = renorm(d, &mut e);
        if j == t {
            dt = (d, e);
        }
    }
    // y = (x − 1)/2 is x/2 to 2^−400; y = y_m·2^(e_y), its powers with the exponent apart.
    let y = 0.5 * x;
    let ey = ((y.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let ym = scalbn(y, -ey);
    let pow = |k: u32, (c, ec): (Dd, i32)| {
        let (mut v, mut ev) = (c, ec);
        for _ in 0..k {
            v = renorm(v.mul_f64(ym), &mut ev);
            ev = ev.saturating_add(ey);
        }
        (v, ev)
    };
    let (a, ea) = pow(t, dt);
    let (b, eb) = pow(n, (d, e));
    if b.hi == 0.0 {
        return (a, ea);
    }
    if a.hi == 0.0 {
        return (b, eb);
    }
    // Their sum at the larger exponent (the smaller term perhaps lost below it).
    let (big, small, es) = if ea >= eb {
        ((a, ea), b, eb)
    } else {
        ((b, eb), a, ea)
    };
    let s = scalbn(1.0, es.saturating_sub(big.1).max(-1100));
    let small = Dd {
        hi: small.hi * s,
        lo: small.lo * s,
    };
    (big.0.add(small), big.1)
}

/// The leading-coefficient ratio `c_i/c_(i−1) = 2(α + i − 1)/i` of the Gegenbauer polynomials.
#[inline]
fn gegenbauer_factor(alpha: f64) -> impl Fn(u32) -> Dd {
    move |i| sum(alpha, (i - 1) as f64).mul_f64(2.0).div(dd(i as f64))
}

pub(crate) fn gegenbauer_f64(n: u32, alpha: f64, x: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if alpha.is_infinite() {
        return f64::NAN;
    }
    if alpha == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        // The leading coefficient 2^n (α)_n / n!.
        let s = product_sign(alpha, 0, n - 1);
        return if s == 0.0 {
            f64::NAN
        } else {
            at_infinity(n, s, x)
        };
    }
    let t = 2.0 * alpha;
    if t < 0.0 && t == t.round() {
        // 2α = −j: the coefficient (k + 2α − 1)k of the recurrence vanishes at k = j + 1.
        if alpha == alpha.round() {
            // α = −l: C_n^(−l) ≡ 0 for n > 2l, as (−l)_(n−k) = 0 for n − k > l in DLMF 18.5.10;
            // below, the recurrence does not reach its vanishing coefficient.
            let l = -alpha;
            if n as f64 > 2.0 * l {
                return 0.0;
            }
            if x.abs() > HUGE_X && n as f64 > l {
                // The leading coefficient vanishes: the term k = n − l of DLMF 18.5.10,
                // (−1)^n C(l, n − l) (2x)^(2l − n), is the largest.
                let (l, d) = (l as u32, (2.0 * l) as u32 - n);
                let mut c = dd(if n % 2 == 1 { -1.0 } else { 1.0 });
                for i in 0..(n - l) {
                    c = c.mul_f64((l - i) as f64).div(dd((i + 1) as f64));
                }
                let (v, mut e) = huge_x(d, x, |_| dd(2.0));
                return round_e((renorm(v.mul(c), &mut e), e));
            }
        } else if n as f64 >= 1.0 - t {
            // α = ½ − l, n ≥ 2l: C_n^(½−l)(x) = 4^l ((½)_l)² (n − 2l)!/n! (1 − x²)^l C_(n−2l)^(l+½)(x)
            // (the reflection λ → 1 − λ of the differential equation of C_n^(λ)). The recurrence of
            // α itself decouples at k = 2l, and its values next to ±1 are small differences of
            // large terms.
            if x.abs() > HUGE_X {
                return round_e(huge_x(n, x, gegenbauer_factor(alpha)));
            }
            let l = ((1.0 - t) * 0.5) as u32;
            let (mut v, mut e) = gegenbauer_dd(n - 2 * l, l as f64 + 0.5, x);
            let w = dd(1.0).sub(prod(x, x)); // 1 − x², exact
            for i in 0..l {
                // 4 (½ + i)² (1 − x²) / ((n − 2i)(n − 2i − 1)).
                let h = 0.5 + i as f64;
                let k = (n - 2 * i) as f64;
                let f = w.mul_f64(4.0 * h * h).div(dd(k).mul_f64(k - 1.0));
                v = renorm(v.mul(f), &mut e);
            }
            return round_e((v, e));
        }
    }
    if x.abs() > HUGE_X {
        return round_e(huge_x(n, x, gegenbauer_factor(alpha)));
    }
    round_e(gegenbauer_dd(n, alpha, x))
}

/// `C_n^(α)(x)` by its recurrence, as a double-double and an exponent (`|x| ≤ 2^400`).
fn gegenbauer_dd(n: u32, alpha: f64, x: f64) -> (Dd, i32) {
    // (k + 1) C_(k+1) = 2(k + α) x C_k − (k + 2α − 1) C_(k−1), C₁ = 2αx.
    let c1 = prod(2.0 * alpha, x);
    if on_grid(alpha) {
        return recur_q_dd(n, dd(1.0), c1, 0, |k| {
            let kf = k as f64;
            (
                prod(2.0 * (kf + alpha), x),
                prod(2.0 * alpha + kf - 1.0, kf),
                kf + 1.0,
            )
        });
    }
    recur_q_dd(n, dd(1.0), c1, 0, |k| {
        let kf = k as f64;
        (
            sum(kf, alpha).mul_f64(2.0 * x),
            sum(kf - 1.0, 2.0 * alpha).mul_f64(kf),
            kf + 1.0,
        )
    })
}

/// The Legendre polynomial `Pₙ(x)`.
pub fn legendre_p<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| legendre_f64(n, x))
}

/// The associated Legendre function `Pₙᵐ(x)` for `|x| ≤ 1`, with the Condon–Shortley phase
/// `(−1)ᵐ` (as SciPy's `lpmv`); 0 for `m > n`.
pub fn assoc_legendre_p<T: Real>(n: u32, m: u32, x: T) -> T {
    super::via(x, |x| assoc_legendre_f64(n, m, x))
}

/// The physicists' Hermite polynomial `Hₙ(x)`.
pub fn hermite_h<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| hermite_h_f64(n, x))
}

/// The probabilists' Hermite polynomial `Heₙ(x)`.
pub fn hermite_he<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| hermite_he_f64(n, x))
}

/// The Laguerre polynomial `Lₙ(x)`.
pub fn laguerre_l<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| laguerre_f64(n, x))
}

/// The generalised Laguerre polynomial `Lₙ^(α)(x)`.
pub fn assoc_laguerre_l<T: Real>(n: u32, alpha: T, x: T) -> T {
    T::from_f64(assoc_laguerre_f64(n, alpha.to_f64(), x.to_f64()))
}

/// The Chebyshev polynomial of the first kind `Tₙ(x)`.
pub fn chebyshev_t<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| chebyshev_t_f64(n, x))
}

/// The Chebyshev polynomial of the second kind `Uₙ(x)`.
pub fn chebyshev_u<T: Real>(n: u32, x: T) -> T {
    super::via(x, |x| chebyshev_u_f64(n, x))
}

/// The Jacobi polynomial `Pₙ^(α,β)(x)`.
pub fn jacobi_p<T: Real>(n: u32, alpha: T, beta: T, x: T) -> T {
    T::from_f64(jacobi_p_f64(n, alpha.to_f64(), beta.to_f64(), x.to_f64()))
}

/// The Gegenbauer (ultraspherical) polynomial `Cₙ^(α)(x)`.
pub fn gegenbauer_c<T: Real>(n: u32, alpha: T, x: T) -> T {
    T::from_f64(gegenbauer_f64(n, alpha.to_f64(), x.to_f64()))
}
