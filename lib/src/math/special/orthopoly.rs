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
//! overflows or underflows. The Chebyshev polynomials of degree above 16 are computed in
//! `O(log n)` by binary powering of `(T₁ + i sin θ U₀)` in double-double.
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

/// `p_n` of the recurrence `p_(k+1) = (a_k p_k − b_k p_(k−1)) / c_k` from `p₀`, `p₁` and the
/// coefficients `coef(k) = (a_k, b_k, c_k)`, compensated and rescaled; the result is `p_n · 2^e0`.
#[inline(always)]
fn recur(n: u32, p0: Dd, p1: Dd, e0: i32, mut coef: impl FnMut(u32) -> (Dd, Dd, Dd)) -> f64 {
    if n == 0 {
        return scalbn(p0.value(), e0);
    }
    let (mut q, mut p) = ((p0.hi, p0.lo), (p1.hi, p1.lo));
    let mut e = e0;
    for k in 1..n {
        let (a, b, c) = coef(k);
        let r = step(a, p, b, q, c, 1.0 / c.hi);
        q = p;
        p = r;
        let big = p.0.abs().max(q.0.abs());
        if big > BIG {
            p = (p.0 * SMALL, p.1 * SMALL);
            q = (q.0 * SMALL, q.1 * SMALL);
            e += 500;
        } else if big < SMALL && big != 0.0 {
            p = (p.0 * BIG, p.1 * BIG);
            q = (q.0 * BIG, q.1 * BIG);
            e -= 500;
        }
    }
    scalbn(p.0 + p.1, e)
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
        let big = p.0.abs().max(q.0.abs());
        if big > BIG {
            p = (p.0 * SMALL, p.1 * SMALL);
            q = (q.0 * SMALL, q.1 * SMALL);
            e += 500;
        } else if big < SMALL && big != 0.0 {
            p = (p.0 * BIG, p.1 * BIG);
            q = (q.0 * BIG, q.1 * BIG);
            e -= 500;
        }
    }
    scalbn(p.0 + p.1, e)
}

/// `p_n` of `c_k p_(k+1) = a_k p_k − b_k p_(k−1)` without its divisions: `Q_k = p_k Π_(j<k) c_j`
/// follows `Q_(k+1) = a_k Q_k − b_k c_(k−1) Q_(k−1)` and the product `D` of the `c_j` is kept
/// beside it, both rescaled; `coef(k) = (a_k, b_k c_(k−1), c_k)` (with `c_0 = 1`) and `c_k` exact.
/// The result is `Q_n / D · 2^e0`.
#[inline(always)]
fn recur_q(n: u32, p0: Dd, p1: Dd, e0: i32, mut coef: impl FnMut(u32) -> (Dd, Dd, f64)) -> f64 {
    if n == 0 {
        return scalbn(p0.value(), e0);
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
        let big = p.0.abs().max(q.0.abs());
        if big > BIG {
            p = (p.0 * SMALL, p.1 * SMALL);
            q = (q.0 * SMALL, q.1 * SMALL);
            e += 500;
        } else if big < SMALL && big != 0.0 {
            p = (p.0 * BIG, p.1 * BIG);
            q = (q.0 * BIG, q.1 * BIG);
            e -= 500;
        }
        if dh > BIG {
            dh *= SMALL;
            dl *= SMALL;
            e -= 500;
        }
    }
    // (p + pl)/(dh + dl), rounded once.
    let q0 = p.0 / dh;
    let r = res(q0, dh, p.0) + p.1 - q0 * dl;
    scalbn(q0 + r / dh, e)
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
    // He_(k+1) = x He_k − k He_(k−1).
    recur_ex(n, x, |k| (x, k as f64))
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
    // (k + 1) L_(k+1) = (2k + 1 + α − x) L_k − (k + α) L_(k−1), L₁ = 1 + α − x.
    let l1 = sum(1.0, alpha).add_f64(-x);
    if on_grid(alpha) {
        // k + α exact: the coefficients by two_sum and two_prod of exact values.
        return recur_q(n, dd(1.0), l1, 0, |k| {
            let kf = k as f64;
            (
                sum((2.0 * kf + 1.0) + alpha, -x),
                prod(kf + alpha, kf),
                kf + 1.0,
            )
        });
    }
    recur_q(n, dd(1.0), l1, 0, |k| {
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

/// Whether `T_n(x)` and `U_n(x)` overflow for `|x| > 1`: `n log₂(|x| + √(x² − 1)) > 1100`.
#[inline]
fn cheb_overflows(n: u32, x: f64) -> bool {
    let a = x.abs();
    a > 1.0 && n as f64 * (a + ((a - 1.0) * (a + 1.0)).sqrt()).log2() > 1100.0
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
    let ab = alpha + beta;
    if x.is_infinite() {
        // The leading coefficient (n + α + β + 1)_n / (2^n n!).
        let s = product_sign(n as f64 + ab, 1, n);
        return if s == 0.0 {
            f64::NAN
        } else {
            at_infinity(n, s, x)
        };
    }
    // P₁ = (α + 1) + (α + β + 2)(x − 1)/2.
    let abd = sum(alpha, beta);
    let xm1 = sum(x, -1.0);
    let p1 = sum(alpha, 1.0).add(abd.add_f64(2.0).mul(xm1).mul_f64(0.5));
    // A recurrence coefficient c_k = 2k(k + α + β)(2k + α + β − 2) of k ≥ 2 that vanishes needs
    // the explicit sum.
    let int_in = |v: f64| v == v.round() && v >= 2.0 && v <= n as f64;
    if n >= 2 && (int_in(-ab) || int_in((2.0 - ab) * 0.5)) {
        return jacobi_sum(n, alpha, beta, x);
    }
    if on_grid(alpha) && on_grid(beta) {
        let (h, l) = two_prod(alpha - beta, ab);
        let a2b2 = Dd { hi: h, lo: l };
        return recur(n, dd(1.0), p1, 0, |k| {
            let k1 = (k + 1) as f64;
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
    //                            − 2(k+α−1)(k+β−1)(2k+α+β) P_(k−2), here with k → k + 1.
    recur(n, dd(1.0), p1, 0, |k| {
        let k1 = (k + 1) as f64;
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

/// `P_n^(α,β)(x) = Σ_j (n+α+β+1)_j (α+j+1)_(n−j) / (j! (n−j)!) ((x − 1)/2)^j`, for the parameters
/// at which the recurrence divides by zero.
fn jacobi_sum(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    let y = (x - 1.0) * 0.5;
    let mut total = 0.0;
    let mut yj = 1.0;
    for j in 0..=n {
        let mut c = 1.0;
        for i in 0..j {
            c *= (n as f64 + alpha + beta + 1.0 + i as f64) / (i + 1) as f64;
        }
        for i in (j + 1)..=n {
            c *= (alpha + i as f64) / (i - j) as f64;
        }
        total += c * yj;
        yj *= y;
    }
    total
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
    // (k + 1) C_(k+1) = 2(k + α) x C_k − (k + 2α − 1) C_(k−1), C₁ = 2αx.
    let c1 = prod(2.0 * alpha, x);
    if on_grid(alpha) {
        return recur_q(n, dd(1.0), c1, 0, |k| {
            let kf = k as f64;
            (
                prod(2.0 * (kf + alpha), x),
                prod(2.0 * alpha + kf - 1.0, kf),
                kf + 1.0,
            )
        });
    }
    recur_q(n, dd(1.0), c1, 0, |k| {
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
