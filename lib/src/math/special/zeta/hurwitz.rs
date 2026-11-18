//! The Hurwitz zeta function of real arguments, by the Euler–Maclaurin formula
//! `ζ(s, a) = Σ_{n<N} (n + a)^(−s) + A^(1−s)/(s − 1) + A^(−s)/2
//! + Σ_{k≤10} B₂ₖ/(2k)! (s)₂ₖ₋₁ A^(−s−2k+1)`, `A = a + N`, with `N` the fewest terms that bring
//! `A` to `10 + 1.2 s` (where the remainder is below `2⁻⁵⁶`) or the series below `2⁻⁵⁶` of
//! its first term. Every term is positive; they are added from the smallest, each
//! `(n + a)^(−s)` of the exact `n + a` (a double-double, its low part to first order).
//!
//! The tables are printed by `gen_util.py` in this directory (with `gen_fit.py`).

use super::real::zeta_f64;
use super::util::{pow_mul, residual, two_sum};

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

/// `(n + a)^(−s)` of the exact sum.
#[inline(always)]
fn term(n: f64, a: f64, s: f64) -> f64 {
    let (h, l) = two_sum(n, a);
    let p = h.powf(-s);
    if l == 0.0 { p } else { p * (1.0 - s * (l / h)) }
}

/// The Euler–Maclaurin tail at `A`: `A^(1−s)/(s − 1) + A^(−s)/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ A^(−s−2k+1)`,
/// as a double-double.
#[inline]
fn tail(big_a: f64, s: f64) -> (f64, f64) {
    let r = 1.0 / big_a;
    let r2 = r * r;
    // (s)₂ₖ₋₁ A^(1−2k), k = 1, 2, …
    let mut fac = s * r;
    let mut sum = 0.0;
    let mut k2 = 1.0;
    for &b in BERNOULLI_2K.iter() {
        let t = b * fac;
        sum += t;
        if t.abs() <= 1e-18 * (big_a / (s - 1.0)) {
            break;
        }
        fac *= (s + k2) * (s + k2 + 1.0) * r2;
        k2 += 2.0;
    }
    // A^(−s) (q + 1/2 + sum), q = A/(s − 1) as a double-double, rounded once.
    let sm1 = s - 1.0;
    let q = big_a / sm1;
    let ql = residual(q, sm1, big_a) / sm1;
    let (h, hl) = two_sum(q, 0.5 + sum);
    let t = pow_mul(big_a, -s, h, hl + ql);
    // Its rounding error, to carry on.
    (t, 0.0)
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
    // The direct terms: to A = a + N ≥ 7 + s/2, or while (n + a)^(−s) ≥ 2⁻⁵⁶ a^(−s).
    let need = 7.0 + 0.5 * s;
    let mut n = if a >= need { 0.0 } else { (need - a).ceil() };
    let mut with_tail = true;
    if n > 0.0 {
        // (1 + n/a)^(−s) < 2⁻⁵⁶ once n ≥ a (2^(56/s) − 1).
        let m = (a * ((56.0 / s).exp2() - 1.0)).ceil() + 1.0;
        if m < n {
            n = m;
            with_tail = false;
        }
    }
    // From the smallest term up, in double-double; the first term a^(−s) rounded once with the
    // rest.
    let (mut hi, mut lo) = if with_tail {
        tail(a + n, s)
    } else {
        (0.0, 0.0)
    };
    while n > 1.0 {
        n -= 1.0;
        let (h, e) = two_sum(hi, term(n, a, s));
        hi = h;
        lo += e;
    }
    if n == 1.0 {
        pow_mul(a, -s, 1.0, 0.0) + (hi + lo)
    } else {
        hi + lo
    }
}
