//! The Riemann zeta function of a complex argument `s = σ + it`.
//!
//! * `σ ≥ 1/2`: the Euler–Maclaurin formula `ζ(s) = Σ_{n<N} n^(−s) + N^(1−s)/(s − 1)
//!   + N^(−s)/2 + Σ_{k≤24} B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(−s−2k+1)`, `N = ⌈(|s| + 40)/2.83⌉`, so that the
//!   ratio of the terms of the remainder stays below `0.45²`. Each `n^(−s)` has its phase
//!   `t ln n / 2π` in double-double arithmetic, reduced exactly, and its modulus `e^(−σ ln n)`
//!   with `ln n` a double-double; the terms are added from the smallest, in double-double.
//!   The same for `σ < 1/2` where `|t|` is large enough that the terms do not cancel by much
//!   (`|t| ≥ 5 + 30 max(0, −σ)` for `σ ≥ −3`, and every `|t| > 200`; past that the error grows
//!   as `2.2^(−σ)`).
//! * Elsewhere (`σ < 1/2`, `|t|` small) the reflection
//!   `ζ(s) = 2 (2π)^(s−1) sin(πs/2) Γ(1−s) ζ(1−s)`, with the phase of `(2π)^(s−1)` and the
//!   hyperbolic functions of `πt/2` from double-double arguments.
//! * `t = 0`: the real function; `t < 0`: `ζ(s̄) = conj ζ(s)`.
//!
//! The cost grows as `|t|` (Riemann–Siegel's formula would be `√|t|`).
//!
//! The tables are printed by `gen_util.py (its Bernoulli numbers live in hurwitz.rs)` in this directory (with `gen_fit.py`).

use super::hurwitz::BERNOULLI_2K as BERNOULLI;
use super::real::zeta_f64;
use super::util::{PI_HI, PI_LO, exp_mul, ln_dd, sincos_pi, two_prod, two_sum};
use crate::math::complex::Complex;
use crate::math::special::gamma::gamma_c64;

/// `1/2π` and `ln(2π)/2π` as double-doubles.
const INV_2PI_HI: f64 = 0.159_154_943_091_895_35;
const INV_2PI_LO: f64 = -9.839_338_337_591_243e-18;
const LN_2PI_HI: f64 = 1.837_877_066_409_345_6;
const LN_2PI_LO: f64 = -7.756_588_316_134_483e-17;
const LN_2PI_OVER_2PI_HI: f64 = 0.292_507_219_914_278_9;
const LN_2PI_OVER_2PI_LO: f64 = 9.893_802_182_295_06e-18;

type C = (f64, f64);

/// Whether the Euler–Maclaurin formula is the better one at `s` (`t > 0`): where its terms do not
/// cancel by much (measured: within a few ε for `σ ≥ −3` past these `t`), and past `t = 200`,
/// where the reflection's `Γ(1 − s)` and `sin(πs/2)` would leave the range.
#[inline]
fn use_em(sigma: f64, t: f64) -> bool {
    sigma >= 0.5 || t > 200.0 || (sigma >= -3.0 && t >= 5.0 - 30.0 * sigma.min(0.0))
}

#[inline(always)]
fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline(always)]
fn div(a: C, b: C) -> C {
    let z = Complex::new(a.0, a.1) / Complex::new(b.0, b.1);
    (z.re, z.im)
}

/// `(cos 2πφ, sin 2πφ)` of the turns `φ = ph + pl` (a double-double), reduced exactly.
#[inline(always)]
fn turn(ph: f64, pl: f64) -> C {
    let f = ph - ph.round();
    let (fh, fl) = two_sum(f, pl);
    let (s, c) = sincos_pi(2.0 * fh);
    let d = 2.0 * PI_HI * fl;
    (c - d * s, s + d * c)
}

/// `e^(a + al) · e^(2πi (ph + pl))`: a modulus from a double-double exponent and a phase in
/// turns.
#[inline(always)]
fn polar(a: f64, al: f64, ph: f64, pl: f64) -> C {
    let m = exp_mul(a, 1.0, al);
    let (c, s) = turn(ph, pl);
    (m * c, m * s)
}

/// `n^(−s)` for an integer `n ≥ 1`, `s = σ + it`.
#[inline(always)]
fn pow_neg(n: f64, sigma: f64, t: f64) -> C {
    let (l, ll) = ln_dd(n);
    // −σ ln n
    let (a, ae) = two_prod(-sigma, l);
    let al = ae - sigma * ll;
    // −t ln n / 2π, in turns
    let (u, ue) = two_prod(l, INV_2PI_HI);
    let ul = ue + (l * INV_2PI_LO + ll * INV_2PI_HI);
    let (p, pe) = two_prod(-t, u);
    let pl = pe - t * ul;
    polar(a, al, p, pl)
}

/// `ζ(s)` for `σ ≥ 1/2`, `t > 0`, by the Euler–Maclaurin formula.
fn zeta_em(sigma: f64, t: f64) -> C {
    let abs_s = sigma.hypot(t);
    let big_n = ((abs_s + 40.0) / 2.83).ceil();
    // The remainder at N: P = N^(−s), N P/(s − 1) + P/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ N^(1−2k) P.
    let p = pow_neg(big_n, sigma, t);
    let s = (sigma, t);
    let r = 1.0 / big_n;
    let r2 = r * r;
    let mut fac = (sigma * r, t * r); // (s)₁ N⁻¹
    let mut acc = (0.0, 0.0);
    let mut k2 = 1.0;
    for &b in BERNOULLI.iter() {
        let term = (b * fac.0, b * fac.1);
        acc.0 += term.0;
        acc.1 += term.1;
        if term.0.abs() + term.1.abs() <= 1e-19 * (acc.0.abs() + acc.1.abs() + 1.0) {
            break;
        }
        // × (s + 2k − 1)(s + 2k) / N²
        let f = mul((sigma + k2, t), (sigma + k2 + 1.0, t));
        fac = mul(fac, (f.0 * r2, f.1 * r2));
        k2 += 2.0;
    }
    let q = div((big_n, 0.0), (s.0 - 1.0, s.1));
    let bracket = (q.0 + 0.5 + acc.0, q.1 + acc.1);
    let tail = mul(p, bracket);
    // The direct terms from the smallest, as double-doubles.
    let (mut rh, mut rl) = (tail.0, 0.0);
    let (mut ih, mut il) = (tail.1, 0.0);
    let mut n = big_n - 1.0;
    while n >= 2.0 {
        let v = pow_neg(n, sigma, t);
        let (a, e) = two_sum(rh, v.0);
        rh = a;
        rl += e;
        let (a, e) = two_sum(ih, v.1);
        ih = a;
        il += e;
        n -= 1.0;
    }
    let (a, e) = two_sum(1.0, rh);
    (a + (e + rl), ih + il)
}

/// `(cosh a, sinh a)` of `a = πt/2`, the argument a double-double for `|a| ≥ 1`.
#[inline]
fn cosh_sinh_half_pi(t: f64) -> C {
    let (a, ae) = two_prod(0.5 * PI_HI, t);
    let al = ae + 0.5 * PI_LO * t;
    if a.abs() < 1.0 {
        let a = a + al;
        return (a.cosh(), a.sinh());
    }
    let ep = exp_mul(a, 1.0, al);
    let em = exp_mul(-a, 1.0, -al);
    (0.5 * (ep + em), 0.5 * (ep - em))
}

/// `ψ(w)` to a few digits for `Re w ≥ 1`: the recurrence to `|w| ≥ 6`, then three terms of
/// the asymptotic series.
fn digamma_rough_c(w: C) -> C {
    let (mut w, mut acc) = (w, (0.0, 0.0));
    while w.0 * w.0 + w.1 * w.1 < 36.0 {
        let r = div((1.0, 0.0), w);
        acc = (acc.0 - r.0, acc.1 - r.1);
        w = (w.0 + 1.0, w.1);
    }
    let lw = (0.5 * (w.0 * w.0 + w.1 * w.1).ln(), w.1.atan2(w.0));
    let r = div((1.0, 0.0), w);
    let r2 = mul(r, r);
    (
        acc.0 + lw.0 - 0.5 * r.0 - r2.0 / 12.0,
        acc.1 + lw.1 - 0.5 * r.1 - r2.1 / 12.0,
    )
}

/// `ζ(s)` for `σ < 1/2`, `t > 0`, by the reflection formula.
fn zeta_reflect(sigma: f64, t: f64) -> C {
    // 2 (2π)^(s−1): modulus e^((σ − 1) ln 2π) (doubled), phase t ln 2π / 2π turns.
    let (sm1, sm1l) = two_sum(sigma, -1.0);
    let (a, ae) = two_prod(sm1, LN_2PI_HI);
    let al = ae + sm1 * LN_2PI_LO + sm1l * LN_2PI_HI;
    let (ph, pe) = two_prod(t, LN_2PI_OVER_2PI_HI);
    let pl = pe + t * LN_2PI_OVER_2PI_LO;
    let w = polar(a, al, ph, pl);
    let w = (2.0 * w.0, 2.0 * w.1);
    // sin(πs/2) = sin(πσ/2) cosh(πt/2) + i cos(πσ/2) sinh(πt/2)
    let (sn, cs) = sincos_pi(0.5 * sigma);
    let (ch, sh) = cosh_sinh_half_pi(t);
    let sin_s = (sn * ch, cs * sh);
    // 1 − σ rounds where it crosses a power of 2 (at σ = −15.x, by 2⁻⁴⁹, which ψ(1 − s) ~ 3.5
    // makes 30 ε of Γ): Γ(1 − s) (1 + zl ψ(1 − s)) with ψ to a few digits.
    let (zr, zl) = two_sum(1.0, -sigma);
    let g = gamma_c64(Complex::new(zr, -t));
    let g = if zl == 0.0 {
        g
    } else {
        let psi = digamma_rough_c((zr, -t));
        let f = (1.0 + zl * psi.0, zl * psi.1);
        let v = mul((g.re, g.im), f);
        Complex::new(v.0, v.1)
    };
    let z1 = zeta_em(zr, -t);
    mul(mul(w, sin_s), mul((g.re, g.im), z1))
}

/// `ζ(s)` of a `Complex<f64>`.
pub(crate) fn zeta_c64(s: Complex<f64>) -> Complex<f64> {
    let (sigma, t) = (s.re, s.im);
    if sigma.is_nan() || t.is_nan() {
        return Complex::new(f64::NAN, f64::NAN);
    }
    if t == 0.0 {
        return Complex::new(zeta_f64(sigma), 0.0);
    }
    let (ta, flip) = if t < 0.0 { (-t, true) } else { (t, false) };
    let z = if use_em(sigma, ta) {
        zeta_em(sigma, ta)
    } else {
        zeta_reflect(sigma, ta)
    };
    Complex::new(z.0, if flip { -z.1 } else { z.1 })
}
