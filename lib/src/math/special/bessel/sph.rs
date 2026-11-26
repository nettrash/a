//! The spherical Bessel functions `jₙ`, `yₙ`, `iₙ`, `kₙ` of an integer order and an `f64`:
//! their power series for small `x`, the terminating Hankel sums `jₙ = (P sin(x − nπ/2) +
//! Q cos(x − nπ/2))/x` for large `x` (the sine and cosine from the double-double reduction),
//! the recurrences in their stable directions, and `kₙ = (π/2)·e^(−x)·Sₙ` with `Sₙ` up by the
//! recurrence of positive terms.

use super::dd::{Dd, two_prod, two_sum};
use super::general::{debye_parts, jy, k as k_general, ln_gamma_large, ratio_j};
use super::kernels::{
    add_c, coef, exp_parts, ldexp, ln_dd, mla, mul_c, pow2i, rec_pos, rec_step, recip_dd, rem_pio2,
    sin_cos_quadrant_dd,
};
use core::f64::consts::FRAC_PI_2;

const EPS: f64 = 1.0 / (1u64 << 55) as f64;
const BIG: i32 = 600;
/// Beyond this order the recurrences (of cost `O(n)`) are not run: see `general::NU_MAX`.
const HUGE_N: u32 = 10_000_000;

/// `(sin(x − nπ/2), cos(x − nπ/2))` as double-doubles.
#[inline]
fn sin_cos_shift(x: f64, n: u32) -> (Dd, Dd) {
    let (q, r) = rem_pio2(x);
    sin_cos_quadrant_dd(q.wrapping_sub(n as i64), r)
}

/// `(a·s + b·c)/x`, rounded once (`a` and `s` the leading products).
#[inline(always)]
fn combine(a: Dd, s: Dd, b: Dd, c: Dd, x: f64) -> f64 {
    let (h, l) = two_prod(a.hi, s.hi);
    let (h2, l2) = two_prod(b.hi, c.hi);
    let (v, e) = two_sum(h, h2);
    let lo = e + (l + l2) + (mla(a.hi, s.lo, a.lo * s.hi) + mla(b.hi, c.lo, b.lo * c.hi));
    let qt = v / x;
    let (p, pe) = two_prod(qt, x);
    qt + (((v - p) - pe) + lo) / x
}

/// The terminating Hankel sums `(P, Q)` of the order `n + 1/2` at `x`.
fn hankel_pq(n: u32, x: f64) -> (Dd, Dd) {
    let xi = 1.0 / x;
    let mut p = Dd::new(1.0);
    let mut q = Dd::new(0.0);
    let mut term = 1.0f64;
    let nf = n as f64;
    for k in 1..=n {
        let kf = k as f64;
        // a_k/x^k = a_(k−1)/x^(k−1)·(n + k)(n − k + 1)/(2k·x).
        term *= (nf + kf) * (nf - kf + 1.0) * xi / (2.0 * kf);
        let s = if (k / 2) % 2 == 0 { term } else { -term };
        if k % 2 == 0 {
            p = p.add_f64(s);
        } else {
            q = q.add_f64(s);
        }
    }
    (p, q)
}

/// `x^n/(2n + 1)!!` as `m·2^e`.
fn series_prefactor(n: u32, x: f64) -> (Dd, i32) {
    if n <= 40 {
        // Π x/(2k + 1), compensated (the running product with its error to first order, each
        // factor a double-double from one division and its remainder).
        let (mut m, mut em) = (1.0f64, 0.0f64);
        let mut e = 0i32;
        for k in 1..=n {
            let (ch, cl) = quot(x, 0.0, (2 * k + 1) as f64);
            (m, em) = mul_c(m, em, ch, cl);
            if m.abs() < pow2i(-BIG) {
                m *= pow2i(BIG);
                em *= pow2i(BIG);
                e -= BIG;
            }
        }
        let (hi, lo) = two_sum(m, em);
        return (Dd { hi, lo }, e);
    }
    // (2n + 1)!! = 2^(n+1)·Γ(n + 3/2)/√π.
    let nf = n as f64;
    const LN_SQRT_PI: Dd = Dd {
        hi: 0.572_364_942_924_700_1,
        lo: 5.132_975_581_353_913e-18,
    };
    let a = ln_dd(x)
        .mul_f64(nf)
        .sub(ln_gamma_large(nf + 1.5))
        .add(LN_SQRT_PI)
        .sub(
            Dd {
                hi: core::f64::consts::LN_2,
                lo: 2.319_046_813_846_299_6e-17,
            }
            .mul_f64(nf + 1.0),
        );
    exp_parts(a.hi, a.lo)
}

/// `Σ (∓x²/2)^k/(k!·(2n + 3)(2n + 5)…(2n + 2k + 1))` (`−` for `j`), compensated: each term and
/// the sum with their errors to first order, the ratio of two terms a double-double from one
/// division and its remainder, off the chain (the double-double run's accuracy).
fn series_sum(n: u32, x: f64, minus: bool) -> Dd {
    let (qh, ql) = super::dd::two_prod(x, x);
    let (qh, ql) = if minus {
        (-0.5 * qh, -0.5 * ql)
    } else {
        (0.5 * qh, 0.5 * ql)
    };
    let (mut t, mut et) = (1.0f64, 0.0f64);
    let (mut s, mut es) = (1.0f64, 0.0f64);
    let mut k = 1.0f64;
    let b = (2 * n + 1) as f64;
    loop {
        // k·(b + 2k) is exact below 2⁵³.
        let (ch, cl) = quot(qh, ql, k * (b + 2.0 * k));
        (t, et) = mul_c(t, et, ch, cl);
        (s, es) = add_c(s, es, t, et);
        if t.abs() <= EPS * 0.25 * s.abs() || k > 400.0 {
            break;
        }
        k += 1.0;
    }
    let (hi, lo) = two_sum(s, es);
    Dd { hi, lo }
}

/// `(h + l)/d` of a double-double and an exact `d`, as a double-double: one division and its
/// remainder.
#[inline(always)]
fn quot(h: f64, l: f64, d: f64) -> (f64, f64) {
    let q = h / d;
    let (p, pe) = two_prod(q, d);
    (q, (((h - p) - pe) + l) / d)
}

fn series(n: u32, x: f64, minus: bool) -> f64 {
    let (m, e) = series_prefactor(n, x);
    ldexp(m.mul(series_sum(n, x, minus)).value(), e)
}

/// `jₙ(x)`.
pub(crate) fn sph_j(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        let v = sph_j(n, -x);
        return if n % 2 == 1 { -v } else { v };
    }
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return 0.0;
    }
    let nf = n as f64;
    if x * x <= 0.5 * (2.0 * nf + 3.0) {
        return series(n, x, true);
    }
    if n > HUGE_N {
        return jy(nf + 0.5, x, true, false).0 * (FRAC_PI_2 / x).sqrt();
    }
    if n == 0 {
        let (s, _) = sin_cos_shift(x, 0);
        return s.div(Dd::new(x)).value();
    }
    if x >= (0.5 * nf * (nf + 1.0)).max(1.5) {
        let (p, q) = hankel_pq(n, x);
        let (s, c) = sin_cos_shift(x, n);
        return combine(p, s, q, c, x);
    }
    if x > nf {
        // Up from j₀ = sin x/x and j₁ = (j₀ − cos x)/x, in double-double from the start (an `f64`
        // run, or rounded starts, gathered up to 3.3 ulps of M by n = 12).
        // The run is compensated: each value with its error to first order, the coefficients
        // from 1/x as a double-double, as accurate as the double-double run it replaces.
        let (s, c) = sin_cos_shift(x, 0);
        let xi = recip_dd(x);
        let j0 = s.mul(xi);
        let j1 = j0.sub(c).mul(xi);
        let (mut a, mut ea, mut b, mut eb) = (j0.hi, j0.lo, j1.hi, j1.lo);
        for m in 1..n {
            let (ch, cl) = coef((2 * m + 1) as f64, xi);
            let (t, et) = rec_step(ch, cl, b, eb, -a, -ea);
            (a, ea) = (b, eb);
            (b, eb) = (t, et);
        }
        return b + eb;
    }
    // x ≤ n: the ratio jₙ/jₙ₋₁ = J_(n+1/2)/J_(n−1/2) by its continued fraction, then down to j₀
    // or j₁ to normalise. The run down passes the ratio's relative error to jₙ whole, so the
    // fraction is summed from its tail in double-double at every x (near x = n an `f64` sum loses
    // a few ulps, and below x = n/2 Lentz's forward `f64` value is 5 to 11 ulps off).
    let r = ratio_j(nf - 0.5, x, true);
    // jₙ = r·jₙ₋₁; run down with fₙ = 1·2^e, compensated (values with their errors to first
    // order, the coefficients from 1/x as a double-double).
    let xd = Dd::new(x);
    let xi = recip_dd(x);
    let f1 = Dd::new(1.0).div(r);
    let (mut h, mut eh, mut l, mut el) = (1.0, 0.0, f1.hi, f1.lo); // f_n, f_(n−1)
    let mut e = 0i32;
    let mut k = n - 1;
    while k >= 1 {
        // f_(k−1) = (2k + 1)/x·f_k − f_(k+1).
        let (ch, cl) = coef((2 * k + 1) as f64, xi);
        let (t, et) = rec_step(ch, cl, l, el, -h, -eh);
        (h, eh) = (l, el);
        (l, el) = (t, et);
        if l.abs() > pow2i(BIG) {
            let s = pow2i(-BIG);
            (h, eh, l, el) = (h * s, eh * s, l * s, el * s);
            e += BIG;
        }
        k -= 1;
    }
    let (hh, hl) = super::dd::fast_two_sum(h, eh);
    let (lh, ll) = super::dd::fast_two_sum(l, el);
    let (hi, lo) = (Dd { hi: hh, lo: hl }, Dd { hi: lh, lo: ll });
    // lo = f₀, hi = f₁ (scaled by 2^−e relative to fₙ = 1). Normalise by the larger of them (a
    // small one carries the recurrence's absolute error), with j₀ and j₁ in double-double.
    let (s, c) = sin_cos_shift(x, 0);
    let j0 = s.div(xd);
    let scale = if lo.hi.abs() >= hi.hi.abs() {
        j0.div(lo)
    } else {
        j0.sub(c).div(xd).div(hi)
    };
    ldexp(scale.value(), -e)
}

/// `yₙ(x)` (`−∞` at 0; `yₙ(−x) = (−1)^(n+1) yₙ(x)`).
pub(crate) fn sph_y(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        let v = sph_y(n, -x);
        return if n.is_multiple_of(2) { -v } else { v };
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let nf = n as f64;
    if n > HUGE_N {
        return jy(nf + 0.5, x, false, true).1 * (FRAC_PI_2 / x).sqrt();
    }
    if x < TINY_X {
        return match tiny_lead(n, x) {
            Some((m, e)) => -ldexp(m.value(), e),
            None => f64::NEG_INFINITY,
        };
    }
    let (s, c) = sin_cos_shift(x, 0);
    if n == 0 {
        return c.div(Dd::new(x)).neg().value();
    }
    if x >= (0.5 * nf * (nf + 1.0)).max(1.0) {
        let (p, q) = hankel_pq(n, x);
        let (s, c) = sin_cos_shift(x, n);
        return combine(q, s, p.neg(), c, x);
    }
    // Up from y₀ = −cos x/x, y₁ = −(cos x/x + sin x)/x.
    // Up from y₀, y₁ in double-double (an `f64` run gathered 2 ulps of M by n = 4).
    // Compensated: each value with its error to first order, the coefficients from 1/x as a
    // double-double.
    let xi = recip_dd(x);
    let y0 = c.mul(xi).neg();
    let y1 = c.mul(xi).add(s).mul(xi).neg();
    let (mut a, mut ea, mut b, mut eb) = (y0.hi, y0.lo, y1.hi, y1.lo);
    let mut e = 0i32;
    for m in 1..n {
        let (ch, cl) = coef((2 * m + 1) as f64, xi);
        // Rescale before the step that would overflow (x ≥ 2⁻⁵⁰⁰, so (2m + 1)/x is finite).
        while b.abs() * ch > pow2i(900) {
            let s = pow2i(-BIG);
            (a, ea, b, eb) = (a * s, ea * s, b * s, eb * s);
            e += BIG;
        }
        let (t, et) = rec_step(ch, cl, b, eb, -a, -ea);
        (a, ea) = (b, eb);
        (b, eb) = (t, et);
    }
    ldexp(b + eb, e)
}

/// Below this `x` the spherical `yₙ` and `kₙ` are their leading terms (`1/x`, `1/x²` and so on
/// would overflow on the way).
const TINY_X: f64 = 3.054_936_363_499_605e-151; // 2⁻⁵⁰⁰

/// `x = f·2^e`, `f` in `[1, 2)`, for a positive finite `x` (the subnormals too).
fn frexp(x: f64) -> (f64, i32) {
    let (x, e0) = if x < f64::MIN_POSITIVE {
        (x * pow2i(64), -64)
    } else {
        (x, 0)
    };
    let b = x.to_bits();
    let e = ((b >> 52) as i32) - 1023 + e0;
    (
        f64::from_bits((b & ((1u64 << 52) - 1)) | (1023u64 << 52)),
        e,
    )
}

/// `(2n − 1)!!/x^(n+1)` as `m·2^e` for `x < 2⁻⁵⁰⁰`, the leading term of `−yₙ(x)` and of
/// `(2/π)·kₙ(x)` (the next is `x²/2` of it), for `n ≤ 1`; `None` for `n ≥ 2`, where it is beyond
/// `3·2¹⁵⁰⁰`.
fn tiny_lead(n: u32, x: f64) -> Option<(Dd, i32)> {
    if n >= 2 {
        return None;
    }
    let (f, e) = frexp(x);
    Some(if n == 0 {
        (Dd::new(1.0).div(Dd::new(f)), -e)
    } else {
        let (p, pe) = two_prod(f, f);
        (Dd::new(1.0).div(Dd { hi: p, lo: pe }), -2 * e)
    })
}

/// `iₙ(x)` (`iₙ(−x) = (−1)ⁿ iₙ(x)`).
pub(crate) fn sph_i(n: u32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        let v = sph_i(n, -x);
        return if n % 2 == 1 { -v } else { v };
    }
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let by_series = if n < I_DEBYE_N {
        // Up to n²/28 (above 20 + n/2 only from n = 34 on, beyond I_DEBYE_N) the closed form below
        // cancels by more than 2⁴⁰; the series takes 70 terms at most there.
        x <= 20.0 + 0.5 * nf || x * x <= 2.0 * (2.0 * nf + 3.0) || 28.0 * x <= nf * nf
    } else {
        // Its terms grow first by (x²/2)/(2n + 3): no more than 4 times, so that it stays short.
        x * x <= 16.0 * (nf + 1.5) || x < 30.0
    };
    if by_series {
        return series(n, x, false);
    }
    if x >= 0.5 * nf * (nf + 1.0) {
        // (e^x·Σ(−1)^k aₖ/xᵏ − (−1)ⁿ e^(−x)·Σ aₖ/xᵏ)/(2x); the second part is below 2⁻⁵⁸ of the
        // first here (x > 20). Beyond x = 720 it overflows (iₙ(x) > e^(x − 1)/(2x) from
        // n(n + 1)/2 on), which is known before the O(n) sum (orders from 38 on start there).
        if x > 720.0 {
            return f64::INFINITY;
        }
        let xi = 1.0 / x;
        let mut s = Dd::new(1.0);
        let mut term = 1.0f64;
        for k in 1..=n {
            let kf = k as f64;
            term *= -(nf + kf) * (nf - kf + 1.0) * xi / (2.0 * kf);
            s = s.add_f64(term);
        }
        let (m, k) = exp_parts(x, 0.0);
        return ldexp(m.mul(s).div(Dd::new(2.0 * x)).value(), k);
    }
    if n < I_DEBYE_N {
        return i_closed(n, x);
    }
    // √(π/(2x))·I_(n+1/2)(x) by Debye's expansion, the scale apart (I alone may overflow or
    // underflow where iₙ does not).
    let (v, k) = debye_parts(nf + 0.5, x, true, false);
    if !v.hi.is_finite() || v.hi == 0.0 {
        // Beyond the range by far, as Debye's exponent decided (its ∞ times a double-double
        // would be NaN).
        return v.hi;
    }
    let a = PIO2_DD.div(Dd::new(x)).sqrt();
    ldexp(v.mul(a).value(), k)
}

/// From this order on `iₙ` comes from Debye's expansion of `I_(n+1/2)` between the series and the
/// closed form (there `x ≥ 30`, `p = ν/√(ν² + x²) ≤ 0.57`, where its first omitted term,
/// `U₁₃(p)/ν¹³`, is far below the rounding).
// Measured on an Apple M4 Max (each order from 12 to 100, x over 1–8000, over the band between
// the series and n(n + 1)/2 and over 1–200): Debye's expansion, 70–100 ns at every order, passes
// the series and the closed form at n = 20 (0.98–1.0, 1.02–1.06 at 18), and takes 0.64–0.83 of
// their time over 20–49; within 0.70 ulp there (120,000 cases: mpmath on the 469 whose result
// changed and 9,000 more).
const I_DEBYE_N: u32 = 20;

/// `π/2` as a double-double.
const PIO2_DD: Dd = Dd {
    hi: FRAC_PI_2,
    lo: 6.123_233_995_736_766e-17,
};

/// `iₙ(x) = (e^x·Σ(−1)ᵏ aₖ/xᵏ − (−1)ⁿ e^(−x)·Σ aₖ/xᵏ)/(2x)`, `aₖ = (n + k)!/(k!(n − k)!·2ᵏ)`, both
/// sums and their terms in double-double, for `n < I_DEBYE_N` from the end of the series on, where
/// the terms cancel by `2⁴⁰` at most (the error of a term, about `2⁻⁹⁸` of it, stays below `2⁻⁵⁸`
/// of the value); both exponentials are kept.
fn i_closed(n: u32, x: f64) -> f64 {
    let nf = n as f64;
    // Compensated: the terms and both sums with their errors to first order (the double-double
    // run's accuracy); 1/x a double-double once, each ratio (n + k)(n − k + 1)/(2k)·(1/x) formed
    // off the chain.
    let xi = recip_dd(x);
    let (mut t, mut et) = (1.0f64, 0.0f64);
    let (mut a, mut ea) = (1.0f64, 0.0f64);
    let (mut p, mut ep) = (1.0f64, 0.0f64);
    for k in 1..=n {
        let kf = k as f64;
        // (n + k)(n − k + 1) is an exact integer, so is 2k: their quotient a double-double, then
        // times 1/x.
        let (qh, ql) = quot((nf + kf) * (nf - kf + 1.0), 0.0, 2.0 * kf);
        let (ch, ce) = two_prod(qh, xi.hi);
        let cl = ce + mla(qh, xi.lo, ql * xi.hi);
        (t, et) = mul_c(t, et, ch, cl);
        (p, ep) = add_c(p, ep, t, et);
        (a, ea) = if !k.is_multiple_of(2) {
            add_c(a, ea, -t, -et)
        } else {
            add_c(a, ea, t, et)
        };
    }
    let (ph, pl) = two_sum(p, ep);
    let (ah, al) = two_sum(a, ea);
    let pos = Dd { hi: ph, lo: pl };
    let alt = Dd { hi: ah, lo: al };
    let (mp, kp) = exp_parts(x, 0.0);
    let (mm, km) = exp_parts(-x, 0.0);
    // e^(−x)·Σ aₖ/xᵏ relative to the scale 2^kp of e^x.
    let t = mm.mul(pos);
    let t = Dd {
        hi: ldexp(t.hi, km - kp),
        lo: ldexp(t.lo, km - kp),
    };
    let a = mp.mul(alt);
    let s = if n.is_multiple_of(2) {
        a.sub(t)
    } else {
        a.add(t)
    };
    ldexp(s.div(Dd::new(2.0 * x)).value(), kp)
}

/// `kₙ(x) = √(π/(2x))·K_(n+1/2)(x)` (`∞` at 0, NaN for `x < 0`).
pub(crate) fn sph_k(n: u32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x > 1.0e6 && (n as f64) < 1.4 * x {
        // kₙ(x) ≤ (π/2)·e^(−x + n²/(2x))/x underflows.
        return 0.0;
    }
    if n > HUGE_N {
        return k_general(n as f64 + 0.5, x, false) * (FRAC_PI_2 / x).sqrt();
    }
    if x < TINY_X {
        return match tiny_lead(n, x) {
            Some((m, e)) => ldexp(m.mul(PIO2_DD).value(), e),
            None => f64::INFINITY,
        };
    }
    // (Horner's rule on the polynomial Σ aₖ/(2x)ᵏ, which took n ≤ 10 before, gives the same bits
    // and is slower at every order: measured on an Apple M4 Max, the recurrence takes 0.98 of its
    // time at n = 0, 0.45 over n = 0–10 and 0.21 at n = 20.)
    // Sₙ: S₀ = 1/x, S₁ = (1 + 1/x)/x, S_(m+1) = S_(m−1) + (2m + 1)/x·S_m: positive terms, run
    // compensated (each value with its error to first order, the coefficients from 1/x as a
    // double-double), which keeps the double-double run's accuracy at the cost of an `f64` one.
    let xi = recip_dd(x);
    let (m, e) = if n == 0 {
        (xi, 0)
    } else {
        let s1 = xi.add_f64(1.0).mul(xi);
        let (mut a, mut ea, mut b, mut eb) = (xi.hi, xi.lo, s1.hi, s1.lo);
        let mut e = 0i32;
        for m in 1..n {
            let (ch, cl) = coef((2 * m + 1) as f64, xi);
            // Rescale before the step that would overflow (x ≥ 2⁻⁵⁰⁰).
            while b * ch > pow2i(900) {
                let s = pow2i(-BIG);
                (a, ea, b, eb) = (a * s, ea * s, b * s, eb * s);
                e += BIG;
            }
            let (t, et) = rec_pos(ch, cl, b, eb, a, ea);
            (a, ea) = (b, eb);
            (b, eb) = (t, et);
        }
        let (hi, lo) = super::dd::fast_two_sum(b, eb);
        (Dd { hi, lo }, e)
    };
    let (em, ek) = exp_parts(-x.min(1.0e6), 0.0);
    let v = m.mul(em).mul(Dd {
        hi: FRAC_PI_2,
        lo: 6.123_233_995_736_766e-17,
    });
    ldexp(v.value(), e + ek)
}
