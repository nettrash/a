//! The spherical Bessel functions `jₙ`, `yₙ`, `iₙ`, `kₙ` of an integer order and an `f64`:
//! their power series for small `x`, the terminating Hankel sums `jₙ = (P sin(x − nπ/2) +
//! Q cos(x − nπ/2))/x` for large `x` (the sine and cosine from the double-double reduction),
//! the recurrences in their stable directions, and `kₙ = (π/2)·e^(−x)·Sₙ` with `Sₙ` up by the
//! recurrence of positive terms.

use super::dd::{Dd, two_prod, two_sum};
use super::general::{debye_parts, jy, k as k_general, ln_gamma_large, ratio_j};
use super::kernels::{exp_parts, ldexp, ln_dd, mla, pow2i, rem_pio2, sin_cos_quadrant_dd};
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
        let mut m = Dd::new(1.0);
        let mut e = 0i32;
        for k in 1..=n {
            m = m.mul_f64(x).div(Dd::new((2 * k + 1) as f64));
            if m.hi.abs() < pow2i(-BIG) {
                m = m.mul_f64(pow2i(BIG));
                e -= BIG;
            }
        }
        return (m, e);
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

/// `Σ (∓x²/2)^k/(k!·(2n + 3)(2n + 5)…(2n + 2k + 1))` (`−` for `j`).
fn series_sum(n: u32, x: f64, minus: bool) -> Dd {
    let (qh, ql) = super::dd::two_prod(x, x);
    let q = Dd {
        hi: 0.5 * qh,
        lo: 0.5 * ql,
    };
    let q = if minus { q.neg() } else { q };
    let mut term = Dd::new(1.0);
    let mut sum = Dd::new(1.0);
    let mut k = 1.0f64;
    let b = (2 * n + 1) as f64;
    loop {
        // k·(b + 2k) is exact below 2⁵³.
        term = term.mul(q).div(Dd::new(k * (b + 2.0 * k)));
        sum = sum.add(term);
        if term.hi.abs() <= EPS * 0.25 * sum.hi.abs() || k > 400.0 {
            break;
        }
        k += 1.0;
    }
    sum
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
        let (s, c) = sin_cos_shift(x, 0);
        let xd = Dd::new(x);
        let xi = Dd::new(1.0).div(xd);
        let j0 = s.mul(xi);
        let (mut a, mut b) = (j0, j0.sub(c).mul(xi));
        for m in 1..n {
            let t = b.mul(xi).mul_f64((2 * m + 1) as f64).sub(a);
            a = b;
            b = t;
        }
        return b.value();
    }
    // x ≤ n: the ratio jₙ/jₙ₋₁ = J_(n+1/2)/J_(n−1/2) by its continued fraction, then down to j₀
    // or j₁ to normalise. The run down passes the ratio's relative error to jₙ whole, so the
    // fraction is summed from its tail in double-double at every x (near x = n an `f64` sum loses
    // a few ulps, and below x = n/2 Lentz's forward `f64` value is 5 to 11 ulps off).
    let r = ratio_j(nf - 0.5, x, true);
    // jₙ = r·jₙ₋₁; run down with fₙ = 1·2^e.
    let xd = Dd::new(x);
    let xi = Dd::new(1.0).div(xd);
    let (mut hi, mut lo) = (Dd::new(1.0), Dd::new(1.0).div(r)); // f_n, f_(n−1)
    let mut e = 0i32;
    let mut k = n - 1;
    while k >= 1 {
        // f_(k−1) = (2k + 1)/x·f_k − f_(k+1).
        let t = lo.mul(xi).mul_f64((2 * k + 1) as f64).sub(hi);
        hi = lo;
        lo = t;
        if lo.hi.abs() > pow2i(BIG) {
            hi = hi.mul_f64(pow2i(-BIG));
            lo = lo.mul_f64(pow2i(-BIG));
            e += BIG;
        }
        k -= 1;
    }
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
    let xd = Dd::new(x);
    let xi = Dd::new(1.0).div(xd);
    let y0 = c.mul(xi).neg();
    let y1 = c.mul(xi).add(s).mul(xi).neg();
    let (mut a, mut b) = (y0, y1);
    let mut e = 0i32;
    for m in 1..n {
        // Rescale before the step that would overflow (x ≥ 2⁻⁵⁰⁰, so (2m + 1)/x is finite).
        let c = (2 * m + 1) as f64 / x;
        while b.hi.abs() * c > pow2i(900) {
            a = a.mul_f64(pow2i(-BIG));
            b = b.mul_f64(pow2i(-BIG));
            e += BIG;
        }
        let t = b.mul(xi).mul_f64((2 * m + 1) as f64).sub(a);
        a = b;
        b = t;
    }
    ldexp(b.value(), e)
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
        // Up to n²/28 (from n = 34 on above 20 + n/2) the closed form below cancels by more than
        // 2⁴⁰; the series takes 70 terms at most there.
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
/// closed form (see `general::DEBYE_NU`).
const I_DEBYE_N: u32 = 50;

/// `π/2` as a double-double.
const PIO2_DD: Dd = Dd {
    hi: FRAC_PI_2,
    lo: 6.123_233_995_736_766e-17,
};

/// `iₙ(x) = (e^x·Σ(−1)ᵏ aₖ/xᵏ − (−1)ⁿ e^(−x)·Σ aₖ/xᵏ)/(2x)`, `aₖ = (n + k)!/(k!(n − k)!·2ᵏ)`, both
/// sums and their terms in double-double, for `n < 50` from the end of the series on, where the
/// terms cancel by `2⁴⁰` at most (the error of a term, about `2⁻⁹⁸` of it, stays below `2⁻⁵⁸` of
/// the value); both exponentials are kept (`e^(−2x)` is `2⁻⁵⁵` of the value at `n = 49`,
/// `x = 44.5`).
fn i_closed(n: u32, x: f64) -> f64 {
    let nf = n as f64;
    let mut term = Dd::new(1.0);
    let mut alt = Dd::new(1.0);
    let mut pos = Dd::new(1.0);
    for k in 1..=n {
        let kf = k as f64;
        // (n + k)(n − k + 1) is an exact integer, 2k·x an exact double-double.
        let (dh, dl) = two_prod(2.0 * kf, x);
        term = term
            .mul_f64((nf + kf) * (nf - kf + 1.0))
            .div(Dd { hi: dh, lo: dl });
        pos = pos.add(term);
        alt = if !k.is_multiple_of(2) {
            alt.sub(term)
        } else {
            alt.add(term)
        };
    }
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
    if n <= 10 && (1.0e-3..=700.0).contains(&x) {
        // kₙ = (π/(2x))·e^(−x)·Σ aₖ tᵏ, aₖ = (n + k)!/(k!(n − k)!) (exact integers), t = 1/(2x):
        // Horner's rule in double-double on the positive terms.
        let nf = n as f64;
        let mut a = [0.0f64; 21];
        a[0] = 1.0;
        for k in 1..=n as usize {
            let kf = k as f64;
            a[k] = a[k - 1] * (nf + kf) * (nf - kf + 1.0) / kf;
        }
        let th = 0.5 / x;
        let (p, pe) = two_prod(th, 2.0 * x);
        let t = Dd {
            hi: th,
            lo: th * ((1.0 - p) - pe),
        };
        let mut acc = Dd::new(a[n as usize]);
        for k in (0..n as usize).rev() {
            acc = acc.mul(t).add_f64(a[k]);
        }
        let (em, ek) = exp_parts(-x, 0.0);
        let v = acc.mul(t).mul(em).mul(Dd {
            hi: core::f64::consts::PI,
            lo: 1.224_646_799_147_353_2e-16,
        });
        return ldexp(v.value(), ek);
    }
    // Sₙ: S₀ = 1/x, S₁ = (1 + 1/x)/x, S_(m+1) = S_(m−1) + (2m + 1)/x·S_m.
    let xd = Dd::new(x);
    let s0 = Dd::new(1.0).div(xd);
    let (m, e) = if n == 0 {
        (s0, 0)
    } else {
        let s1 = s0.add_f64(1.0).div(xd);
        let (mut a, mut b) = (s0, s1);
        let mut e = 0i32;
        for m in 1..n {
            // Rescale before the step that would overflow (x ≥ 2⁻⁵⁰⁰).
            let c = (2 * m + 1) as f64 / x;
            while b.hi * c > pow2i(900) {
                a = a.mul_f64(pow2i(-BIG));
                b = b.mul_f64(pow2i(-BIG));
                e += BIG;
            }
            let t = b.mul_f64((2 * m + 1) as f64).div(xd).add(a);
            a = b;
            b = t;
        }
        (b, e)
    };
    let (em, ek) = exp_parts(-x.min(1.0e6), 0.0);
    let v = m.mul(em).mul(Dd {
        hi: FRAC_PI_2,
        lo: 6.123_233_995_736_766e-17,
    });
    ldexp(v.value(), e + ek)
}
