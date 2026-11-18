//! The Bessel functions of a real order `ν ≥ 0` and an argument `x > 0` (both finite), in `f64`,
//! with `ν = n + μ`, `|μ| ≤ 1/2`:
//!
//! * `J_ν`, `Y_ν`: Hankel's expansion where `x ≥ 30` and `x ≥ 0.6ν²`; else the values at `μ` and
//!   `μ + 1` (Hankel's expansion from 30; below, `J` by its power series in double-double and `Y`
//!   by Temme's series below 2 or Steed's CF2 summed from the tail above), then up by the
//!   recurrences in double-double (`Y` always, `J` while `x > ν`); for `x ≤ ν`, `J_ν` by its series
//!   or by the Wronskian with CF1. Integer orders start from `J₀, J₁, Y₀, Y₁`.
//! * `I_ν`, `K_ν`: the power series of `I` (positive terms, double-double) below 30, else Hankel's
//!   expansion or the Wronskian with CF1; `K_μ`, `K_(μ+1)` by Temme's series below 1/2, the
//!   Wronskian with `I_μ`, `I_(μ+1)` and CF2 for `K_(μ+1)/K_μ` up to 30, Hankel's expansion beyond,
//!   then up by the recurrence. Integer orders start from `I₀, I₁, K₀, K₁`.
//! * The scale of the recurrences and of `e^(∓x)` is kept in exponents apart, and the orders above
//!   `NU_MAX` are decided by Debye's exponents (see there).

use super::dd::{Dd, fast_two_sum, two_prod};
use super::ik01::{i0, i1, k0, k1};
use super::int01::{TWO_OVER_PI, j0, j1, jy01_large, y0, y1};
use super::kernels::{exp_parts, ldexp, ln_dd, mla, sin_cos_quadrant_dd};
use super::tables_ik::{RGAM, RGAM_DD};
use core::f64::consts::PI;

const EPS: f64 = 1.0 / (1u64 << 55) as f64;
const PI_DD: Dd = Dd {
    hi: PI,
    lo: 1.224_646_799_147_353_2e-16,
};
const PIO2_DD: Dd = super::kernels::PIO2;
const LN2_DD: Dd = Dd {
    hi: core::f64::consts::LN_2,
    lo: 2.319_046_813_846_299_6e-17,
};
/// Rescaling of the recurrences: by 2^±BIG beyond 2^BIG.
const BIG: i32 = 600;

/// `(sin πx, cos πx)`, exact at the integers and half-integers.
pub(crate) fn sin_cos_pi(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let r = x - 2.0 * (0.5 * x).round_ties_even();
    let n = (2.0 * r).round_ties_even();
    let f = r - 0.5 * n;
    let (h, l) = two_prod(f, PI);
    let (h, l) = fast_two_sum(h, f.mul_add(PI_DD.lo, l));
    let (s, c) = sin_cos_quadrant_dd(n as i64, Dd { hi: h, lo: l });
    (s.value(), c.value())
}

/// `1/Γ(1 + μ)`, `1/Γ(1 − μ)` and Temme's `Γ₁ = (1/Γ(1 − μ) − 1/Γ(1 + μ))/(2μ)`,
/// `Γ₂ = (1/Γ(1 − μ) + 1/Γ(1 + μ))/2`, for `|μ| ≤ 1/2`.
#[inline]
fn temme_gammas(mu: f64) -> (f64, f64, f64, f64) {
    let m2 = mu * mu;
    let mut even = 0.0f64;
    let mut odd = 0.0f64;
    let n = RGAM.len();
    let mut k = n;
    while k > 0 {
        k -= 1;
        if k.is_multiple_of(2) {
            even = even.mul_add(m2, RGAM[k]);
        } else {
            odd = odd.mul_add(m2, RGAM[k]);
        }
    }
    let plus = mu.mul_add(odd, even);
    let minus = (-mu).mul_add(odd, even);
    (plus, minus, -odd, even)
}

/// `sinh(σ)/σ` and `cosh σ`.
#[inline]
fn sinhc_cosh(s: f64) -> (f64, f64) {
    if s.abs() < 0.5 {
        let s2 = s * s;
        let sc = 1.0
            + s2 * (1.0 / 6.0
                + s2 * (1.0 / 120.0
                    + s2 * (1.0 / 5040.0
                        + s2 * (1.0 / 362_880.0
                            + s2 * (1.0 / 39_916_800.0 + s2 * (1.0 / 6_227_020_800.0))))));
        let ch = 1.0
            + s2 * (0.5
                + s2 * (1.0 / 24.0
                    + s2 * (1.0 / 720.0
                        + s2 * (1.0 / 40_320.0
                            + s2 * (1.0 / 3_628_800.0 + s2 * (1.0 / 479_001_600.0))))));
        return (sc, ch);
    }
    let e = exp(s);
    let ei = 1.0 / e;
    (0.5 * (e - ei) / s, 0.5 * (e + ei))
}

/// `e^(hi + lo)` of a double-double exponent in the finite range.
#[inline]
fn exp_dd_value(e: Dd) -> f64 {
    let (m, k) = exp_parts(e.hi, e.lo);
    ldexp(m.value(), k)
}

/// `sinh(σ)/σ` and `cosh σ` given `e^σ`.
#[inline]
fn sinhc_cosh_with(s: f64, es: f64) -> (f64, f64) {
    if s.abs() < 0.5 {
        return sinhc_cosh(s);
    }
    let ei = 1.0 / es;
    (0.5 * (es - ei) / s, 0.5 * (es + ei))
}

/// `e^x`, finite range.
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    super::kernels::exp(x)
}

/// `μπ/sin(μπ)`, 1 at 0.
#[inline]
fn pi_over_sinc(mu: f64) -> f64 {
    if mu == 0.0 {
        1.0
    } else {
        PI * mu / sin_cos_pi(mu).0
    }
}

/// `ln Γ(z)` for `z ≥ 10` as a double-double: Stirling's series to `z⁻¹⁵`.
pub(crate) fn ln_gamma_large(z: f64) -> Dd {
    // (z − 1/2)·ln z − z + ln √(2π) + Σ B₂ₖ/(2k(2k − 1) z^(2k−1)).
    const LN_SQRT_2PI: Dd = Dd {
        hi: 0.918_938_533_204_672_8,
        lo: -3.878_294_158_067_241_4e-17,
    };
    let l = ln_dd(z);
    let (zh, zl) = fast_two_sum(z, -0.5);
    let a = l.mul(Dd { hi: zh, lo: zl });
    let zi = 1.0 / z;
    let s = zi * zi;
    let series = zi
        * (1.0 / 12.0
            + s * (-1.0 / 360.0
                + s * (1.0 / 1260.0
                    + s * (-1.0 / 1680.0
                        + s * (1.0 / 1188.0
                            + s * (-691.0 / 360_360.0
                                + s * (1.0 / 156.0 + s * (-3617.0 / 122_400.0))))))));
    a.add_f64(-z).add(LN_SQRT_2PI).add_f64(series)
}

/// `(x/2)^ν / Γ(ν + 1)` as `m·2^e` (`m` a double-double), for `ν ≥ 0`, `x > 0`.
fn series_prefactor(nu: f64, x: f64) -> (Dd, i32) {
    // ν·ln(x/2) in double-double (x/2 is exact but for the subnormals).
    let l = if x > 4.0 * f64::MIN_POSITIVE {
        ln_dd(0.5 * x)
    } else {
        ln_dd(x).sub(Dd {
            hi: core::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        })
    };
    let a = l.mul_f64(nu);
    if nu + 1.0 >= 10.0 {
        // ln Γ(ν + 1) with ν + 1 = z + err exactly: ln Γ(z) + err·ψ(z).
        let (z, err) = super::dd::two_sum(nu, 1.0);
        let psi = ln_dd(z).hi - 0.5 / z;
        let e = a.sub(ln_gamma_large(z).add_f64(err * psi));
        let (m, k) = exp_parts(e.hi, e.lo);
        return (m, k);
    }
    // Γ(ν + 1) = Γ(1 + μ)·(μ + 1)(μ + 2)…(μ + n), ν = μ + n, |μ| ≤ 1/2.
    let n = nu.round_ties_even();
    let mu = nu - n;
    let rg = rgam_dd(mu);
    let mut g = Dd::new(1.0);
    let mut k = 1.0;
    while k <= n {
        let (h, l2) = fast_two_sum(k, mu);
        g = g.mul(Dd { hi: h, lo: l2 });
        k += 1.0;
    }
    let (m, k) = exp_parts(a.hi, a.lo);
    (m.mul(rg).div(g), k)
}

/// `Σ (∓x²/4)^k / (k!·(ν + 1)ₖ)` (minus for `J`, plus for `I`), terms and sum in
/// double-double (in `f64` the roundings of the terms, `k` of them in the `k`-th, add up to
/// several ulps once the terms grow).
fn series_sum(nu: f64, x: f64, minus: bool) -> Dd {
    let (qh, ql) = two_prod(0.5 * x, 0.5 * x);
    let q = if minus {
        Dd { hi: -qh, lo: -ql }
    } else {
        Dd { hi: qh, lo: ql }
    };
    let mut term = Dd::new(1.0);
    let mut sum = Dd::new(1.0);
    let mut k = 1.0f64;
    loop {
        term = term.mul(q).mul(recip_dd(k, nu));
        sum = sum.add(term);
        if term.hi.abs() <= EPS * 0.25 * sum.hi.abs() || k > 500.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// `J_ν(x)` or `I_ν(x)` by the power series, times `scale` (`e^(−x)` for the scaled `I`).
fn series(nu: f64, x: f64, minus: bool, scaled: bool) -> f64 {
    let (m, k) = series_prefactor(nu, x);
    let s = series_sum(nu, x, minus);
    let v = m.mul(s);
    if scaled {
        let (e, ke) = exp_parts(-x, 0.0);
        return ldexp(v.mul(e).value(), k + ke);
    }
    ldexp(v.value(), k)
}

/// A value `m·2^e`.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    m: f64,
    e: i32,
}

impl Scaled {
    fn value(self) -> f64 {
        ldexp(self.m, self.e)
    }
}

/// CF1 for `J`: `f = J′_ν/J_ν` and the sign of `J_ν` relative to the start (Lentz's method).
fn cf1_j(nu: f64, x: f64) -> (f64, f64) {
    const TINY: f64 = 1e-300;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let mut h = (nu * xi).max(TINY);
    let mut b = xi2 * nu;
    let mut d = 0.0f64;
    let mut c = h;
    let mut sign = 1.0;
    for _ in 0..1_000_000 {
        b += xi2;
        d = b - d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b - 1.0 / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = c * d;
        h *= del;
        if d < 0.0 {
            sign = -sign;
        }
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (h, sign)
}

/// CF1 for `I`: `f = I′_ν/I_ν = ν/x + 1/(2(ν + 1)/x + 1/(2(ν + 2)/x + …))`: a forward pass
/// (Lentz) finds the length, then the fraction is summed from the tail with 16 more terms.
fn cf1_i(nu: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let mut b = xi2 * nu;
    let mut d = 0.0f64;
    let mut c = (nu * xi).max(TINY);
    let mut n = 0u64;
    while n < 1_000_000 {
        n += 1;
        b += xi2;
        d = 1.0 / (b + d);
        c = b + 1.0 / c;
        let del = c * d;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    let mut t = 0.0f64;
    let mut k = (n + 16) as f64;
    while k >= 1.0 {
        t = 1.0 / (2.0 * (nu + k) / x + t);
        k -= 1.0;
    }
    nu / x + t
}

/// Temme's series for `x ≤ 2`, `|μ| ≤ 1/2`: `(Y_μ, Y_(μ+1))`.
fn temme_y(mu: f64, x: f64) -> (f64, f64) {
    let x2 = 0.5 * x;
    let pimu = PI * mu;
    let fact = pi_over_sinc(mu);
    // −ln(x/2) = ln 2 − ln x (x/2 is inexact in the subnormals).
    let dd_ = LN2_DD.sub(ln_dd(x));
    let d = dd_.value();
    let edd = dd_.mul_f64(mu);
    let e = edd.value();
    // e^e from the double-double exponent: e reaches ±372, where its rounding would cost
    // hundreds of ulps.
    let ee = exp_dd_value(edd);
    let (fact2, ch) = sinhc_cosh_with(e, ee);
    let (gampl, gammi, gam1, gam2) = temme_gammas(mu);
    let mut ff = 2.0 / PI * fact * (gam1 * ch + gam2 * fact2 * d);
    let mut p = ee / (gampl * PI);
    let mut q = 1.0 / (ee * PI * gammi);
    let pimu2 = 0.5 * pimu;
    let fact3 = if pimu2 == 0.0 {
        1.0
    } else {
        sin_cos_pi(0.5 * mu).0 / pimu2
    };
    let r = PI * pimu2 * fact3 * fact3;
    let mut c = 1.0;
    let dd = -x2 * x2;
    let mut sum = ff + r * q;
    let mut sum1 = p;
    let mu2 = mu * mu;
    let mut i = 1.0f64;
    loop {
        // h = p_i − i·f_i = (μ·p_(i−1) − i²·f_(i−1) − i·q_(i−1))/(i² − μ²), without the
        // cancellation of the difference.
        let den = i * i - mu2;
        let h = (mu * p - i * (i * ff + q)) / den;
        ff = (i * ff + p + q) / den;
        c *= dd / i;
        p /= i - mu;
        q /= i + mu;
        let del = c * (ff + r * q);
        sum += del;
        let del1 = c * (h - i * r * q);
        sum1 += del1;
        if (del.abs() <= EPS * sum.abs().max(1e-300) && del1.abs() <= EPS * sum1.abs()) || i > 60.0
        {
            break;
        }
        i += 1.0;
    }
    (-sum, -sum1 * (2.0 / x))
}

/// Steed's CF2 for `J`, `Y` at `x ≥ 2`: `p + iq = (J′_μ + iY′_μ)/(J_μ + iY_μ)
/// = −1/(2x) + i + (i/x)·a₁/(b₁ + a₂/(b₂ + …))`, `aₖ = (k − 1/2)² − μ²`, `bₖ = 2(x + ki)`,
/// summed from the tail, `120/x + 8` terms (measured for `|μ| ≤ 3/2` against mpmath: 57 terms
/// at `x = 2`, 14 at 8, 7 at 30 reach 2⁻⁵⁷), which keeps the rounding errors from piling up.
fn cf2_jy(mu: f64, x: f64) -> (f64, f64) {
    let mu2 = mu * mu;
    let (mut tr, mut ti) = (0.0f64, 0.0f64);
    let mut k = (120.0 / x).ceil() + 8.0;
    while k >= 1.0 {
        let a = (k - 0.5) * (k - 0.5) - mu2;
        let (br, bi) = (2.0 * x + tr, 2.0 * k + ti);
        let den = br * br + bi * bi;
        tr = a * br / den;
        ti = -a * bi / den;
        k -= 1.0;
    }
    // p + iq = −1/(2x) + i + (i/x)·t.
    let p = -0.5 / x - ti / x;
    let q = 1.0 + tr / x;
    (p, q)
}

/// `2/x` as a double-double, for the recurrences (a rounded `2/x` would bias every step alike).
#[inline(always)]
fn two_over(x: f64) -> (f64, f64) {
    let h = 2.0 / x;
    (h, (-h).mul_add(x, 2.0) / x)
}

/// `Y` (or `J`, for `x` above the orders; `K` with `plus`) up from `(F_μ, F_(μ+1))` to
/// `(F_ν, F_(ν+1)/F_ν)`, `ν = μ + n`, in double-double (an `f64` run gathers an ulp every few
/// steps), the scale kept apart.
fn up_dd(mu: f64, x: f64, n: u64, f_mu: f64, f_mu1: f64, plus: bool) -> (Scaled, f64) {
    let (xh, xl) = two_over(x);
    let xi2 = Dd { hi: xh, lo: xl };
    let (mut a, mut b) = (Dd::new(f_mu), Dd::new(f_mu1));
    let mut e = 0i32;
    let mut i = 1u64;
    while i <= n {
        let (mh, ml) = super::dd::two_sum(mu, i as f64);
        let c = xi2.mul(Dd { hi: mh, lo: ml });
        // Keep c·b in range: rescale first (c reaches 2/x).
        while b.hi.abs() * c.hi.abs() > pow2(900) {
            a = a.mul_f64(pow2(-BIG));
            b = b.mul_f64(pow2(-BIG));
            e += BIG;
        }
        let cb = c.mul(b);
        let t = if plus { cb.add(a) } else { cb.sub(a) };
        a = b;
        b = t;
        i += 1;
    }
    (Scaled { m: a.value(), e }, b.value() / a.value())
}

/// `Y` (or `J`) up, see `up_dd`.
fn up(mu: f64, x: f64, n: u64, f_mu: f64, f_mu1: f64) -> (Scaled, f64) {
    up_dd(mu, x, n, f_mu, f_mu1, false)
}

#[inline(always)]
fn pow2(e: i32) -> f64 {
    super::kernels::pow2i(e)
}

/// `1/Γ(1 + μ)` for `|μ| ≤ 1/2` as a double-double.
#[inline]
fn rgam_dd(mu: f64) -> Dd {
    super::int01::poly_dd(
        mu,
        0.0,
        super::int01::dd2(&RGAM_DD, 0),
        super::int01::dd2(&RGAM_DD, 2),
        &RGAM_DD[4..],
    )
}

/// `J_μ(x)` and `J_(μ+1)(x)` for `|μ| ≤ 1/2`, `0 < x ≤ 32`, by the power series summed in
/// double-double (its terms cancel down to `e^(−x)` of the largest).
fn j_pair_series(mu: f64, x: f64) -> (Dd, Dd) {
    let l = if x > 4.0 * f64::MIN_POSITIVE {
        ln_dd(0.5 * x)
    } else {
        ln_dd(x).sub(Dd {
            hi: core::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        })
    };
    let a = l.mul_f64(mu);
    let (m, k) = exp_parts(a.hi, a.lo);
    let pre = m.mul(rgam_dd(mu));
    let pre = Dd {
        hi: ldexp(pre.hi, k),
        lo: ldexp(pre.lo, k),
    };
    // J_μ = pre·Σ tₖ and J_(μ+1) = −(2/x)·pre·Σ k·tₖ, one run over the terms.
    let (s0, s1) = pair_sum(mu, x, true);
    let j0 = pre.mul(s0);
    let j1 = pre.mul(s1).mul_f64(-2.0 / x);
    (j0, j1)
}

/// `(Σ tₖ, Σ k·tₖ)`, `tₖ = (∓x²/4)^k/(k!·(μ + 1)ₖ)` (minus for `J`), in double-double, to a
/// relative 2⁻⁶⁴ of the larger sum.
fn pair_sum(mu: f64, x: f64, minus: bool) -> (Dd, Dd) {
    let (qh, ql) = two_prod(0.5 * x, 0.5 * x);
    let q = if minus {
        Dd { hi: -qh, lo: -ql }
    } else {
        Dd { hi: qh, lo: ql }
    };
    let mut term = Dd::new(1.0);
    let mut s0 = Dd::new(1.0);
    let mut s1 = Dd::new(0.0);
    let mut kf = 1.0f64;
    loop {
        term = term.mul(q).mul(recip_dd(kf, mu));
        s0 = s0.add(term);
        s1 = s1.add(term.mul_f64(kf));
        let big = s0.hi.abs().max(s1.hi.abs());
        if term.hi.abs() * kf <= 5.4e-20 * big || kf > 400.0 {
            break;
        }
        kf += 1.0;
    }
    (s0, s1)
}

/// `1/(k·(μ + k))` as a double-double (one division and a correction).
#[inline(always)]
fn recip_dd(k: f64, mu: f64) -> Dd {
    let (h, l) = super::dd::two_sum(mu, k);
    let d = Dd { hi: h, lo: l }.mul_f64(k);
    let r = 1.0 / d.hi;
    let e = (-d.hi).mul_add(r, 1.0) - d.lo * r;
    Dd { hi: r, lo: r * e }
}

/// `(e^(−x)·I_μ, e^(−x)·I_(μ+1))` (or without the factor) for `|μ| ≤ 1/2` and `x` below 32:
/// `I_(μ+1) = (2/x)·pre·Σ k·tₖ`.
fn i_pair_series(mu: f64, x: f64, scaled: bool) -> (f64, f64) {
    let (m, k) = series_prefactor(mu, x);
    let (s0, s1) = pair_sum(mu, x, false);
    let (m, k) = if scaled {
        let (e, ke) = exp_parts(-x, 0.0);
        (m.mul(e), k + ke)
    } else {
        (m, k)
    };
    (
        ldexp(m.mul(s0).value(), k),
        ldexp(m.mul(s1).mul_f64(2.0 / x).value(), k),
    )
}

/// `(J_μ, J_(μ+1), Y_μ, Y_(μ+1))` at `x > 0` for `|μ| ≤ 1/2` (`μ = 0` from the dedicated `J₀`,
/// `J₁`, `Y₀`, `Y₁`); `want_j`, `want_y` false skip what is not needed (NaN in its place).
fn base(mu: f64, x: f64, want_j: bool, want_y: bool) -> (f64, f64, f64, f64) {
    if mu == 0.0 {
        if x >= 24.0 {
            let (a, b) = jy01_large(x, false);
            let (c, d) = jy01_large(x, true);
            return (a, c, b, d);
        }
        return if want_j {
            (j0(x), j1(x), y0(x), y1(x))
        } else {
            (f64::NAN, f64::NAN, y0(x), y1(x))
        };
    }
    if x >= 30.0
        && let (Some((p0, q0)), Some((p1, q1))) = (hankel(mu, x), hankel(mu + 1.0, x))
    {
        let (ja, ya) = jy_hankel(mu, x, p0, q0);
        let (jb, yb) = jy_hankel(mu + 1.0, x, p1, q1);
        return (ja, jb, ya, yb);
    }
    if x < 2.0 {
        let (ya, yb) = temme_y(mu, x);
        if !want_j {
            return (f64::NAN, f64::NAN, ya, yb);
        }
        let (ja, jb) = j_pair_series(mu, x);
        return (ja.value(), jb.value(), ya, yb);
    }
    // 2 ≤ x < 30: J from the series; Y from CF2, p + iq = (J′ + iY′)/(J + iY):
    // Y_μ = (p·J_μ − J′_μ)/q and Y′_μ = q·J_μ + p·Y_μ.
    let (ja, jb) = j_pair_series(mu, x);
    if !want_y {
        return (ja.value(), jb.value(), f64::NAN, f64::NAN);
    }
    let (p, q) = cf2_jy(mu, x);
    let jd = ja.mul_f64(mu / x).sub(jb);
    let ya = ja.mul_f64(p).sub(jd).value() / q;
    let yd = ja.value().mul_add(q, p * ya);
    let yb = (mu / x).mul_add(ya, -yd);
    (ja.value(), jb.value(), ya, yb)
}

/// `Y_ν(x) = −Γ(ν)(2/x)^ν/π·(1 + O(x²))` for `ν ≥ 1/2` and `x < 10⁻¹⁵⁰` (the next terms are `O(x^(2ν))`), as `m·2^e`:
/// `−1/(πν·(x/2)^ν/Γ(ν + 1))`.
pub(crate) fn y_tiny(nu: f64, x: f64) -> (Dd, i32) {
    let (m, k) = series_prefactor(nu, x);
    (Dd::new(-1.0).div(PI_DD.mul_f64(nu).mul(m)), -k)
}

/// `K_ν(x) = Γ(ν)(2/x)^ν/2·(1 + O(x²))` for `ν ≥ 1/2` and `x < 10⁻¹⁵⁰` (the next terms are `O(x^(2ν))`), as `m·2^e`.
pub(crate) fn k_tiny(nu: f64, x: f64) -> (Dd, i32) {
    let (m, k) = series_prefactor(nu, x);
    (Dd::new(0.5).div(m.mul_f64(nu)), -k)
}

/// The orders beyond which the paths of cost `O(ν)` (the recurrences) are not taken: there the
/// results are those that overflow or underflow by the exponents of Debye's expansions, and NaN
/// otherwise (not resolved: the uniform expansions are for the optimisation pass).
const NU_MAX: f64 = 1.0e7;

/// Debye's exponent `ν·η(x/ν)`, `η(z) = √(1 + z²) + ln(z/(1 + √(1 + z²)))`, of `I_ν ~ e^(νη)`
/// and `K_ν ~ e^(−νη)` (negative for `x ≪ ν`); and for `x < ν` that of `J_ν ~ e^(−ν(α − tanh α))`,
/// `sech α = x/ν` (`Y_ν ~ −e^(+…)`).
fn debye_exponents(nu: f64, x: f64) -> (f64, f64) {
    let z = x / nu;
    let r = (1.0 + z * z).sqrt();
    let eta = nu * (r + ln_dd(z / (1.0 + r)).hi);
    let jexp = if z < 1.0 {
        let t = (1.0 - z * z).sqrt();
        nu * (ln_dd((1.0 + t) / z).hi - t)
    } else {
        0.0
    };
    (eta, jexp)
}

/// From this order on `I_ν` and `K_ν` come from Debye's uniform expansions (`U₀` to `U₁₂`: the
/// next term is below `2⁻⁵⁸` of the value).
const DEBYE_NU: f64 = 50.0;

/// Below this `x`, `K_μ` and `K_(μ+1)` come from Temme's series.
const TEMME_K_X: f64 = 0.5;

/// Debye's uniform expansions of `I_ν(x)` (`want_i`) or `K_ν(x)`, scaled by `e^(∓x)` when
/// `scaled`: with `r = √(ν² + x²)`, `νη = r + ν·ln(x/(ν + r))` in double-double and `p = ν/r`,
/// `I = e^(νη)/√(2πr)·Σ Uₖ(p)/νᵏ`, `K = √(π/2)·e^(−νη)/√r·Σ (−1)ᵏ Uₖ(p)/νᵏ`, `k ≤ 12`.
fn ik_debye(nu: f64, x: f64, want_i: bool, scaled: bool) -> f64 {
    let (a, ae) = two_prod(nu, nu);
    let (b, be) = two_prod(x, x);
    let r2 = Dd { hi: a, lo: ae }.add(Dd { hi: b, lo: be });
    let r = r2.sqrt();
    let s = r.add_f64(nu);
    // ln(ν + r) of a double-double: ln(h) + l/h.
    let ls = ln_dd(s.hi).add_f64(s.lo / s.hi);
    let eta = r.add(ln_dd(x).sub(ls).mul_f64(nu));
    let e = match (want_i, scaled) {
        (true, false) => eta,
        (true, true) => eta.add_f64(-x),
        (false, false) => eta.neg(),
        (false, true) => eta.neg().add_f64(x),
    };
    if e.hi > 1.0e6 {
        return f64::INFINITY;
    }
    if e.hi < -1.0e6 {
        return 0.0;
    }
    let p = nu / r.hi;
    let p2 = p * p;
    let t = 1.0 / nu;
    let sg = if want_i { t } else { -t };
    // Σ_(k=1..12) Uₖ(p)·(±1/ν)^k by Horner's rule in ±1/ν, each Uₖ by Horner's rule in p².
    let mut corr = 0.0f64;
    let mut pk = [0.0f64; 13];
    pk[0] = 1.0;
    for k in 1..13 {
        pk[k] = pk[k - 1] * p;
    }
    for k in (1..13).rev() {
        let row = &super::tables_olver::DEBYE_U[k - 1];
        let mut u = 0.0f64;
        for j in (0..=k).rev() {
            u = mla(u, p2, row[j]);
        }
        corr = sg * (corr + pk[k] * u);
    }
    let sum = Dd::new(1.0).add_f64(corr);
    // √(2πr) or √(2r/π).
    let c = if want_i {
        PI_DD.mul(r).mul_f64(2.0)
    } else {
        r.mul_f64(2.0).div(PI_DD)
    };
    let v = sum.div(c.sqrt());
    let (m, k) = exp_parts(e.hi, e.lo);
    ldexp(v.mul(m).value(), k)
}

/// `x^e` for `x > 0` and a double-double exponent (own arithmetic, for the same bits
/// everywhere), as `m·2^k`.
fn pow_pos(x: f64, e: Dd) -> (Dd, i32) {
    let l = ln_dd(x).mul(e);
    exp_parts(l.hi, l.lo)
}

const TWO_THIRDS: Dd = Dd {
    hi: 0.666_666_666_666_666_6,
    lo: 3.700_743_415_417_188e-17,
};

/// `atan u` for `u ≥ 0`: `π/2 − atan(1/u)` above 1, then `atan(k/8) + atan((u − k/8)/(1 + uk/8))`
/// with the nearest `k`, and the series to `v¹⁷` (`|v| ≤ 1/16`).
fn atan(u: f64) -> f64 {
    let (inv, a) = if u > 1.0 { (true, 1.0 / u) } else { (false, u) };
    let kf = (a * 8.0).round_ties_even();
    let c = 0.125 * kf;
    let v = (a - c) / mla(a, c, 1.0);
    let v2 = v * v;
    let mut s = 0.0f64;
    let mut k = 17.0f64;
    while k > 1.0 {
        s = mla(s, v2, if (k as i64) % 4 == 1 { 1.0 } else { -1.0 } / k);
        k -= 2.0;
    }
    let base = super::tables_olver::ATAN8[kf as usize];
    let r = Dd {
        hi: base[0],
        lo: base[1],
    }
    .add_f64(v * mla(s, v2, 1.0));
    if inv {
        super::kernels::PIO2.sub(r).value()
    } else {
        r.value()
    }
}

/// Olver's uniform expansion for `ν > NU_MAX`, to the terms `A₀ = 1` and `B₀`:
/// `J_ν(νz) ≈ φ·(Ai(ν^(2/3)ζ)/ν^(1/3) + B₀·Ai′(ν^(2/3)ζ)/ν^(5/3))`, `Y` with `−Bi`, `−Bi′`;
/// `ζ`, `φ = (4ζ/(1 − z²))^(1/4)` and `B₀` by their series about `z = 1` for `|1 − z| ≤ 0.06`,
/// else by their closed forms (the next terms are below `10⁻¹⁶` of the value for these orders;
/// the argument of the Airy functions, rounded, bounds the error by a few `max(ν, x)·ε` of `M`).
fn jy_olver(nu: f64, x: f64) -> (f64, f64) {
    use super::tables_olver::{B0, PHI, ZW};
    let z = x / nu;
    let w = 1.0 - z;
    let (zeta, phi, b0) = if w.abs() <= 0.06 {
        (
            w * super::kernels::estrin(w, &ZW),
            super::kernels::estrin(w, &PHI),
            super::kernels::estrin(w, &B0),
        )
    } else {
        // z > 1.06 (below 0.94 the order's J underflows and Y overflows, decided before).
        let s = (z * z - 1.0).sqrt();
        let f = s - atan(s);
        let (m, k) = pow_pos(1.5 * f, TWO_THIRDS);
        let zeta = -ldexp(m.value(), k);
        let phi = (4.0 * -zeta / (s * s)).sqrt().sqrt();
        let b0 =
            -5.0 / (48.0 * zeta * zeta) + (5.0 / (24.0 * s * s * s) + 0.125 / s) / (-zeta).sqrt();
        (zeta, phi, b0)
    };
    // ν^(2/3) in double-double: y carries the phase ν·(2/3)|ζ|^(3/2).
    let (m, k) = pow_pos(nu, TWO_THIRDS);
    let n23d = Dd {
        hi: ldexp(m.hi, k),
        lo: ldexp(m.lo, k),
    };
    let n23 = n23d.hi;
    let n13 = n23.sqrt();
    let y = n23d.mul_f64(zeta).value();
    let ai = super::super::airy::airy_ai(y);
    let aip = super::super::airy::airy_ai_prime(y);
    let bi = super::super::airy::airy_bi(y);
    let bip = super::super::airy::airy_bi_prime(y);
    let c = b0 / (n23 * n23);
    let j = phi / n13 * mla(c, aip, ai);
    let yv = -phi / n13 * mla(c, bip, bi);
    (j, yv)
}

/// `(J_ν(x), Y_ν(x))` for `ν ≥ 0`, `x > 0` finite (`want_j`, `want_y` skip the work of the
/// other where they can).
pub(crate) fn jy(nu: f64, x: f64, want_j: bool, want_y: bool) -> (f64, f64) {
    if nu == 0.0 || nu == 1.0 {
        let one = nu == 1.0;
        if x >= 24.0 {
            return jy01_large(x, one);
        }
        return if one { (j1(x), y1(x)) } else { (j0(x), y0(x)) };
    }
    if hankel_region(nu, x)
        && let Some((p, q)) = hankel(nu, x)
    {
        return jy_hankel(nu, x, p, q);
    }
    // J alone where its series does not cancel.
    let qq = 0.25 * x * x;
    if !want_y && qq <= 0.25 * (nu + 1.0) {
        return (series(nu, x, true, false), f64::NAN);
    }
    if nu > NU_MAX {
        let (_, e) = debye_exponents(nu, x);
        return if x < nu && e > 800.0 {
            let j = if qq <= 0.25 * (nu + 1.0) {
                series(nu, x, true, false)
            } else {
                0.0
            };
            (j, f64::NEG_INFINITY)
        } else {
            jy_olver(nu, x)
        };
    }
    if x < 1.0e-150 && nu >= 0.5 {
        let (m, k) = y_tiny(nu, x);
        let y = ldexp(m.value(), k);
        let j = if want_j {
            series(nu, x, true, false)
        } else {
            f64::NAN
        };
        return (j, y);
    }
    let n = (nu + 0.5).floor();
    let mu = nu - n;
    let n = n as u64;
    let oscillating = x > nu;
    let need_y = want_y || !oscillating;
    let (ja, jb, ya, yb) = base(mu, x, want_j && oscillating, need_y);
    if !need_y {
        let (js, _) = up(mu, x, n, ja, jb);
        return (js.value(), f64::NAN);
    }
    let (ys, ratio) = up(mu, x, n, ya, yb);
    let yv = ys.value();
    if !want_j {
        return (f64::NAN, yv);
    }
    if oscillating {
        let (js, _) = up(mu, x, n, ja, jb);
        return (js.value(), yv);
    }
    if qq <= 0.25 * (nu + 1.0) {
        return (series(nu, x, true, false), yv);
    }
    // x ≤ ν: CF1 gives f = J′_ν/J_ν; J_ν = W/(Y′_ν − f·Y_ν), Y′_ν = (ν/x)·Y_ν − Y_(ν+1).
    let (f, _) = cf1_j(nu, x);
    let w = TWO_OVER_PI.hi / x;
    let jv = Scaled {
        m: w / (ys.m * (nu / x - ratio - f)),
        e: -ys.e,
    }
    .value();
    (jv, yv)
}

/// Hankel's expansions `P, Q` of `J_ν, Y_ν` for large `x` (`None` when the terms do not get
/// small enough), summed in double-double; the rounding of `1/x`, common to the terms (`k`
/// times in the `k`-th), is corrected through `Σ k·tₖ`.
fn hankel(nu: f64, x: f64) -> Option<(Dd, Dd)> {
    let mu = 4.0 * nu * nu;
    let xi = 1.0 / x;
    let rel = (-x).mul_add(xi, 1.0);
    let mut term = 1.0f64;
    let mut p = Dd::new(1.0);
    let mut q = Dd::new(0.0);
    let (mut dp, mut dq) = (0.0f64, 0.0f64);
    let mut last = f64::INFINITY;
    let mut k = 1.0f64;
    while k < 200.0 {
        let o = 2.0 * k - 1.0;
        term *= (mu - o * o) * xi / (8.0 * k);
        let t = term.abs();
        if t > last && t > 1e-300 {
            return None;
        }
        last = t;
        // Signs (−1)^⌊k/2⌋, P of the even k and Q of the odd.
        let kk = k as u64;
        let s = if (kk / 2).is_multiple_of(2) {
            term
        } else {
            -term
        };
        if kk.is_multiple_of(2) {
            p = p.add_f64(s);
            dp = k.mul_add(s, dp);
        } else {
            q = q.add_f64(s);
            dq = k.mul_add(s, dq);
        }
        if t <= EPS * 0.25 || term == 0.0 {
            return Some((p.add_f64(dp * rel), q.add_f64(dq * rel)));
        }
        k += 1.0;
    }
    None
}

/// `Σ aₖ(ν)/xᵏ` with `aₖ` of Hankel's expansion, signs `(±1)^k`, for `I` (`−`) and `K` (`+`).
fn hankel_ik(nu: f64, x: f64, alternate: bool) -> Option<Dd> {
    let mu = 4.0 * nu * nu;
    let xi = 1.0 / x;
    let rel = (-x).mul_add(xi, 1.0);
    let mut term = 1.0f64;
    let mut s = Dd::new(1.0);
    let mut ds = 0.0f64;
    let mut sgn = 1.0f64;
    let mut last = f64::INFINITY;
    let mut k = 1.0f64;
    while k < 200.0 {
        let o = 2.0 * k - 1.0;
        term *= (mu - o * o) * xi / (8.0 * k);
        let t = term.abs();
        if t > last && t > 1e-300 {
            return None;
        }
        last = t;
        sgn = if alternate { -sgn } else { sgn };
        s = s.add_f64(sgn * term);
        ds = k.mul_add(sgn * term, ds);
        if t <= EPS * 0.25 * s.hi.abs() || term == 0.0 {
            return Some(s.add_f64(ds * rel));
        }
        k += 1.0;
    }
    None
}

/// Whether Hankel's expansion of order `ν` serves at `x`.
#[inline]
fn hankel_region(nu: f64, x: f64) -> bool {
    x >= 30.0 && x >= 0.6 * nu * nu
}

/// `(J_ν(x), Y_ν(x))` from Hankel's expansion, `x` large.
fn jy_hankel(nu: f64, x: f64, p: Dd, q: Dd) -> (f64, f64) {
    // ω = x − (ν/2 + 1/4)π: φ = −(ν/2 + 1/4)π modulo 2π.
    let r = nu - 4.0 * (0.25 * nu).floor();
    let (h, l) = fast_two_sum(0.25, 0.5 * r);
    let phi = PI_DD.mul(Dd { hi: h, lo: l }).neg();
    let (s, c) = super::kernels::sin_cos_shifted_dd(x, phi);
    let a = TWO_OVER_PI.div(Dd::new(x)).sqrt();
    (
        a.mul(p.mul(c).sub(q.mul(s))).value(),
        a.mul(p.mul(s).add(q.mul(c))).value(),
    )
}

/// Temme's series for `x ≤ 2`: `(K_μ, K_(μ+1))`.
fn temme_k(mu: f64, x: f64) -> (f64, f64) {
    let x2 = 0.5 * x;
    let fact = pi_over_sinc(mu);
    // −ln(x/2) = ln 2 − ln x (x/2 is inexact in the subnormals).
    let dd_ = LN2_DD.sub(ln_dd(x));
    let d = dd_.value();
    let edd = dd_.mul_f64(mu);
    let e = edd.value();
    let ee = exp_dd_value(edd);
    let (fact2, ch) = sinhc_cosh_with(e, ee);
    let (gampl, gammi, gam1, gam2) = temme_gammas(mu);
    let mut ff = fact * (gam1 * ch + gam2 * fact2 * d);
    let mut sum = ff;
    let mut p = 0.5 * ee / gampl;
    let mut q = 0.5 / (ee * gammi);
    let mut c = 1.0;
    let dd = x2 * x2;
    let mut sum1 = p;
    let mu2 = mu * mu;
    let mut i = 1.0f64;
    loop {
        let den = i * i - mu2;
        let h = (mu * p - i * (i * ff + q)) / den;
        ff = (i * ff + p + q) / den;
        c *= dd / i;
        p /= i - mu;
        q /= i + mu;
        let del = c * ff;
        sum += del;
        let del1 = c * h;
        sum1 += del1;
        if del.abs() <= EPS * sum.abs() || i > 60.0 {
            break;
        }
        i += 1.0;
    }
    (sum, sum1 * (2.0 / x))
}

/// Steed's CF2 for `K` at `x ≥ 2`: `(e^x·K_μ, e^x·K_(μ+1))`.
fn cf2_k(mu: f64, x: f64) -> (f64, f64) {
    let xi = 1.0 / x;
    let mut b = 2.0 * (1.0 + x);
    let mut d = 1.0 / b;
    let mut h = d;
    let mut delh = d;
    let mut q1 = 0.0f64;
    let mut q2 = 1.0f64;
    let a1 = 0.25 - mu * mu;
    let mut q = a1;
    let mut c = a1;
    let mut a = -a1;
    let mut s = 1.0 + q * delh;
    let mut i = 2.0f64;
    while i < 100_000.0 {
        a -= 2.0 * (i - 1.0);
        c = -a * c / i;
        let qnew = (q1 - b * q2) / a;
        q1 = q2;
        q2 = qnew;
        q += c * qnew;
        b += 2.0;
        d = 1.0 / (b + a * d);
        delh *= b * d - 1.0;
        h += delh;
        let dels = q * delh;
        s += dels;
        if (dels / s).abs() < EPS {
            break;
        }
        i += 1.0;
    }
    let h = a1 * h;
    let kmu = (PI / (2.0 * x)).sqrt() / s;
    let k1 = kmu * (mu + x + 0.5 - h) * xi;
    (kmu, k1)
}

/// `K_(μ+1)(x)/K_μ(x)` by its continued fraction (Steed's CF2 for `K`) summed from the tail,
/// `60/x + 8` terms (measured against mpmath: 56 at `x = 1`, 31 at 2, 11 at 8, 6 at 30 reach
/// 2⁻⁵⁷): `(μ + x + 1/2 − (1/4 − μ²)/(b₁ + a₂/(b₂ + …)))/x`, `aᵢ = −((i − 1/2)² − μ²)`,
/// `bᵢ = 2(x + i)`.
fn k_ratio(mu: f64, x: f64) -> f64 {
    let mu2 = mu * mu;
    let mut t = 0.0f64;
    let mut i = (60.0 / x).ceil() + 8.0;
    while i >= 2.0 {
        let a = mu2 - (i - 0.5) * (i - 0.5);
        t = a / (2.0 * (i + x) + t);
        i -= 1.0;
    }
    let cf = 1.0 / (2.0 * (1.0 + x) + t);
    (mu + x + 0.5 - (0.25 - mu2) * cf) / x
}

/// `(e^x·K_μ, e^x·K_(μ+1))` for `x ≥ 1/2`, `|μ| ≤ 1/2`: Hankel's expansion from 30 on, below it
/// the Wronskian `I_μ·K_(μ+1) + I_(μ+1)·K_μ = 1/x` with `I` from its series (positive terms)
/// and `K_(μ+1)/K_μ` from CF2, so that no step cancels.
fn k_base_scaled(mu: f64, x: f64) -> (f64, f64) {
    if x >= 30.0 {
        if let (Some(a), Some(b)) = (hankel_ik(mu, x, false), hankel_ik(mu + 1.0, x, false)) {
            let c = PIO2_DD.div(Dd::new(x)).sqrt();
            return (c.mul(a).value(), c.mul(b).value());
        }
        return cf2_k(mu, x);
    }
    let h = k_ratio(mu, x);
    let (ia, ib) = i_pair_series(mu, x, true);
    let ka = 1.0 / (x * ia.mul_add(h, ib));
    (ka, ka * h)
}

/// `K` up from `(K_μ, K_(μ+1))` to `(K_ν, K_(ν+1)/K_ν)`, `ν = μ + n`.
fn k_up(mu: f64, x: f64, n: u64, k_mu: f64, k_mu1: f64) -> (Scaled, f64) {
    up_dd(mu, x, n, k_mu, k_mu1, true)
}

/// `(K_ν(x), K_(ν+1)(x)/K_ν(x))` with `K` scaled by `e^x` when `scaled`, `ν ≥ 0`, `x > 0`;
/// the value as `m·2^e`.
fn k_scaled(nu: f64, x: f64, scaled: bool) -> (Scaled, f64) {
    let is_int = nu == nu.floor() && nu < 1e15;
    if is_int {
        return k_up(0.0, x, nu as u64, k0(x, scaled), k1(x, scaled));
    }
    let n = (nu + 0.5).floor();
    let mu = nu - n;
    let (kmu, kmu1) = if x < TEMME_K_X {
        let (a, b) = temme_k(mu, x);
        if scaled {
            let e = exp(x);
            (a * e, b * e)
        } else {
            (a, b)
        }
    } else {
        let (a, b) = k_base_scaled(mu, x);
        if scaled {
            (a, b)
        } else {
            let (m, k) = exp_parts(-x, 0.0);
            let s = m.value();
            // keep the scale apart: e^(−x) = s·2^k.
            let (u, r) = k_up(mu, x, n as u64, a, b);
            return (
                Scaled {
                    m: u.m * s,
                    e: u.e + k,
                },
                r,
            );
        }
    };
    k_up(mu, x, n as u64, kmu, kmu1)
}

/// `K_ν(x)` (scaled by `e^x` when `scaled`), `ν ≥ 0`, `x > 0` finite.
pub(crate) fn k(nu: f64, x: f64, scaled: bool) -> f64 {
    if x < 1.0e-150 && nu >= 0.5 {
        let (m, k) = k_tiny(nu, x);
        return ldexp(m.value(), k);
    }
    if nu >= DEBYE_NU && !hankel_region(nu, x) {
        return ik_debye(nu, x, false, scaled);
    }
    if nu == 0.0 {
        return k0(x, scaled);
    }
    if nu == 1.0 {
        return k1(x, scaled);
    }
    if hankel_region(nu, x)
        && let Some(s) = hankel_ik(nu, x, false)
    {
        let a = PIO2_DD.div(Dd::new(x)).sqrt().mul(s);
        return if scaled {
            a.value()
        } else if x > 1.0e6 {
            0.0
        } else {
            let (m, k) = exp_parts(-x, 0.0);
            ldexp(m.mul(a).value(), k)
        };
    }
    k_scaled(nu, x, scaled).0.value()
}

/// `I_ν(x)` (scaled by `e^(−x)` when `scaled`), `ν ≥ 0`, `x > 0` finite.
pub(crate) fn i(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return i0(x, scaled);
    }
    if nu == 1.0 {
        return i1(x, scaled);
    }
    if hankel_region(nu, x)
        && let Some(s) = hankel_ik(nu, x, true)
    {
        let a = s.div(PI_DD.mul_f64(2.0 * x).sqrt());
        return if scaled {
            a.value()
        } else if x > 720.0 {
            f64::INFINITY
        } else {
            let (m, k) = exp_parts(x, 0.0);
            ldexp(m.mul(a).value(), k)
        };
    }
    let qq = 0.25 * x * x;
    if qq <= 4.0 * (nu + 1.0) || x < 30.0 {
        return series(nu, x, false, scaled);
    }
    if nu >= DEBYE_NU {
        return ik_debye(nu, x, true, scaled);
    }
    // Wronskian: I_ν = 1/(x·(f·K_ν − K′_ν)), K′_ν = (ν/x)·K_ν − K_(ν+1), with e^(∓x) carried.
    let f = cf1_i(nu, x);
    let (kv, ratio) = k_scaled(nu, x, scaled);
    let kp_over_k = nu / x - ratio;
    Scaled {
        m: 1.0 / (x * kv.m * (f - kp_over_k)),
        e: -kv.e,
    }
    .value()
}
