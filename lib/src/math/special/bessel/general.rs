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
    let (s, c) = sin_cos_pi_dd(x);
    (s.value(), c.value())
}

/// `(sin πx, cos πx)` as double-doubles (to about 2⁻¹⁰⁴), exact at the integers and
/// half-integers.
pub(crate) fn sin_cos_pi_dd(x: f64) -> (Dd, Dd) {
    if !x.is_finite() {
        return (Dd::new(f64::NAN), Dd::new(f64::NAN));
    }
    let r = x - 2.0 * (0.5 * x).round_ties_even();
    let n = (2.0 * r).round_ties_even();
    let f = r - 0.5 * n;
    let (h, l) = two_prod(f, PI);
    let (h, l) = fast_two_sum(h, f.mul_add(PI_DD.lo, l));
    sin_cos_quadrant_dd(n as i64, Dd { hi: h, lo: l })
}

/// `I_(−a)(x) = Σ (x/2)^(2k−a)/(k!·Γ(k + 1 − a))` for a non-integer `a` below 100 and `x ≤ 2a + 30`
/// (times `e^(−x)` when `scaled`), its terms and sum in double-double from
/// `t₀ = (x/2)^(−a)/Γ(1 − a) = sin(πa)/(πa·(x/2)^a/Γ(a + 1))`: for the arguments where the reflection
/// `I_a + (2/π)·sin(aπ)·K_a` cancels, about the one zero of `I_(−a)` (for `a` in `(2m − 1, 2m)`, near
/// `x = 0.7a`). The largest term is `2^5.4` of the reflection's terms `T` at `a = 25.5`, `2^12.6` at
/// 49.5, `2^20.6` at 75.5 and `2^28` at 99.5, and the sum keeps about `2⁻¹⁰⁶` of it: the result keeps
/// its relative accuracy (within an ulp of the value down to `|I_(−a)| ≈ 10⁻¹⁰·T` up to `a = 75`),
/// but near `a = 100` it is 8 ulps of the value at `10⁻⁹·T` and 300 at `10⁻¹⁰·T`.
pub(crate) fn i_neg_series(a: f64, x: f64, scaled: bool) -> f64 {
    let (m, k) = series_prefactor(a, x);
    let (s, _) = sin_cos_pi_dd(a);
    let mut t = s.div(PI_DD.mul_f64(a).mul(m));
    let (qh, ql) = two_prod(0.5 * x, 0.5 * x);
    let q = Dd { hi: qh, lo: ql };
    let mut sum = t;
    let mut big = t.hi.abs();
    let mut j = 0.0f64;
    let top = 2.0 * a + 4.0 * x + 200.0;
    while j < top {
        j += 1.0;
        // t_j = t_(j−1)·(x²/4)/(j·(j − a)).
        let (dh, dl) = super::dd::two_sum(j, -a);
        t = t.mul(q).div(Dd { hi: dh, lo: dl }.mul_f64(j));
        sum = sum.add(t);
        big = big.max(t.hi.abs());
        if j > a && t.hi.abs() <= 1.0e-33 * big {
            break;
        }
    }
    if scaled {
        let (e, ke) = exp_parts(-x, 0.0);
        return ldexp(sum.mul(e).value(), ke - k);
    }
    ldexp(sum.value(), -k)
}

/// `a + b·2^e` rounded once, `a` and `b` double-doubles (where one is below `2⁻¹¹⁰` of the other,
/// the other; an infinite `b`, out of range by far, wins).
pub(crate) fn add_scaled(a: Dd, b: Dd, e: i32) -> f64 {
    if !b.hi.is_finite() {
        return b.hi;
    }
    if b.hi == 0.0 {
        return a.value();
    }
    if a.hi == 0.0 {
        return ldexp(b.value(), e);
    }
    let ea = exponent(a.hi);
    let eb = exponent(b.hi) + e;
    if ea > eb + 110 {
        return a.value();
    }
    if eb > ea + 110 {
        return ldexp(b.value(), e);
    }
    // Both to about 1, then the sum, then back: one rounding (but in the subnormals).
    let s = ea.max(eb);
    let a1 = Dd {
        hi: ldexp(a.hi, -s),
        lo: ldexp(a.lo, -s),
    };
    let b1 = Dd {
        hi: ldexp(b.hi, e - s),
        lo: ldexp(b.lo, e - s),
    };
    ldexp(a1.add(b1).value(), s)
}

/// `⌊log₂ |v|⌋` of a finite nonzero `v`.
#[inline]
fn exponent(v: f64) -> i32 {
    let a = v.abs();
    if a < f64::MIN_POSITIVE {
        ((a * pow2(64)).to_bits() >> 52) as i32 - 1023 - 64
    } else {
        (a.to_bits() >> 52) as i32 - 1023
    }
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

/// `(x/2)^ν / Γ(ν + 1)` as `m·2^e` (`m` a double-double), for `ν ≥ 0`, `x > 0`, and for
/// `ν > 10³⁰` with `x ≤ 4√(ν + 1)` (as all the callers have it), where it is below `(2e/√ν)^ν`, 0
/// (its logarithm formed as below would be `∞ − ∞` at the top of the range).
fn series_prefactor(nu: f64, x: f64) -> (Dd, i32) {
    if nu > 1.0e30 {
        return (Dd::new(0.0), 0);
    }
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
        if e.hi < -1.0e6 {
            // Below e^(−10⁶), 0 for every caller (exp_parts serves |e| < 2²⁵ only).
            return (Dd::new(0.0), 0);
        }
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
    if m.hi == 0.0 {
        // Below every range. Neither the sum nor e^(−x) is formed: near the top of the range
        // k·(ν + k) overflows in the sum's terms, and e^(−x) of an x beyond 2²⁵ is out of reach of
        // exp_parts (either gave NaN times 0).
        return 0.0;
    }
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

/// CF1 for `J`, `r = J_(ν+1)/J_ν = 1/(2(ν + 1)/x − 1/(2(ν + 2)/x − …))`, by Lentz's method in
/// `f64`: `(r, n)` with `n` the terms it takes to converge.
fn cf1_j_lentz(nu: f64, x: f64) -> (f64, u64) {
    const TINY: f64 = 1e-300;
    let xi2 = 2.0 / x;
    let mut b = xi2 * nu + xi2;
    // The first step by hand (its numerator is +1): C₁ = b₁ + 1/0⁺, D₁ = 1/b₁, h₁ = 1/b₁.
    let mut d = 1.0 / b;
    let mut c = f64::MAX;
    let mut h = d;
    let mut n = 1u64;
    while n < 1_000_000 {
        n += 1;
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
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (h, n)
}

/// `J_(ν+1)(x)/J_ν(x)` by CF1 summed from the tail in double-double, for `x ≤ ν + 1`. Near the
/// turning point `x ≈ ν` the partial ratios are close to 1 for about `7ν^(1/3)` terms, through
/// which the rounding errors of an `f64` run (forward or backward) are not damped and pile up (86
/// ulps at `ν = 200`, 951 at 2,800), and the Wronskian of `jy` passes them to `J` magnified by up
/// to `ν^(1/3)`. The length: 16 terms beyond the one Lentz's method finds in `f64`, and from
/// `ν = 50` on at least so many that the zero tail costs below `e^(−48)` (Lentz's test, met by
/// steps of `2⁻⁵⁵` that are not yet the tail's size there, stops short near the turning point:
/// `2·10⁻¹⁷` at `ν = 1.8·10⁶`). That error is `e^(−2Φ)` with `Φ = F(ν + n) − F(ν)` the decay of
/// `J_(ν+k)` over the terms, `F(u) = u·acosh(u/x) − √(u² − x²)` Debye's exponent.
///
/// Below `x = ν/2`, unless `whole`, the `f64` value of Lentz's method is returned. It is 5 to 11
/// ulps off there (its forward products gather the roundings), but the Wronskian of `jy` takes
/// the ratio's error with the factor `r/|r − Y_(ν+1)/Y_ν| < 0.08`, so `J` does not see it. The
/// spherical `jₙ`, normalised by a run down from the ratio, takes its error whole and asks for
/// the tail sum (`whole`), whose ratios below `1/4` there damp the roundings instead.
pub(crate) fn ratio_j(nu: f64, x: f64, whole: bool) -> Dd {
    let (r64, n) = cf1_j_lentz(nu, x);
    if !whole && x < 0.5 * nu {
        return Dd::new(r64);
    }
    let mut n = n + 16;
    if nu >= 50.0 {
        let f = |u: f64| {
            if u <= x {
                return 0.0;
            }
            let s = ((u - x) * (u + x)).sqrt();
            u * ln_dd((u + s) / x).hi - s
        };
        let f0 = f(nu);
        while f(nu + n as f64) - f0 < 24.0 && n < 100_000_000 {
            n += n / 4 + 1;
        }
    }
    let (xh, xl) = two_over(x);
    let xi2 = Dd { hi: xh, lo: xl };
    // b_k = (2/x)·(ν + k) from k = n down, t_k = J_(ν+k)/J_(ν+k−1) = 1/(b_k − t_(k+1)). Here
    // b_k ≥ 2 and 0 < t < 1, so neither difference cancels: one two-sum each is enough, and the
    // reciprocal is left unnormalised (its low part below an ulp of the high).
    let (sh, sl) = super::dd::two_sum(nu, n as f64);
    let mut b = xi2.mul(Dd { hi: sh, lo: sl });
    let mut t = Dd::new(0.0);
    let mut k = n;
    while k >= 1 {
        let (s, e) = super::dd::two_sum(b.hi, -t.hi);
        let (dh, dl) = fast_two_sum(s, e + (b.lo - t.lo));
        let r = 1.0 / dh;
        t = Dd {
            hi: r,
            lo: r * ((-dh).mul_add(r, 1.0) - dl * r),
        };
        let (s, e) = super::dd::two_sum(b.hi, -xi2.hi);
        let (bh, bl) = fast_two_sum(s, e + (b.lo - xi2.lo));
        b = Dd { hi: bh, lo: bl };
        k -= 1;
    }
    let (hi, lo) = fast_two_sum(t.hi, t.lo);
    Dd { hi, lo }
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

/// The steps of Temme's series for `Y` taken in double-double (see `temme_y`).
const TEMME_DD: f64 = 2.0;

/// `1/6` as a double-double.
const SIXTH: Dd = Dd {
    hi: 0.166_666_666_666_666_66,
    lo: 9.251_858_538_542_97e-18,
};

/// `(sinh σ/σ, cosh σ)` as double-doubles, given `e^σ`: their series below `|σ| = 1/2` (the
/// first two terms in double-double), else from `e^(±σ)`.
fn sinhc_cosh_dd(s: Dd, es: Dd) -> (Dd, Dd) {
    if s.hi.abs() < 0.5 {
        // 1/(2k + 1)! and 1/(2k)! for k = 2..=9.
        const ODD: [f64; 8] = [
            0.008_333_333_333_333_333,
            0.000_198_412_698_412_698_4,
            2.755_731_922_398_589_3e-6,
            2.505_210_838_544_172e-8,
            1.605_904_383_682_161_3e-10,
            7.647_163_731_819_816e-13,
            2.811_457_254_345_520_6e-15,
            8.220_635_246_624_33e-18,
        ];
        const EVEN: [f64; 8] = [
            0.041_666_666_666_666_664,
            0.001_388_888_888_888_889,
            2.480_158_730_158_73e-5,
            2.755_731_922_398_589e-7,
            2.087_675_698_786_81e-9,
            1.147_074_559_772_972_5e-11,
            4.779_477_332_387_385e-14,
            1.561_920_696_858_622_5e-16,
        ];
        let s2 = s.mul(s);
        let t = s2.hi;
        // Σ σ^(2k)/(2k + 1)! and Σ σ^(2k)/(2k)! from k = 2 on, to σ¹⁸ (the rest below 2⁻⁸⁰).
        let (mut a, mut b) = (0.0f64, 0.0f64);
        for k in (0..8).rev() {
            a = mla(a, t, ODD[k]);
            b = mla(b, t, EVEN[k]);
        }
        let sc = Dd::new(1.0).add(s2.mul(SIXTH.add(s2.mul_f64(a))));
        let ch = Dd::new(1.0).add(s2.mul(Dd::new(0.5).add(s2.mul_f64(b))));
        return (sc, ch);
    }
    let ei = Dd::new(1.0).div(es);
    (es.sub(ei).div(s).mul_f64(0.5), es.add(ei).mul_f64(0.5))
}

/// `1/Γ(1 + μ)`, `1/Γ(1 − μ)`, `Γ₁` and `Γ₂` (see `temme_gammas`) as double-doubles, from the
/// fit `RGAM_DD` (its error, `2⁻⁵⁵·⁵`, the limit of these).
fn temme_gammas_dd(mu: f64) -> (Dd, Dd, Dd, Dd) {
    let c = &RGAM_DD;
    // even = c₀ + c₂μ² + …, odd = c₁ + c₃μ² + …: from c₄ on in f64, c₂ and c₃ in double-double.
    let m2 = mu * mu;
    let (mut even, mut odd) = (0.0f64, 0.0f64);
    let mut k = c.len() - 3;
    while k >= 4 {
        if k.is_multiple_of(2) {
            even = mla(even, m2, c[k + 2]);
        } else {
            odd = mla(odd, m2, c[k + 2]);
        }
        k -= 1;
    }
    let (ph, pl) = two_prod(mu, mu);
    let m2d = Dd { hi: ph, lo: pl };
    let even = Dd::new(c[4]).add(m2d.mul_f64(even));
    let even = Dd { hi: c[0], lo: c[1] }.add(m2d.mul(even));
    let odd = Dd::new(c[5]).add(m2d.mul_f64(odd));
    let odd = Dd { hi: c[2], lo: c[3] }.add(m2d.mul(odd));
    let mo = odd.mul_f64(mu);
    (even.add(mo), even.sub(mo), odd.neg(), even)
}

/// Temme's series for `x ≤ 2`, `|μ| ≤ 1/2`: `(Y_μ, Y_(μ+1))`. Its leading terms are of order 1
/// and cancel (`f₀` against `r·q₀`: 1.03 against −1.01 at `μ = 0.48`, `x = 1.9`), so its
/// ingredients and first `TEMME_DD` steps are formed in double-double (in `f64` throughout `Y`
/// reached 6 ulps of `M`); the steps after, below `(x²/4)^k/k!` of the start, in `f64`.
fn temme_y(mu: f64, x: f64) -> (f64, f64) {
    // μπ/sin(μπ) and r = (2/μ)·sin²(μπ/2).
    let (fact, r) = if mu == 0.0 {
        (Dd::new(1.0), Dd::new(0.0))
    } else {
        let (s1, _) = sin_cos_pi_dd(mu);
        let (s2, _) = sin_cos_pi_dd(0.5 * mu);
        (
            PI_DD.mul_f64(mu).div(s1),
            s2.mul(s2).mul_f64(2.0).div(Dd::new(mu)),
        )
    };
    // d = −ln(x/2) = ln 2 − ln x (x/2 is inexact in the subnormals), σ = μ·d and e^σ from the
    // double-double exponent (σ reaches ±372).
    let d = LN2_DD.sub(ln_dd(x));
    let sg = d.mul_f64(mu);
    let (em, ek) = exp_parts(sg.hi, sg.lo);
    let ee = Dd {
        hi: ldexp(em.hi, ek),
        lo: ldexp(em.lo, ek),
    };
    let (fact2, ch) = sinhc_cosh_dd(sg, ee);
    let (gampl, gammi, gam1, gam2) = temme_gammas_dd(mu);
    let mut ffd = TWO_OVER_PI
        .mul(fact)
        .mul(gam1.mul(ch).add(gam2.mul(fact2).mul(d)));
    let mut pd = ee.div(gampl.mul(PI_DD));
    let mut qd = Dd::new(1.0).div(ee.mul(PI_DD).mul(gammi));
    let (qh, ql) = two_prod(0.5 * x, 0.5 * x);
    let dq = Dd { hi: -qh, lo: -ql };
    let mut cd = Dd::new(1.0);
    let mut sum = ffd.add(r.mul(qd));
    let mut sum1 = pd;
    let (m2h, m2l) = two_prod(mu, mu);
    let mu2d = Dd { hi: m2h, lo: m2l };
    let mut i = 1.0f64;
    while i <= TEMME_DD {
        // As below, in double-double.
        let den = Dd::new(i * i).sub(mu2d);
        let h = pd
            .mul_f64(mu)
            .sub(ffd.mul_f64(i).add(qd).mul_f64(i))
            .div(den);
        ffd = ffd.mul_f64(i).add(pd).add(qd).div(den);
        cd = cd.mul(dq).div(Dd::new(i));
        let (a, b) = super::dd::two_sum(i, -mu);
        pd = pd.div(Dd { hi: a, lo: b });
        let (a, b) = super::dd::two_sum(i, mu);
        qd = qd.div(Dd { hi: a, lo: b });
        let rq = r.mul(qd);
        sum = sum.add(cd.mul(ffd.add(rq)));
        sum1 = sum1.add(cd.mul(h.sub(rq.mul_f64(i))));
        i += 1.0;
    }
    let (mut ff, mut p, mut q, mut c) = (ffd.value(), pd.value(), qd.value(), cd.value());
    let (r, dd) = (r.value(), dq.value());
    let mu2 = mu * mu;
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
        sum = sum.add_f64(del);
        let del1 = c * (h - i * r * q);
        sum1 = sum1.add_f64(del1);
        if (del.abs() <= EPS * sum.hi.abs().max(1e-300) && del1.abs() <= EPS * sum1.hi.abs())
            || i > 60.0
        {
            break;
        }
        i += 1.0;
    }
    let (th, tl) = two_over(x);
    (-sum.value(), -sum1.mul(Dd { hi: th, lo: tl }).value())
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
    let (a, b, e) = up_parts(mu, x, n, f_mu, f_mu1, plus);
    (Scaled { m: a.value(), e }, b.value() / a.value())
}

/// The recurrence of `up_dd`, its last two values `(F_ν, F_(ν+1))` as double-doubles scaled by
/// `2^(−e)`.
fn up_parts(mu: f64, x: f64, n: u64, f_mu: f64, f_mu1: f64, plus: bool) -> (Dd, Dd, i32) {
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
    (a, b, e)
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
    if m.hi == 0.0 {
        // Beyond every range: a finite mantissa with an exponent far out, so that the callers'
        // products stay defined.
        return (Dd::new(-1.0), 1 << 20);
    }
    (Dd::new(-1.0).div(PI_DD.mul_f64(nu).mul(m)), -k)
}

/// `K_ν(x) = Γ(ν)(2/x)^ν/2·(1 + O(x²))` for `ν ≥ 1/2` and `x < 10⁻¹⁵⁰` (the next terms are `O(x^(2ν))`), as `m·2^e`.
pub(crate) fn k_tiny(nu: f64, x: f64) -> (Dd, i32) {
    let (m, k) = series_prefactor(nu, x);
    if m.hi == 0.0 {
        return (Dd::new(1.0), 1 << 20);
    }
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
/// `scaled`, for `ν ≥ 50`, see `debye_parts`.
fn ik_debye(nu: f64, x: f64, want_i: bool, scaled: bool) -> f64 {
    let (m, e) = debye_parts(nu, x, want_i, scaled);
    ldexp(m.value(), e)
}

/// From this order on the plain `I_ν` and `K_ν` take Debye's exponent from `x − z₀ν` (see
/// `debye_parts`): below it `ν` times the error of `ln_dd`, `2⁻⁷²·⁴`, stays below `2⁻⁵⁶`.
const NU_Z0: f64 = 65_536.0;

/// `1/3`, `1/5`, `1/7` and `1/9` as double-doubles.
const RECIP_ODD: [Dd; 4] = [
    Dd {
        hi: 0.333_333_333_333_333_3,
        lo: 1.850_371_707_708_594e-17,
    },
    Dd {
        hi: 0.2,
        lo: -1.110_223_024_625_156_6e-17,
    },
    Dd {
        hi: 0.142_857_142_857_142_85,
        lo: 7.930_164_461_608_26e-18,
    },
    Dd {
        hi: 0.111_111_111_111_111_1,
        lo: 6.167_905_692_361_980_4e-18,
    },
];

/// `1/(2j + 3)` for `j = 0..=36`, rounded (for the series of `debye_a`).
const INV_ODD: [f64; 37] = {
    let mut a = [0.0f64; 37];
    let mut j = 0;
    while j < 37 {
        a[j] = 1.0 / (2 * j + 3) as f64;
        j += 1;
    }
    a
};

/// Debye's uniform expansions of `I_ν(x)` (`want_i`) or `K_ν(x)`, scaled by `e^(∓x)` when
/// `scaled`, for `ν ≥ 50`: with `r = √(ν² + x²)` and `p = ν/r`,
/// `I = e^(νη)/√(2πr)·Σ Uₖ(p)/νᵏ`, `K = √(π/2)·e^(−νη)/√r·Σ (−1)ᵏ Uₖ(p)/νᵏ`, `k ≤ 12`, as `m·2^e`
/// (`m` infinite or 0 where the value is out of range by far).
///
/// The exponent, `νη = r − ν·atanh p`, is the size of the result in `e`-folds: its absolute error
/// is the relative error of the result. `r + ν·ln(x/(ν + r))` serves below `ν = 2¹⁶` (for the
/// scaled forms below `x = 2⁴⁵`); beyond it carries the error `ν·2⁻⁷²` of the logarithm, and for
/// the scaled forms the cancellation of `r − x` (`2⁻¹⁰⁵·x`), and these take its place:
/// * For the scaled forms `νη − x = −xν²/(r(r + x)) − A`, `A = ν(atanh p − p) ≥ 0` (from
///   `νη = x²/r − A`), whose terms have one sign; `A` by its series `νp³·Σ p^(2j)/(2j + 3)` for
///   small `p`, else by `ν·ln((r + ν)/x) − νp` (see `debye_a`; from `ν = 2¹⁶` on `A > 3000` beyond
///   `p = 1/2`, out of range).
/// * From `ν = 2¹⁶` on, the plain forms, whose `x²/r` and `A` (about `0.37ν` each) cancel to `νη`,
///   in range only for `|νη| ≤ 1100` or so: `νη = D·P(D/ν)`, `D = x − z₀ν` with `z₀ = 0.6627…` the
///   root of `η`, its 1152 bits multiplied out exactly (as Payne and Hanek reduce by `2/π`), and
///   `P` the Taylor series of `η(z)/(z − z₀)` about `z₀`.
///
/// `ν` and `x` are scaled by `2⁻⁶⁰⁰` at the top of the range (the exponents and `r` are
/// homogeneous of degree 1).
pub(crate) fn debye_parts(nu: f64, x: f64, want_i: bool, scaled: bool) -> (Dd, i32) {
    let huge = (Dd::new(f64::INFINITY), 0);
    let tiny = (Dd::new(0.0), 0);
    if x < nu * pow2(-30) {
        // νη < ν(1 + ln(x/(2ν))) < −1000: I underflows and K overflows, scaled or not.
        return if want_i { tiny } else { huge };
    }
    let sh = if nu.max(x) > pow2(500) { 600 } else { 0 };
    let (v, w) = (nu * pow2(-sh), x * pow2(-sh));
    let (a, ae) = two_prod(v, v);
    let (b, be) = two_prod(w, w);
    let nu2 = Dd { hi: a, lo: ae };
    let x2 = Dd { hi: b, lo: be };
    let r = nu2.add(x2).sqrt();
    let p = v / r.hi;
    // The exponent of the function: νη, νη − x, −νη or x − νη.
    let e = if nu < NU_Z0 && (!scaled || x < 3.5e13) {
        // r + ν·ln(x/(ν + r)) serves here: ν times the error of ln_dd stays below 2⁻⁵⁶, and for
        // the scaled forms r − x, which cancels, keeps the absolute error 2⁻¹⁰⁵·x of r, below
        // 2⁻⁶⁰ up to 2⁴⁵.
        let s = r.add_f64(v);
        let ls = ln_dd(s.hi).add_f64(s.lo / s.hi);
        let eta = r.add(ln_dd(w).sub(ls).mul_f64(v));
        let t = match (want_i, scaled) {
            (true, false) => eta,
            (true, true) => eta.add_f64(-w),
            (false, false) => eta.neg(),
            (false, true) => eta.neg().add_f64(w),
        };
        Dd {
            hi: ldexp(t.hi, sh),
            lo: ldexp(t.lo, sh),
        }
    } else if scaled {
        let pd = Dd::new(v).div(r);
        let at = debye_a(nu, v, w, r, pd);
        // x − x²/r = x·ν²/(r(r + x)) = p·(x·ν/(r + x)), and the exponent −(that + A) of I,
        // (that + A) of K. Both factors stay normal (p above 2⁻¹⁰¹⁹, x·ν/(r + x) about min(ν, x)):
        // the quotient ν²/(r(r + x)), about (ν/x)²/2, is subnormal below ν/x = 2⁻⁵¹⁰, and its
        // absolute error, 2⁻¹⁰⁷⁵, times x was that of the exponent (3.4 ulps at the top of the
        // range); and ν²·x may overflow.
        let c = pd.mul(Dd::new(v).div(r.add_f64(w)).mul_f64(w));
        let t = c.add(at);
        let s = if want_i { t.neg() } else { t };
        Dd {
            hi: ldexp(s.hi, sh),
            lo: ldexp(s.lo, sh),
        }
    } else {
        match laplace_exponent(nu, x) {
            Ok(t) => {
                if want_i {
                    t
                } else {
                    t.neg()
                }
            }
            Err(positive) => return if positive == want_i { huge } else { tiny },
        }
    };
    // Beyond every range by far (up to 3·10⁷ e-folds exp_parts still gives the scale apart, as
    // k_exp_neg_parts once needed below ν = 3.3·10⁷; from ν = 2¹⁶ on it forms its own exponent).
    if e.hi > 3.0e7 {
        return huge;
    }
    if e.hi < -3.0e7 {
        return tiny;
    }
    debye_value(nu, r, p, e, want_i, sh)
}

/// Debye's `Σ (±1)ᵏUₖ(p)/νᵏ` over `√(2πr)` (`want_i`) or `√(2r/π)`, times `e^e`, as `m·2^k`, for
/// `r` in the units `2^sh` (the rest of `debye_parts`, given its exponent `e`).
fn debye_value(nu: f64, r: Dd, p: f64, e: Dd, want_i: bool, sh: i32) -> (Dd, i32) {
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
    // √(2πr) or √(2r/π), of r in the units 2^sh: the factor 2^(sh/2) goes to the exponent.
    let c = if want_i {
        PI_DD.mul(r).mul_f64(2.0)
    } else {
        r.mul_f64(2.0).div(PI_DD)
    };
    let val = sum.div(c.sqrt());
    let (m, k) = exp_parts(e.hi, e.lo);
    (val.mul(m), k - sh / 2)
}

/// `ν·(atanh p − p)` in the units of `v` (`ν` scaled), `p = v/r`, `w` the scaled `x`: by the series
/// `νp³·Σ p^(2j)/(2j + 3)` up to `p = 1/4`, and up to `p = 1/2` from `ν = 2¹⁶` on (its first four
/// terms in double-double), else by `ν·ln((r + ν)/x) − νp` (`atanh p = ln((1 + p)/(1 − p))/2 =
/// ln((r + ν)/x)`), whose error `ν·2⁻⁷²·⁴` stays below `2⁻⁵⁶` there.
fn debye_a(nu: f64, v: f64, w: f64, r: Dd, p: Dd) -> Dd {
    if p.hi <= 0.25 || (p.hi <= 0.5 && nu >= NU_Z0) {
        let q = p.mul(p);
        // Σ_(j≥4) q^(j−4)/(2j + 3) in f64, as many terms as q needs for 2⁻⁶⁴ (q ≤ 1/4).
        let top = if q.hi < 1.0e-6 {
            7
        } else if q.hi < 1.0 / 256.0 {
            12
        } else if q.hi < 1.0 / 16.0 {
            20
        } else {
            36
        };
        let mut t = 0.0f64;
        for j in (4..=top).rev() {
            t = mla(t, q.hi, INV_ODD[j]);
        }
        let mut s = Dd::new(t);
        for c in RECIP_ODD.iter().rev() {
            s = s.mul(q).add(*c);
        }
        return p.mul(q).mul(s).mul_f64(v);
    }
    // ln of the double-double g = (r + ν)/x: ln(g_hi) + g_lo/g_hi.
    let g = r.add_f64(v).div(Dd::new(w));
    let l = ln_dd(g.hi).add_f64(g.lo / g.hi);
    l.sub(p).mul_f64(v)
}

/// `z₀` rounded to `f64`.
const Z0_HI: f64 = 0.662_743_419_349_181_6;

/// Debye's exponent `νη(x/ν)` for `ν ≥ 2¹⁶` from `D = x − z₀ν` (see `root_exponent`). `Err(D > 0)`
/// where `|D| > 1300`, where `|νη| > |D|` (as `η′ > 1`) is out of range for every order.
fn laplace_exponent(nu: f64, x: f64) -> Result<Dd, bool> {
    use super::tables_olver::{ETA_Z0, ETA_Z0_DD, Z0_BITS};
    root_exponent(nu, x, Z0_HI, &Z0_BITS, &ETA_Z0_DD, &ETA_Z0)
}

/// `ν·g(x/ν)` for `ν ≥ 2¹⁶` and a function `g` with the root `z` (`z_hi` rounded, `bits` its bits
/// after the binary point), from `D = x − zν`: `D·P(D/ν)`, `P(u) = g₁ + g₂u + …` to `g₁₆u¹⁵`
/// (`gₖ = g^(k)(z)/k!`, the first three, `dd`, in double-double, then `c`). `Err(D > 0)` where
/// `|D| > 1300` (beyond the range for Debye's `η` about `z₀` and `η(z) + z` about `z₁`, whose
/// slopes exceed 1).
fn root_exponent(
    nu: f64,
    x: f64,
    z_hi: f64,
    bits: &[u64; 18],
    dd: &[[f64; 2]; 3],
    c: &[f64; 13],
) -> Result<Dd, bool> {
    // D in f64 first, within 2⁻⁵¹·ν of it.
    let dc = x - z_hi * nu;
    if dc.abs() > 1300.0 + nu * pow2(-50) {
        return Err(dc > 0.0);
    }
    let d = laplace_offset(nu, x, bits);
    if d.hi.abs() > 1300.0 {
        return Err(d.hi > 0.0);
    }
    let u = d.div(Dd::new(nu));
    let mut t = 0.0f64;
    for &c in c.iter().rev() {
        t = mla(t, u.hi, c);
    }
    let mut s = Dd::new(t);
    for c in dd.iter().rev() {
        s = s.mul(u).add(Dd { hi: c[0], lo: c[1] });
    }
    Ok(d.mul(s))
}

/// The significand and exponent of a positive normal `v`: `v = m·2^q`, `m` in `[2⁵², 2⁵³)`.
#[inline]
fn mant_exp(v: f64) -> (u64, i32) {
    let b = v.to_bits();
    (
        (b & ((1u64 << 52) - 1)) | (1u64 << 52),
        ((b >> 52) as i32) - 1075,
    )
}

/// `x − zν` to about `2⁻⁷⁰` absolute, for `ν ≥ 2¹⁶`, a `z` in `(0, 1)` given by the bits after
/// its binary point and `x/ν` in `(1/4, 1)` (`z₀`, `z₁` and `x` near `zν`): with `ν = M·2^q` and
/// `x = X·2^p` (`q − p` is 0, 1 or 2), `x − zν = 2^p·(X − z·M·2^(q−p))`, the product `z·M·2^(q−p)`
/// (below `2¹¹⁹` a word) by long multiplication with the bits of `z`: its integer part and as
/// many words of its fraction as `p` needs (the rest, below `2^(55 + p − 64n)`, under `2⁻⁷²`).
fn laplace_offset(nu: f64, x: f64, bits: &[u64; 18]) -> Dd {
    let (m, q) = mant_exp(nu);
    let (xm, p) = mant_exp(x);
    let y = (m as u128) << (q - p);
    let n = ((((p + 126).max(0) + 63) / 64 + 1) as usize).min(bits.len());
    let mut f = [0u64; 18];
    let mut carry = 0u128;
    for i in (0..n).rev() {
        let t = y * bits[i] as u128 + carry;
        f[i] = t as u64;
        carry = t >> 64;
    }
    // X − z₀Y = N − F, F = Σ f[i]·2^(−64(i+1)) in [0, 1): as ±(I + G), G in [0, 1].
    let big_n = xm as i64 - carry as i64;
    let (neg, ipart, g) = if big_n >= 1 {
        // (N − 1) + (1 − F), 1 − F = (¬F) + 2^(−64n).
        let mut g = [0u64; 18];
        let mut c = 1u64;
        for i in (0..n).rev() {
            let (s, o) = (!f[i]).overflowing_add(c);
            g[i] = s;
            c = o as u64;
        }
        (false, big_n - 1 + c as i64, g)
    } else {
        (true, -big_n, f)
    };
    let word = |i: usize| if i < n { g[i] } else { 0 };
    let v = if ipart > 0 {
        // I below 2⁵³ (else |D| ≥ 2^p·I is far beyond 1300): I + G to 2⁻¹⁰⁶ of it.
        let hi = ((word(0) as u128) << 64) | word(1) as u128;
        let gd = {
            let h = (hi >> 75) as f64;
            let l = (hi & ((1u128 << 75) - 1)) as f64 * pow2(-75);
            let (h, l) = fast_two_sum(h, l);
            Dd {
                hi: h * pow2(-53),
                lo: l * pow2(-53),
            }
        };
        Dd::new(ipart as f64).add(gd)
    } else {
        match (0..n).find(|&i| g[i] != 0) {
            None => Dd::new(0.0),
            Some(k) => {
                // 128 bits from the leading one of G: G ≈ top·2^(−lz − 64k − 128).
                let (w0, w1, w2) = (word(k), word(k + 1), word(k + 2));
                let lz = w0.leading_zeros();
                let top = if lz == 0 {
                    ((w0 as u128) << 64) | w1 as u128
                } else {
                    ((w0 as u128) << (64 + lz)) | ((w1 as u128) << lz) | (w2 >> (64 - lz)) as u128
                };
                let h = (top >> 75) as f64;
                let l = (top & ((1u128 << 75) - 1)) as f64 * pow2(-75);
                let (h, l) = fast_two_sum(h, l);
                let e = 75 - lz as i32 - 64 * k as i32 - 128;
                Dd {
                    hi: ldexp(h, e),
                    lo: ldexp(l, e),
                }
            }
        }
    };
    let v = Dd {
        hi: ldexp(v.hi, p),
        lo: ldexp(v.lo, p),
    };
    if neg { v.neg() } else { v }
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
    if y.is_nan() || y.abs() > 1.0e200 {
        // x above about 6.7·10²⁹⁹ (y ≈ −(3x/2)^(2/3) for x ≫ ν): the Airy functions have no phase
        // there (Ai and Bi are 0, their derivatives NaN), and J and Y, whose phase is lost long
        // before (from x ≈ 2⁵² on), are 0 alike, within their envelope √(2/(πx)).
        return (0.0, 0.0);
    }
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
    let (j, ym, ye) = jy_parts(nu, x, want_j, want_y);
    (j, ldexp(ym.value(), ye))
}

/// `(J_ν(x), Y_ν(x))` as `jy`, with `Y_ν = m·2^e` given as `(m, e)` (it may overflow where
/// `sin(νπ)·Y_ν` does not).
pub(crate) fn jy_parts(nu: f64, x: f64, want_j: bool, want_y: bool) -> (f64, Dd, i32) {
    let plain = |(j, y): (f64, f64)| (j, Dd::new(y), 0);
    if nu == 0.0 || nu == 1.0 {
        let one = nu == 1.0;
        if x >= 24.0 {
            return plain(jy01_large(x, one));
        }
        return plain(if one { (j1(x), y1(x)) } else { (j0(x), y0(x)) });
    }
    if hankel_region(nu, x)
        && let Some((p, q)) = hankel(nu, x)
    {
        return plain(jy_hankel(nu, x, p, q));
    }
    // J alone where its series does not cancel.
    let qq = 0.25 * x * x;
    if !want_y && qq <= 0.25 * (nu + 1.0) {
        return plain((series(nu, x, true, false), f64::NAN));
    }
    if nu > NU_MAX {
        let (_, e) = debye_exponents(nu, x);
        return plain(if x < nu && e > 800.0 {
            let j = if qq <= 0.25 * (nu + 1.0) {
                series(nu, x, true, false)
            } else {
                0.0
            };
            (j, f64::NEG_INFINITY)
        } else {
            jy_olver(nu, x)
        });
    }
    if x < 1.0e-150 && nu >= 0.5 {
        let (m, k) = y_tiny(nu, x);
        let j = if want_j {
            series(nu, x, true, false)
        } else {
            f64::NAN
        };
        return (j, m, k);
    }
    let n = (nu + 0.5).floor();
    let mu = nu - n;
    let n = n as u64;
    let oscillating = x > nu;
    let need_y = want_y || !oscillating;
    let (ja, jb, ya, yb) = base(mu, x, want_j && oscillating, need_y);
    if !need_y {
        let (js, _) = up(mu, x, n, ja, jb);
        return plain((js.value(), f64::NAN));
    }
    let (yn, yn1, ye) = up_parts(mu, x, n, ya, yb, false);
    if !want_j {
        return (f64::NAN, yn, ye);
    }
    if oscillating {
        let (js, _) = up(mu, x, n, ja, jb);
        return (js.value(), yn, ye);
    }
    if qq <= 0.25 * (nu + 1.0) {
        return (series(nu, x, true, false), yn, ye);
    }
    // x ≤ ν: with r = J_(ν+1)/J_ν from CF1, the Wronskian J_(ν+1)·Y_ν − J_ν·Y_(ν+1) = 2/(πx) gives
    // J_ν = (2/(πx))/(r·Y_ν − Y_(ν+1)), formed in double-double: near the turning point the two
    // terms cancel to about ν^(−1/3) of each, and an error of the recurrence that is a multiple of
    // J itself drops out (only the scale of Y's run reaches J).
    let r = ratio_j(nu, x, false);
    let den = yn.mul(r).sub(yn1);
    let jv = ldexp(TWO_OVER_PI.div(Dd::new(x)).div(den).value(), -ye);
    (jv, yn, ye)
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

/// Above this `x` the plain `K_n` of an integer order starts from the scaled `K₀`, `K₁` (see
/// `k_scaled`; below it the plain ones are normal and rounded once, and the path is unchanged).
const K_APART_X: f64 = 700.0;

/// `K` up as `k_up` from `(e^x·K_μ, e^x·K_(μ+1))` to `(K_ν, K_(ν+1)/K_ν)`: the recurrence's last
/// value times `e^(−x)` in double-double, its scale apart, rounded once.
fn k_up_exp_neg(mu: f64, x: f64, n: u64, k_mu: f64, k_mu1: f64) -> (Scaled, f64) {
    let (a, b, e) = up_parts(mu, x, n, k_mu, k_mu1, true);
    let (m, k) = exp_parts(-x, 0.0);
    (
        Scaled {
            m: a.mul(m).value(),
            e: e + k,
        },
        b.value() / a.value(),
    )
}

/// `(K_ν(x), K_(ν+1)(x)/K_ν(x))` with `K` scaled by `e^x` when `scaled`, `ν ≥ 0`, `x > 0`;
/// the value as `m·2^e`.
fn k_scaled(nu: f64, x: f64, scaled: bool) -> (Scaled, f64) {
    let is_int = nu == nu.floor() && nu < 1e15;
    if is_int {
        if !scaled && x > K_APART_X {
            // K₀ and K₁ fall into the subnormals from x ≈ 705.5 (and are 0 from 745): started
            // there, the recurrence kept a few bits of K_n, and the Wronskian of `i` gave I_n
            // 6·10⁴ ulps off below its overflow and 0/0 beyond (orders 35 to 49 reach x = 705
            // before Hankel's region). From the scaled K₀, K₁ instead, with e^(−x) apart.
            return k_up_exp_neg(0.0, x, nu as u64, k0(x, true), k1(x, true));
        }
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
    let (m, e) = k_parts(nu, x, scaled);
    ldexp(m.value(), e)
}

/// `K_ν(x)` as `k`, as `m·2^e` given as `(m, e)` (it may overflow where `sin(νπ)·K_ν` does not).
pub(crate) fn k_parts(nu: f64, x: f64, scaled: bool) -> (Dd, i32) {
    if x < 1.0e-150 && nu >= 0.5 {
        return k_tiny(nu, x);
    }
    if nu >= DEBYE_NU {
        // For every x: Hankel's expansion, where it serves (x ≥ 0.6ν²), sums terms that start near
        // 1 and loses up to 1.3 ulps of e^x·K.
        return debye_parts(nu, x, false, scaled);
    }
    if nu == 0.0 {
        return (Dd::new(k0(x, scaled)), 0);
    }
    if nu == 1.0 {
        return (Dd::new(k1(x, scaled)), 0);
    }
    if hankel_region(nu, x)
        && let Some(s) = hankel_ik(nu, x, false)
    {
        let a = PIO2_DD.div(Dd::new(x)).sqrt().mul(s);
        return if scaled {
            (a, 0)
        } else if x > 1.0e6 {
            (Dd::new(0.0), 0)
        } else {
            exp_times(a, -x)
        };
    }
    let (s, _) = k_scaled(nu, x, scaled);
    (Dd::new(s.m), s.e)
}

/// `v·e^t` as `m·2^e`, for `|t| ≤ 2²⁰`.
#[inline]
pub(crate) fn exp_times(v: Dd, t: f64) -> (Dd, i32) {
    let (m, k) = exp_parts(t, 0.0);
    (m.mul(v), k)
}

/// `e^(−x)·K_ν(x)` as `m·2^e` (`m` infinite or 0 where it is beyond the range by far), for the
/// reflection of the scaled `I` at a negative order: `(e^x·K_ν)·e^(−2x)`, the scales apart.
/// From `ν = 50` its exponent `−νη − x` (Debye's), estimated in `f64` first, decides where it is
/// out of range: it is within only about `x = z₁ν`, `z₁ = 0.4477…` the root of `η(z) + z`. From
/// `ν = 2¹⁶` on that exponent comes whole from `D = x − z₁ν` (see `k_exp_neg_debye`): the product
/// took the error `ν·2⁻⁷²` of the scaled exponent's logarithm (2.4 ulps by `ν = 3·10⁷`), and its
/// factors leave `exp_parts`' reach from `ν = 3.3·10⁷` on (`∞` there). Below 50, from `x = 10⁶` on
/// it is 0 (below `e^(−2x)` of `e^x·K_ν`).
pub(crate) fn k_exp_neg_parts(nu: f64, x: f64) -> (Dd, i32) {
    if nu >= NU_Z0 {
        return k_exp_neg_debye(nu, x);
    }
    if nu >= DEBYE_NU {
        let (eta, _) = debye_exponents(nu, x);
        let e = -eta - x;
        if e > 900.0 {
            return (Dd::new(f64::INFINITY), 0);
        }
        if e < -900.0 || e.is_nan() {
            return (Dd::new(0.0), 0);
        }
    } else if x > 1.0e6 {
        return (Dd::new(0.0), 0);
    }
    let (m, e) = k_parts(nu, x, true);
    if !m.hi.is_finite() || m.hi == 0.0 {
        return (m, e);
    }
    let (m2, k2) = exp_parts(-2.0 * x, 0.0);
    (m.mul(m2), e + k2)
}

/// `e^(−x)·K_ν(x)` for `ν ≥ 2¹⁶` by Debye's expansion with its exponent
/// `−νη − x = −ν·(η(z) + z)`, `z = x/ν`, as `−D·P(D/ν)` from `D = x − z₁ν` formed exactly (see
/// `root_exponent`), so that its absolute error, the result's relative error, does not grow with
/// `ν`: `∞` below the band about `z₁ν` where it is in range, 0 above.
fn k_exp_neg_debye(nu: f64, x: f64) -> (Dd, i32) {
    use super::tables_olver::{ETA_Z1, ETA_Z1_DD, Z1_BITS, Z1_HI};
    let f = match root_exponent(nu, x, Z1_HI, &Z1_BITS, &ETA_Z1_DD, &ETA_Z1) {
        Ok(f) => f,
        Err(above) => {
            return if above {
                (Dd::new(0.0), 0)
            } else {
                (Dd::new(f64::INFINITY), 0)
            };
        }
    };
    // r = √(ν² + x²) and p = ν/r as in debye_parts, with ν and x scaled by 2⁻⁶⁰⁰ at the top.
    let sh = if nu.max(x) > pow2(500) { 600 } else { 0 };
    let (v, w) = (nu * pow2(-sh), x * pow2(-sh));
    let (a, ae) = two_prod(v, v);
    let (b, be) = two_prod(w, w);
    let r = Dd { hi: a, lo: ae }.add(Dd { hi: b, lo: be }).sqrt();
    debye_value(nu, r, v / r.hi, f.neg(), false, sh)
}

/// `I_ν(x)` (scaled by `e^(−x)` when `scaled`), `ν ≥ 0`, `x > 0` finite.
pub(crate) fn i(nu: f64, x: f64, scaled: bool) -> f64 {
    if nu == 0.0 {
        return i0(x, scaled);
    }
    if nu == 1.0 {
        return i1(x, scaled);
    }
    // From ν = 50 on Debye's expansion for every x beyond the series (Hankel's loses up to 1.9 ulps
    // of e^(−x)·I there).
    if nu < DEBYE_NU
        && hankel_region(nu, x)
        && let Some(s) = hankel_ik(nu, x, true)
    {
        // √(2πx), as 4·√(πx/8) at the top of the range, where 2πx overflows.
        let c = if x < 1.0e300 {
            PI_DD.mul_f64(2.0 * x).sqrt()
        } else {
            PI_DD.mul_f64(0.125 * x).sqrt().mul_f64(4.0)
        };
        let a = s.div(c);
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
