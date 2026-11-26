//! The elliptic integrals, Carlson's symmetric forms and the Jacobi elliptic functions, all of the
//! parameter `m = k²`.
//!
//! * `K(m)` and `E(m)`: fitted polynomials in `m` on intervals of width 0.05 for `−1 ≤ m < 0.8`,
//!   the logarithmic forms `K = −ln(m₁) K(m₁)/π + B(m₁)` and `E = 1 + m₁(−ln(m₁) S₁(m₁) + S₂(m₁))`
//!   (DLMF 19.12.1–2, `m₁ = 1 − m`) for `m ≥ 0.8`, and the imaginary-modulus transformation
//!   (DLMF 19.7.2) for `m < −1`.
//! * Carlson's integrals by the duplication algorithm (`carlson`), and the incomplete integrals
//!   through them, `φ` reduced by the periodicity, with forms whose terms are all positive where
//!   the textbook forms cancel.
//! * The Jacobi functions by the descending Landen transformation of their values (`jacobi`).

mod carlson;
pub(in crate::math::special) mod eft;
mod jacobi;
mod tables;

use super::Real;
use super::dd::Dd;
use core::f64::consts::{FRAC_PI_2, PI};
use eft::{fast_two_sum, fma, mla, two_prod, two_sum};
use tables::{E_HI, E_LO, E_S1, E_S2, K_B, K_HI, K_LO, K_P};

pub(crate) use jacobi::jacobi_f64;

/// A fitted polynomial `[centre, c0 hi, c0 lo, c1, …]` at `t = x − centre`, as a double-double.
#[inline(always)]
fn poly_dd(r: &[f64], t: f64) -> Dd {
    let n = r.len();
    let mut p = r[n - 1];
    let mut k = n - 2;
    while k >= 3 {
        p = mla(p, t, r[k]);
        k -= 1;
    }
    let v = mla(t, p, r[2]);
    let (hi, lo) = fast_two_sum(r[1], v);
    Dd { hi, lo }
}

/// `P(t) + P′(t) dt` of a fitted polynomial (as `poly_dd`) at the double-double argument
/// `t + dt`, to first order in `dt`.
#[inline(always)]
fn poly_dd_at(r: &[f64], t: f64, dt: f64) -> Dd {
    let v = poly_dd(r, t);
    // P′(t) = Σ k c_k t^(k−1), c_k = r[k + 2] for k ≥ 1.
    let n = r.len();
    let mut q = (n - 3) as f64 * r[n - 1];
    let mut k = n - 2;
    while k >= 3 {
        q = mla(q, t, (k - 2) as f64 * r[k]);
        k -= 1;
    }
    Dd {
        hi: v.hi,
        lo: mla(q, dt, v.lo),
    }
}

/// The table row of `m` in `[−1, 0.8)`.
#[inline(always)]
fn row<const A: usize, const B: usize>(lo: &[[f64; A]; 30], hi: &[[f64; B]; 6], m: f64) -> Dd {
    let i = ((m + 1.0) * 20.0) as usize;
    if i < 30 {
        let r = &lo[i];
        poly_dd(r, m - r[0])
    } else {
        let r = &hi[(i - 30).min(5)];
        poly_dd(r, m - r[0])
    }
}

/// The table row at the double-double `m + dm` in `[−1, 0.8)`, to first order in `dm` (each
/// branch with the row's length known, as `row`).
#[inline(always)]
fn row_at<const A: usize, const B: usize>(
    lo: &[[f64; A]; 30],
    hi: &[[f64; B]; 6],
    m: f64,
    dm: f64,
) -> Dd {
    let i = ((m + 1.0) * 20.0) as usize;
    if i < 30 {
        let r = &lo[i];
        poly_dd_at(r, m - r[0], dm)
    } else {
        let r = &hi[(i - 30).min(5)];
        poly_dd_at(r, m - r[0], dm)
    }
}

/// `ln x` of a positive normal `x` as a double-double: `x = 2^k f` with `f` in `[1/√2, √2)`,
/// `k ln 2` (a double-double constant) plus `ln f`, whose rounding is below `2^−55` absolute
/// (`|ln f| < 0.35`), where that of `ln x` itself is half its ulp: `2^−4` of it for `ln x` near 1,
/// `2^−9` near 37.
#[inline(always)]
fn ln_split(x: f64) -> Dd {
    const LN2_HI: f64 = core::f64::consts::LN_2;
    const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
    let b = x.to_bits();
    let mut k = (b >> 52) as i32 - 1023;
    let mut f = f64::from_bits((b & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if f > core::f64::consts::SQRT_2 {
        f *= 0.5;
        k += 1;
    }
    let kf = k as f64;
    let (p, pe) = two_prod(kf, LN2_HI);
    let (hi, lo) = two_sum(p, f.ln());
    let (hi, lo) = fast_two_sum(hi, lo + mla(kf, LN2_LO, pe));
    Dd { hi, lo }
}

/// `K(m)` for `m₁ = 1 − m` in `(0, 0.2]`, with `l = −ln m₁` a double-double:
/// `l·K(m₁)/π + B(m₁)`.
#[inline(always)]
fn k_log(m1: f64, l: Dd) -> Dd {
    let kp = poly_dd(&K_P, m1 - K_P[0]);
    let b = poly_dd(&K_B, m1 - K_B[0]);
    let (p, pe) = two_prod(l.hi, kp.hi);
    let (hi, lo) = two_sum(p, b.hi);
    let pe = mla(l.lo, kp.hi, pe);
    let (hi, lo) = fast_two_sum(hi, lo + (mla(l.hi, kp.lo, pe) + b.lo));
    Dd { hi, lo }
}

/// `E(m)` for `m₁ = 1 − m` in `[0, 0.2]` exact (`m ≥ 0.8`), with `l = −ln m₁` a double-double:
/// `1 + m₁(l S₁ + S₂)`, the inner sum a double-double too. Rounded once, it would cost up to
/// 0.1 ulp of `E` (its weight `m₁` is up to 0.2), and `E` is held to 0.6 ulp: at
/// `m = 0.8911385566414913`, within 0.1 ulp of a midpoint, the inner sum rounded once (or twice,
/// without a fused multiply-add) took `E` to the wrong side of it.
#[inline(always)]
fn e_log(m1: f64, l: Dd) -> Dd {
    let s1 = poly_dd(&E_S1, m1 - E_S1[0]);
    let s2 = poly_dd(&E_S2, m1 - E_S2[0]);
    let (p, pe) = two_prod(l.hi, s1.hi);
    let (t, te) = two_sum(p, s2.hi);
    let tl = te + (pe + mla(l.hi, s1.lo, mla(l.lo, s1.hi, s2.lo)));
    let (q, qe) = two_prod(m1, t);
    let (hi, lo) = fast_two_sum(1.0, q);
    Dd {
        hi,
        lo: lo + mla(m1, tl, qe),
    }
}

/// `e_log` of `m₁′ = 1/(1 − m)` a double-double (`m < −4`), all of it in double-double: `E`
/// depends on `m₁′` through the factor `m₁′`, and its rounding would cost up to 0.12 ulp.
#[inline(always)]
fn e_log_dd(m1: Dd, l: Dd) -> Dd {
    let s1 = poly_dd(&E_S1, m1.hi - E_S1[0]);
    let s2 = poly_dd(&E_S2, m1.hi - E_S2[0]);
    let t = l.mul(s1).add(s2).mul(m1);
    let (hi, lo) = fast_two_sum(1.0, t.hi);
    Dd { hi, lo: lo + t.lo }
}

/// `m′ = −m/(1 − m)` of `m < −1` as a double-double, and `1 − m` as one.
#[inline(always)]
fn m_prime(m: f64) -> (Dd, Dd) {
    let (dh, dl) = two_sum(1.0, -m);
    let d = Dd { hi: dh, lo: dl };
    (Dd::new(-m).div(d), d)
}

/// `K(m)` of a finite `m < 1`, as a double-double (about 0.6 ulp).
#[inline]
pub(super) fn k_dd(m: f64) -> Dd {
    if m >= 0.8 {
        let m1 = 1.0 - m;
        return k_log(m1, ln_split(m1).neg());
    }
    if m >= -1.0 {
        return row(&K_LO, &K_HI, m);
    }
    // K(m) = K(m′)/√(1 − m), m′ = −m/(1 − m) (a double-double: its rounding would cost 0.35 ulp),
    // 1 − m′ = 1/(1 − m).
    let (mp, d) = m_prime(m);
    let kp = if m > -4.0 {
        row_at(&K_LO, &K_HI, mp.hi, mp.lo)
    } else {
        // ln(1 − m) = ln(d.hi) + d.lo/d.hi.
        let l = ln_split(d.hi).add_f64(d.lo / d.hi);
        k_log(1.0 / d.hi, l)
    };
    kp.div(d.sqrt())
}

/// `K(m + dm)` of `m` in `[−1, 0.8)` and `dm` within a rounding of `m`, to first order in `dm`, as a
/// double-double: for the Jacobi functions at a transformed parameter, which is a double-double.
pub(super) fn k_dd_at(m: f64, dm: f64) -> Dd {
    row_at(&K_LO, &K_HI, m, dm)
}

/// `E(m)` of a finite `m ≤ 1`, as a double-double.
#[inline]
fn e_dd(m: f64) -> Dd {
    if m >= 0.8 {
        let m1 = 1.0 - m;
        if m1 == 0.0 {
            return Dd::new(1.0);
        }
        return e_log(m1, ln_split(m1).neg());
    }
    if m >= -1.0 {
        return row(&E_LO, &E_HI, m);
    }
    // E(m) = √(1 − m) E(m′).
    let (mp, d) = m_prime(m);
    let ep = if m > -4.0 {
        row_at(&E_LO, &E_HI, mp.hi, mp.lo)
    } else {
        // m₁′ = 1/(1 − m) as a double-double: E depends on it through the factor m₁′.
        e_log_dd(Dd::new(1.0).div(d), ln_split(d.hi).add_f64(d.lo / d.hi))
    };
    ep.mul(d.sqrt())
}

/// `K(m)` of an `f64`.
#[inline]
pub(crate) fn ellip_k_f64(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return f64::INFINITY;
    }
    if m == f64::NEG_INFINITY {
        return 0.0;
    }
    k_dd(m).value()
}

/// `E(m)` of an `f64`.
#[inline]
pub(crate) fn ellip_e_f64(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    e_dd(m).value()
}

// ---------------------------------------------------------------------------------------------
// The incomplete integrals.

/// `1 − m s²` for `s, c = sin φ, cos φ`, without cancellation where its terms have one sign.
#[inline(always)]
fn delta2(s: f64, c: f64, m: f64) -> f64 {
    if (0.0..=1.0).contains(&m) {
        // c² + (1 − m) s².
        mla(1.0 - m, s * s, c * c)
    } else if m > 1.0 && c < s.abs() {
        delta2_cos(c, m)
    } else {
        let (s2, e2) = two_prod(s, s);
        fma(-m, s2, 1.0) - m * e2
    }
}

/// `1 − m s² = m c² − (m − 1)` of `m > 1` past `φ = π/4` (so `m < 2` in the real domain, and
/// `1 − m` exact), from the cosine. Next to the edge `Δ = 0` both forms cancel, but the rounding
/// of `c` moves the angle they describe by `ε/tan φ`, that of `s` by `ε tan φ`: `F` and `Π`, whose
/// condition number `1/Δ` is unbounded there, would lose `tan φ` times their `1 + κ` as `m → 1+`
/// (11.6 (1 + κ) of `F` at `(1.558, 1.00016)`) from the sine.
#[cold]
#[inline(never)]
fn delta2_cos(c: f64, m: f64) -> f64 {
    let (c2, e2) = two_prod(c, c);
    fma(m, c2, 1.0 - m) + m * e2
}

/// `φ = jπ + r` with `|r| ≤ π/2`: `(j, sin r, cos r)`, from the sine and cosine of `φ` itself.
#[inline(always)]
fn reduce(phi: f64) -> (f64, f64, f64) {
    let (s, c) = phi.sin_cos();
    if phi.abs() <= FRAC_PI_2 {
        return (0.0, s, c);
    }
    let mut j = (phi / PI).round();
    let odd = j % 2.0 != 0.0;
    let (mut sr, mut cr) = if odd { (-s, -c) } else { (s, c) };
    if cr < 0.0 {
        j += sr.signum();
        sr = -sr;
        cr = -cr;
    }
    (j, sr, cr)
}

/// The real domain of `m > 1`, `|s| ≤ 1/√m` (`s, c = sin φ, cos φ`, `|φ| ≤ π/2`): `(s, c, Δ²)` of
/// `φ` inside it (`Δ² = 1 − m s²`); those of the edge, `(±1/√m, √(1 − 1/m), 0)`, where the computed
/// `Δ²` is negative but within the rounding of `s` (`|Δ²| ≤ 4ε m s²`, where the condition number
/// of `F` and `Π` in `φ` is unbounded); `None` beyond. (The point itself with `Δ² = 0` would not
/// be the edge where `m − 1` is a few ulps: `F(π/2, 1 + 2^−52)` twice its value at the edge.)
#[inline]
fn domain_gt1(s: f64, c: f64, m: f64) -> Option<(f64, f64, f64)> {
    let d2 = delta2(s, c, m);
    if d2 >= 0.0 {
        Some((s, c, d2))
    } else if d2 >= -4.0 * f64::EPSILON * (m * s * s) {
        Some(((1.0 / m).sqrt().copysign(s), ((m - 1.0) / m).sqrt(), 0.0))
    } else {
        None
    }
}

/// `2/π` as a double-double.
const TWO_OVER_PI: Dd = Dd {
    hi: core::f64::consts::FRAC_2_PI,
    lo: -3.935_735_335_036_497e-17,
};

/// The `|φ|` (`2^52`) above which `j = round(φ/π)` is not exact in `f64`: there
/// `F(φ) = 2jK + F(r) = (2K/π) φ + (F(r) − (2K/π) r)` (and so for `E` and `Π`), without `j`.
const PHI_J: f64 = 4.503_599_627_370_496e15;

/// `g φ + (v_r − g r)` for `|φ| > 2^52`, `g = 2C/π` of a finite complete integral `C`, `v_r` the
/// integral over `r = atan2(sr, cr)` in `[−π/2, π/2]`.
#[inline]
fn huge_phi(g: Dd, phi: f64, sr: f64, cr: f64, vr: f64) -> f64 {
    let b = vr - g.value() * sr.atan2(cr);
    if (g.hi * phi).is_finite() {
        return g.mul_f64(phi).add_f64(b).value();
    }
    // At or beyond f64::MAX, where the error term of the product g φ is ∞ − ∞: half the sum,
    // doubled (exactly, so that the result reaches f64::MAX or overflows as the sum rounds).
    let h = g.mul_f64(0.5);
    if (h.hi * phi).is_infinite() {
        return h.hi * phi;
    }
    2.0 * h.mul_f64(phi).add_f64(0.5 * b).value()
}

/// `F` of `s = sin r ≥ 0`, `c = cos r ≥ 0`.
#[inline(always)]
fn f_kernel(s: f64, c: f64, m: f64) -> f64 {
    s * carlson::rf(c * c, delta2(s, c, m), 1.0)
}

/// `E` of `s = sin r ≥ 0`, `c = cos r ≥ 0`, `m < 1` or `c > 0`.
#[inline(always)]
fn e_kernel(s: f64, c: f64, m: f64) -> f64 {
    e_kernel_d(s, c, m, delta2(s, c, m))
}

/// `e_kernel` with `Δ²` given (`m ≤ 1`; `m > 1` goes by `e_gt1`).
#[inline(always)]
fn e_kernel_d(s: f64, c: f64, m: f64, d2: f64) -> f64 {
    if m > 0.0 && m < 1.0 && m * s * s > 0.5 {
        // DLMF 19.25.10: three positive terms, where the usual form below would cancel.
        let m1 = 1.0 - m;
        let c2 = c * c;
        let t1 = m1 * s * carlson::rf(c2, d2, 1.0);
        let t2 = (m * m1 / 3.0) * (s * s * s) * carlson::rd(c2, 1.0, d2);
        let t3 = m * s * c / d2.sqrt();
        t1 + t2 + t3
    } else if m < 0.0 {
        e_neg(s, c, m, d2)
    } else {
        let c2 = c * c;
        s * carlson::rf(c2, d2, 1.0) - (m / 3.0) * (s * s * s) * carlson::rd(c2, d2, 1.0)
    }
}

/// `E` of `m < 0`: `s R_F(c², Δ², 1) + (−m/3) s³ R_D(c², Δ², 1)`, two positive terms, the second
/// the larger for large `−m s²`. Its factor `(−m/3) s³` (formed as `((−m s) s) s`, which does not
/// underflow where `−m s²` is large) is a double-double and the sum is rounded once: rounded step
/// by step, the factor and the products cost up to 3 ulps of `E` (4.56 in all at
/// `(0.553, −43155)`).
#[inline(never)]
fn e_neg(s: f64, c: f64, m: f64, d2: f64) -> f64 {
    let c2 = c * c;
    let t1 = s * carlson::rf(c2, d2, 1.0);
    let rd = carlson::rd(c2, d2, 1.0);
    let (p, pe) = two_prod(-m, s);
    let f = Dd { hi: p, lo: pe }.mul_f64(s).mul_f64(s).div(Dd::new(3.0));
    f.mul_f64(rd).add_f64(t1).value()
}

/// `E(φ, m)` of `m > 1` in the real domain, `s, c = sin φ, cos φ ≥ 0`, by DLMF 19.25.11,
/// `E = s Δ/c + ((m − 1)/3) s³ R_D(Δ², 1, c²)`: two positive terms, where
/// `s R_F(c², Δ², 1) − (m/3) s³ R_D(c², Δ², 1)` cancels next to the edge by up to `F/E`, which
/// grows as `ln(1/(m − 1))` as `m → 1+` (9.75 ulps at `(1.539, 1.001)`). `s²`, `c²` and
/// `Δ² = 1 − m s²` are double-doubles of one angle: of `s` up to `π/4`, of `c` beyond (where
/// `s` may round to 1), at which `E` differs from `E(φ)` by `Δ tan φ` times the rounding of `s`
/// (or `Δ/tan φ` times that of `c`), below half an ulp (`E` is well conditioned at the edge,
/// `Δ → 0`). `s Δ/c` and the factor of `R_D` in double-double, the sum rounded once.
fn e_gt1(s: f64, c: f64, m: f64) -> f64 {
    if s == 0.0 {
        return s;
    }
    let (sq, cq, sd) = if s <= c {
        let (p, e) = two_prod(s, s);
        let sq = Dd { hi: p, lo: e };
        (sq, Dd::new(1.0).sub(sq), Dd::new(s))
    } else {
        let (p, e) = two_prod(c, c);
        let cq = Dd { hi: p, lo: e };
        let sq = Dd::new(1.0).sub(cq);
        (sq, cq, sq.sqrt())
    };
    let d2 = Dd::new(1.0).sub(sq.mul_f64(m));
    // The edge itself (the caller moves a point within the rounding of s of it there): Δ² of the
    // double-doubles may be a little below 0.
    let d2 = if d2.hi > 0.0 { d2 } else { Dd::new(0.0) };
    let t1 = d2.div(cq).sqrt().mul(sd);
    let (mh, ml) = two_sum(m, -1.0);
    let f = Dd { hi: mh, lo: ml }.mul(sq).mul(sd).div(Dd::new(3.0));
    let t2 = f.mul_f64(carlson::rd(d2.hi, 1.0, cq.hi));
    t1.add(t2).value()
}

/// The complementary part `∫₀^ρ dθ/√(1 − m cos²θ)` of `F` with `sin ρ = c`, `cos ρ = s`.
#[inline(always)]
fn f_comp(s: f64, c: f64, m: f64) -> f64 {
    let m1 = 1.0 - m;
    c * carlson::rf(m1 * s * s, delta2(s, c, m), m1)
}

/// The complementary part `∫₀^ρ √(1 − m cos²θ) dθ` of `E` with `sin ρ = c`, `cos ρ = s`, `0 ≤ m < 1`.
#[inline(always)]
fn e_comp(s: f64, c: f64, m: f64) -> f64 {
    let m1 = 1.0 - m;
    let x = m1 * s * s;
    let y = delta2(s, c, m);
    c * m1 * carlson::rf(x, y, m1) + (m1 * m / 3.0) * (c * c * c) * carlson::rd(x, y, m1)
}

/// `ln x` of a double-double `x ≥ 1` (normal): `ln x.hi + x.lo/x.hi`, the first by `ln_split`.
#[inline(always)]
fn ln_dd(x: Dd) -> Dd {
    ln_split(x.hi).add_f64(x.lo / x.hi)
}

/// `F(φ, 1) = atanh(sin φ) = ½ ln((1 + sin φ)/(1 − sin φ))` of `s = |sin φ|` and `c = cos φ > 0`
/// (`|φ| < π/2`), from the one of the two whose rounding moves it least: the rounded sine moves
/// `F` by `δs/c²`, the cosine by `δc/(s c)`, equal at `π/4` (an ulp of `F` there for an ulp of
/// either). Up to `π/4` the quotient of `1 ± s`; beyond, `(1 + √(1 − c²))²/c²`, accurate next to
/// `π/2`, where `1 − s` of the rounded sine is not. In double-double, rounded once: `s R_C(1, c²)`
/// (or `s R_F(c², c², 1)`) adds the error of `R_C` (or of `R_F`), of the product and of `c²` as
/// `1 − s²` to that of the sine, up to 3.10 ulps (2.32) at `|φ|` near 0.45.
#[inline(never)]
fn f_one(s: f64, c: f64) -> f64 {
    if s < 3.725_290_298_461_914e-9 {
        // 2^−28: atanh s = s (1 + s²/3 + …), s²/3 below 0.04 ulp.
        return s;
    }
    let q = if s <= c {
        let (ph, pl) = two_sum(1.0, s);
        let (mh, ml) = two_sum(1.0, -s);
        Dd { hi: ph, lo: pl }.div(Dd { hi: mh, lo: ml })
    } else {
        let (c2, e2) = two_prod(c, c);
        let c2 = Dd { hi: c2, lo: e2 };
        let p = Dd::new(1.0).sub(c2).sqrt().add_f64(1.0); // 1 + √(1 − c²)
        p.mul(p).div(c2)
    };
    // ln q at least 0.35 but for s below 0.17: the rounding of ln f in ln_split is below 2^−56 of
    // F, an eighth of its ulp near π/4.
    0.5 * ln_dd(q).value()
}

/// `F(φ, m)` of `f64`s.
pub(crate) fn ellip_f_f64(phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        let Some((s, c, d2)) = domain_gt1(s, c, m) else {
            return f64::NAN;
        };
        return s.signum() * (s.abs() * carlson::rf(c * c, d2, 1.0));
    }
    if phi.is_infinite() {
        return if m == f64::NEG_INFINITY {
            f64::NAN
        } else {
            phi
        };
    }
    if m == f64::NEG_INFINITY {
        return 0.0 * phi;
    }
    if m == 0.0 {
        // F(φ, 0) = φ, which s R_F(c², 1, 1) of the rounded s, c is not exactly.
        return phi;
    }
    let (j, sr, cr) = reduce(phi);
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if m == 1.0 {
        if j != 0.0 || cr == 0.0 {
            return f64::INFINITY.copysign(phi);
        }
        return sg * f_one(a, cr);
    }
    if j == 0.0 {
        return sg * f_kernel(a, cr, m);
    }
    if phi.abs() > PHI_J {
        return huge_phi(
            k_dd(m).mul(TWO_OVER_PI),
            phi,
            sr,
            cr,
            sg * f_kernel(a, cr, m),
        );
    }
    let k = k_dd(m).value();
    let sj = j.signum();
    if sg != sj && a * a * (1.0 - m).sqrt() > cr * cr {
        // φ = (2j − σ)π/2 + σρ: (2j − σ)K + σ G(ρ), no cancellation.
        return fma(2.0 * j - sj, k, sj * f_comp(a, cr, m));
    }
    fma(2.0 * j, k, sg * f_kernel(a, cr, m))
}

/// `E(φ, m)` of `f64`s.
pub(crate) fn ellip_e_inc_f64(phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        let Some((s, c, _)) = domain_gt1(s, c, m) else {
            return f64::NAN;
        };
        return s.signum() * e_gt1(s.abs(), c, m);
    }
    if phi.is_infinite() || m == f64::NEG_INFINITY {
        return if phi == 0.0 {
            phi
        } else {
            f64::INFINITY.copysign(phi)
        };
    }
    if m == 0.0 {
        // E(φ, 0) = φ.
        return phi;
    }
    let (j, sr, cr) = reduce(phi);
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if phi.abs() > PHI_J {
        let (g, er) = if m == 1.0 {
            (TWO_OVER_PI, sr)
        } else {
            (e_dd(m).mul(TWO_OVER_PI), sg * e_kernel(a, cr, m))
        };
        return huge_phi(g, phi, sr, cr, er);
    }
    if m == 1.0 {
        return fma(2.0, j, sr);
    }
    if j == 0.0 {
        return sg * e_kernel(a, cr, m);
    }
    let e = e_dd(m).value();
    let sj = j.signum();
    if m >= 0.0 && sg != sj && a > cr {
        return fma(2.0 * j - sj, e, sj * e_comp(a, cr, m));
    }
    fma(2.0 * j, e, sg * e_kernel(a, cr, m))
}

/// `Π(n; r, m)` of `s = sin r ≥ 0`, `c = cos r ≥ 0` and `m ≤ 1` (`c > 0` if `m = 1`): the principal
/// value past a pole.
fn pi_kernel(n: f64, s: f64, c: f64, m: f64) -> f64 {
    pi_kernel_d(n, s, c, m, 1.0 - m, delta2(s, c, m), 1.0 - n)
}

/// `pi_kernel` with `m₁ = 1 − m`, `Δ² = 1 − m s²` and `n₁ = 1 − n` given. The reciprocal-modulus
/// transformation of `m > 1` calls it at `μ = 1/m` and `N = n/m`, which are rounded: next to
/// `μ = 1` (`m → 1+`) `1 − μ` would carry that rounding (`ε` absolute, `ε/μ₁` relative: 234 (1 + κ)
/// at `m = 1 + 5e−9` next to the edge), and `1 − N` next to `n = m`, so it passes
/// `μ₁ = (m − 1)/m`, `Δ_β² = 1 − μ sin²β = cos²φ` of the original angle and `(m − n)/m`.
#[inline(always)]
fn pi_kernel_d(n: f64, s: f64, c: f64, m: f64, m1: f64, d2: f64, n1: f64) -> f64 {
    if s == 0.0 {
        return s;
    }
    let c2 = c * c;
    let s3 = s * s * s;
    if n == 0.0 {
        return s * carlson::rf(c2, d2, 1.0);
    }
    if n > 1.0 {
        // A&S 17.7.7: Π(n) = F − Π(m/n) + ln|(Δ + p₁ tan φ)/(Δ − p₁ tan φ)| / (2p₁).
        let nn = m / n;
        let q = (n - 1.0) * (1.0 - nn);
        let p1 = if q.is_finite() {
            q.sqrt()
        } else {
            (n - 1.0).sqrt() * (1.0 - nn).sqrt()
        };
        let lt = pole_log(n, s, c, m, p1 * s, c * d2.sqrt());
        // 1 − N s² = c² + (1 − N) s², 1 − N = (n − m)/n.
        let pn = mla((n - m) / n, s * s, c2);
        let j = if nn == 0.0 {
            0.0
        } else {
            // R_J underflows where −m s² is huge, and N s³ is then large: their product by v·2^e;
            // N s³ without the underflow of s³.
            let (v, e) = carlson::rj_exp(c2, d2, 1.0, pn);
            let ns3 = if s3 > 1e-290 {
                (nn / 3.0) * s3
            } else {
                ((nn * s) * s) * s / 3.0
            };
            carlson::mul_exp(ns3, v, e)
        };
        return lt / p1 - j;
    }
    // 1 − n s² = c² + (1 − n) s²: no cancellation for 0 < n ≤ 1, two positive terms for n < 0.
    let p = mla(n1, s * s, c2);
    if p == 0.0 {
        return f64::INFINITY;
    }
    if n < 0.0 && n < m {
        // Boost's form of A&S 17.7.15 (by R. Romeo): with N = (m − n)/(1 − n),
        // Π(n) = F/(1 − n) + A (N/3) s³ R_J(…, 1 − N s²) + C atan(√(−nN) s c/Δ), all positive.
        let n1 = 1.0 - n;
        let nn = (m - n) / n1;
        let a = (-n / n1) * (m1 / (m - n));
        let dn = (m - n) * n1;
        let cc = if dn.is_finite() {
            (-n / dn).sqrt()
        } else {
            // n < −1.34e154: C ≈ 1/√(−n), the whole value with the atan term.
            (-n / (m - n)).sqrt() / n1.sqrt()
        };
        let p2 = (-n * nn).sqrt();
        let pn = mla(m1 / n1, s * s, c2); // 1 − N s², 1 − N = (1 − m)/(1 − n)
        let rf = carlson::rf(c2, d2, 1.0);
        let rj = carlson::rj(c2, d2, 1.0, pn);
        let at = (p2 * s * c / d2.sqrt()).atan();
        return (s * rf / n1 + (a * nn / 3.0) * s3 * rj) + cc * at;
    }
    if n < 0.0 {
        // m ≤ n < 0: the terms below cancel by up to ln|m| where −n s² is large, and R_J
        // underflows for huge |m|; there the complement, all its terms positive.
        if -n * (s * s) > 4.0 {
            return pi_neg_tail(n, s, c, m);
        }
        // (n/3) s³ without the underflow of s³ for huge |n|.
        let t = ((n * s) * s) * s / 3.0;
        return s * carlson::rf(c2, d2, 1.0) + t * carlson::rj(c2, d2, 1.0, p);
    }
    s * carlson::rf(c2, d2, 1.0) + (n / 3.0) * s3 * carlson::rj(c2, d2, 1.0, p)
}

/// `Π(n; φ, m)` of `m > 1` and a finite `n > m` (the pole `n s² = 1` inside the real domain) for
/// `s = sin φ ≥ 0`, `c = cos φ`, `Δ² = 1 − m s²`: the reciprocal-modulus transformation (DLMF
/// 19.7.4) and A&S 17.7.7 at `N = n/m > 1`, `μ = 1/m`, written in the original variables,
/// `atanh(min(v, 1/v))/p − (m/(3n)) s³ R_J(Δ², c², 1, P)` with `p² = (n − m)(n − 1)/n`,
/// `v = p s/(c Δ)` and `P = 1 − s²_β/n = ((n − 1) + Δ²)/n`. From the rounded `N` and `μ`, the
/// factors `N − 1 = (n − m)/m`, `1 − μ/N = (n − 1)/n`, `N s²_β − 1 = n s² − 1` and `P` lose
/// `ε` absolutely each, which is all of them where `n` and `m` are next to 1 and `φ` next to the
/// pole: 903 (1 + κ) at `(1 + 1.6e−8; 1.5707, 1 + 1.0e−8)`, κ the condition number in `φ` only.
/// Here `n − m` and `n − 1` are exact (next to 1), `n s² − 1` is formed from the smaller of `s`
/// and `c`, and `P` is a sum of two positive terms.
#[inline(never)]
fn pi_gt1_pole(n: f64, s: f64, c: f64, m: f64, d2: f64) -> f64 {
    if s == 0.0 {
        return s;
    }
    let n1 = n - 1.0;
    let p = ((n - m) * (n1 / n)).sqrt();
    let pp = (n1 + d2) / n;
    let lt = pole_atanh(p * s, c * d2.sqrt(), || {
        let f1 = if c < s {
            let (c2, ec) = two_prod(c, c);
            fma(-n, c2, n1) - n * ec
        } else {
            let (s2, es) = two_prod(s, s);
            fma(n, s2, -1.0) + n * es
        };
        (f1, pp)
    });
    lt / p - (m / (3.0 * n)) * (s * s * s) * carlson::rj(d2, c * c, 1.0, pp)
}

/// `Π(n, m)` for `m ≤ n < 0`, where `K + (n/3) R_J(0, m₁, 1, 1 − n)` cancels by up to `ln|m|`:
/// `θ → π/2 − θ` gives `Π(n'', m′)/((1 − n)√(1 − m))` with `n'' = −n/(1 − n)` and
/// `m′ = −m/(1 − m)` in `(0, 1)`, whose terms `K(m′) + (n''/3) R_J` are positive; by
/// homogeneity, `K(m)/(1 − n) + (n''/3) √ρ R_J(0, ρ, 1 − n, 1)` with `ρ = (1 − n)/(1 − m) ≤ 1`,
/// whose `R_J` is of the order of `(1 − n)^(−1/2)` and cannot underflow.
fn pi_complete_neg(n: f64, m: f64) -> f64 {
    // 1 − n, 1 − m, ρ, the factor (n''/3) √ρ = (−n/(3(1 − n))) √ρ and K/(1 − n) in double-double,
    // rounded once at the end: R_J is the only error of more than an ulp.
    let (nh, nl) = two_sum(1.0, -n);
    let n1 = Dd { hi: nh, lo: nl };
    let (mh, ml) = two_sum(1.0, -m);
    let rho = n1.div(Dd { hi: mh, lo: ml });
    let f = Dd::new(-n).div(n1.mul_f64(3.0)).mul(rho.sqrt());
    let rj = carlson::rj(0.0, rho.hi, nh, 1.0);
    f.mul_f64(rj).add(k_dd(m).div(n1)).value()
}

/// `Π(n; r, m)` for `m ≤ n < 0` and `−n s² > 4`: the complete integral less the tail
/// `∫_r^(π/2)` (`pi_tail`), below half the complete integral there.
fn pi_neg_tail(n: f64, s: f64, c: f64, m: f64) -> f64 {
    pi_complete_neg(n, m) - pi_tail(n, s, c, m)
}

/// The tail `∫_r^(π/2) dθ/((1 − n sin²θ)√(1 − m sin²θ))` of `n < 0`, `m < 1`, `s, c = sin r,
/// cos r ≥ 0`: `θ → π/2 − θ` makes it `Π(n''; π/2 − r, m′)/((1 − n)√(1 − m))` (as in
/// `pi_complete_neg`), scaled by homogeneity to
/// `[c √ρ R_F(a, b, 1 − n)/(1 − n) + (n''/3) c³ √ρ R_J(a, b, 1 − n, a + c²)]`,
/// `a = (1 − n) s²`, `b = a + ρc²`, `ρ = (1 − n)/(1 − m)`: all positive.
fn pi_tail(n: f64, s: f64, c: f64, m: f64) -> f64 {
    let n1 = 1.0 - n;
    let rho = n1 / (1.0 - m);
    let sr = rho.sqrt();
    let c2 = c * c;
    let a = n1 * (s * s);
    let b = mla(rho, c2, a);
    c * sr * carlson::rf(a, b, n1) / n1
        + (-n / n1 / 3.0) * (c2 * c) * sr * carlson::rj(a, b, n1, a + c2)
}

/// `atanh(min/max)` of `num = p₁ s`, `den = c Δ`, the logarithmic term
/// `½ ln|(Δ + p₁ tan φ)/(Δ − p₁ tan φ)|` of A&S 17.7.7 (`n > 1`). Next to the pole `n s² = 1` the
/// quotient `v` is near 1 and `1 − v` of the rounded quotient is not the true one (it may even be
/// 0 far from the pole, where `n s² ≫ 1` and `m ≪ 0`): there `atanh v = ln(1 + v) − ½ ln(1 − v²)`
/// with `1 − v² = |num² − den²|/max²` from the identity `num² − den² = (n s² − 1)(n − m s²)/n`.
fn pole_log(n: f64, s: f64, c: f64, m: f64, num: f64, den: f64) -> f64 {
    pole_atanh(num, den, || {
        // n s² − 1 and (n − m s²)/n = 1 − N s², N = m/n, from the smaller of s and c, whose
        // square is accurate where 1 − s² or 1 − c² of the other is not: with c, (n − 1) − n c²
        // and (1 − N) + N c² (both terms positive for 0 ≤ N < 1; 1 + |N| s² for N < 0).
        let nn = m / n;
        if c < s {
            let (c2, ec) = two_prod(c, c);
            let f1 = fma(-n, c2, n - 1.0) - n * ec;
            let f2 = if nn >= 0.0 {
                mla(nn, c2, 1.0 - nn)
            } else {
                mla(-nn, s * s, 1.0)
            };
            (f1, f2)
        } else {
            let (s2, es) = two_prod(s, s);
            (fma(n, s2, -1.0) + n * es, mla(-nn, s2, 1.0))
        }
    })
}

/// `atanh(min/max)` of `num`, `den` as in `pole_log`, with the factors of
/// `num² − den² = f₁ f₂` (`f₁ = n s² − 1`, zero at the pole, and `f₂ = (n − m s²)/n`) from
/// `factors`, which is called only where the quotient is above ½.
#[inline(always)]
fn pole_atanh(num: f64, den: f64, factors: impl FnOnce() -> (f64, f64)) -> f64 {
    let (lo, hi) = if num < den { (num, den) } else { (den, num) };
    let v = lo / hi;
    if v <= 0.5 {
        return 0.5 * (2.0 * v / (1.0 - v)).ln_1p();
    }
    let (f1, f2) = factors();
    if f1 == 0.0 {
        return f64::INFINITY;
    }
    let w = (f1.abs() / hi) * (f2.abs() / hi); // 1 − v²
    if w > 0.0 && w.is_finite() {
        return v.ln_1p() - 0.5 * w.ln();
    }
    // Beyond the range of f64: the logarithms of the factors.
    v.ln_1p() - 0.5 * (f1.abs().ln() + f2.abs().ln() - 2.0 * hi.ln())
}

/// `Π(n, m)` of `f64`s (the principal value for `n > 1`).
pub(crate) fn ellip_pi_f64(n: f64, m: f64) -> f64 {
    if n.is_nan() || m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if n == 1.0 {
        return f64::INFINITY;
    }
    if m == 1.0 {
        return if n < 1.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if n.is_infinite() || m == f64::NEG_INFINITY {
        return 0.0;
    }
    let m1 = 1.0 - m;
    if n == 0.0 {
        return k_dd(m).value();
    }
    if n > 1.0 {
        // DLMF 19.6.5: Π(n, m) = K − Π(m/n, m) = −(N/3) R_J(0, m₁, 1, 1 − N), N = m/n; R_J
        // underflows for huge −m, where N is huge: their product by v·2^e.
        let nn = m / n;
        if nn == 0.0 {
            return 0.0;
        }
        let (v, e) = carlson::rj_exp(0.0, m1, 1.0, (n - m) / n);
        return carlson::mul_exp(-(nn / 3.0), v, e);
    }
    if n < 0.0 && m <= n {
        return pi_complete_neg(n, m);
    }
    let k = k_dd(m).value();
    if n < 0.0 {
        // n < m: the positive form of A&S 17.7.15, (A N/3) R_J(0, m₁, 1, p), p = m₁/(1 − n).
        let n1 = 1.0 - n;
        let p = m1 / n1;
        if p < 1e-270 {
            // n ≪ 0 and m → 1: p is subnormal or near it, and A ≈ p: A N R_J = (−n/(1 − n)) p R_J
            // with p R_J(0, m₁, 1, p) = λp √λ R_J(0, λm₁, λ, λp), λ = 2^400, λp formed directly.
            const L: f64 = 2.582_249_878_086_908_6e120; // 2^400
            const RL: f64 = 1.606_938_044_258_990_3e60; // 2^200
            let lp = (L * m1) / n1;
            let t = lp * carlson::rj(0.0, L * m1, L, lp) * RL;
            return k / n1 + (-n / n1 / 3.0) * t;
        }
        let nn = (m - n) / n1;
        let a = (-n / n1) * (m1 / (m - n));
        return k / n1 + (a * nn / 3.0) * carlson::rj(0.0, m1, 1.0, p);
    }
    k + (n / 3.0) * carlson::rj(0.0, m1, 1.0, 1.0 - n)
}

/// `Π(n; φ, m)` of `f64`s.
pub(crate) fn ellip_pi_inc_f64(n: f64, phi: f64, m: f64) -> f64 {
    if n.is_nan() || phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        // DLMF 19.7.4: Π(n; φ, m) = Π(n/m; β, 1/m)/√m with sin β = √m sin φ, cos β = Δ.
        if phi.is_infinite() || phi.abs() > FRAC_PI_2 {
            return f64::NAN;
        }
        let (s, c) = phi.sin_cos();
        let Some((s, c, d2)) = domain_gt1(s, c, m) else {
            return f64::NAN;
        };
        if n == f64::NEG_INFINITY {
            // The integrand vanishes but at θ = 0: Π → 0, odd in φ (as for n = +∞).
            return 0.0f64.copysign(phi);
        }
        let v = if n > m && n.is_finite() {
            pi_gt1_pole(n, s.abs(), c, m, d2)
        } else {
            let r = m.sqrt();
            let sb = (r * s.abs()).min(1.0);
            // At μ = 1/m: μ₁ = (m − 1)/m (m − 1 exact up to 2), 1 − μ sin²β = cos²φ and
            // 1 − N = (m − n)/m (N = n/m, m − n exact next to m), not from the rounded μ and N:
            // 1 − N s_β² = Δ² + (m − n) s² is small where both terms are.
            let n1 = (m - n) / m;
            let n1 = if n1.is_finite() { n1 } else { 1.0 - n / m };
            pi_kernel_d(n / m, sb, d2.sqrt(), 1.0 / m, (m - 1.0) / m, c * c, n1) / r
        };
        return if phi.is_sign_negative() { -v } else { v };
    }
    if phi.is_infinite() || m == f64::NEG_INFINITY {
        if m == f64::NEG_INFINITY {
            return 0.0 * phi;
        }
        let p = ellip_pi_f64(n, m);
        return if p == 0.0 { f64::NAN } else { p * phi };
    }
    if n == f64::NEG_INFINITY {
        // The integrand vanishes but at θ = 0: Π → 0, odd in φ (as for n = +∞).
        return 0.0f64.copysign(phi);
    }
    let (j, sr, cr) = reduce(phi);
    let sg = if sr.is_sign_negative() { -1.0 } else { 1.0 };
    let a = sr.abs();
    if m == 1.0 && (j != 0.0 || cr == 0.0) {
        // Past ±π/2: the divergence of 2jΠ(n, 1), +∞ for n ≤ 1 and −∞ for n > 1 (as the
        // complete integral), with the sign of φ.
        let s = if n > 1.0 { -1.0 } else { 1.0 };
        return f64::INFINITY.copysign(s * phi);
    }
    let v = sg * pi_kernel(n, a, cr, m);
    if j == 0.0 {
        return v;
    }
    if phi.abs() > PHI_J {
        let pc = ellip_pi_f64(n, m);
        if pc.is_infinite() {
            // Π(1, m) = ∞: the divergence 2jΠ(n, m), with the sign of φ (as the fma below).
            return pc * phi;
        }
        return huge_phi(TWO_OVER_PI.mul_f64(pc), phi, sr, cr, v);
    }
    let pc = ellip_pi_f64(n, m);
    let sj = j.signum();
    if n < 0.0 && sg != sj && 2.0 * v.abs() > pc {
        // φ = jπ + r with r of the other sign and past half of Π: 2jΠ + Π(r) = (2j − σ)Π + σT, T
        // the tail ∫_|r|^(π/2), of one sign, where 2|j|Π − Π(|r|) cancels (to Π for j = ±1, and
        // κ ≈ 0 there for n ≪ −1: 7.39 ulps at (−7.7e14; −2.96, −2.54)).
        return fma(2.0 * j - sj, pc, sj * pi_tail(n, a, cr, m));
    }
    fma(2.0 * j, pc, v)
}

// ---------------------------------------------------------------------------------------------
// Carlson's integrals of `f64`s.

/// `R_F(x, y, z)` of `f64`s.
pub(crate) fn carlson_rf_f64(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return 0.0;
    }
    let zeros = (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8;
    if zeros >= 2 {
        return f64::INFINITY;
    }
    carlson::rf(x, y, z)
}

/// `R_D(x, y, z)` of `f64`s.
pub(crate) fn carlson_rd_f64(x: f64, y: f64, z: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    if z == 0.0 || (x == 0.0 && y == 0.0) {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return 0.0;
    }
    carlson::rd(x, y, z)
}

/// `R_J(x, y, z, p)` of `f64`s.
pub(crate) fn carlson_rj_f64(x: f64, y: f64, z: f64, p: f64) -> f64 {
    if x.is_nan() || y.is_nan() || z.is_nan() || p.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    let zeros = (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8;
    if p == 0.0 || zeros >= 2 {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() || p.is_infinite() {
        return 0.0;
    }
    if p < 0.0 {
        return carlson::rj_pv_or_big(x, y, z, p);
    }
    carlson::rj(x, y, z, p)
}

/// `R_C(x, y)` of `f64`s.
#[inline]
pub(crate) fn carlson_rc_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() || y.is_infinite() {
        return 0.0;
    }
    carlson::rc(x, y)
}

// ---------------------------------------------------------------------------------------------
// The public functions.

#[inline(always)]
fn via2<T: Real>(a: T, b: T, f: impl FnOnce(f64, f64) -> f64) -> T {
    T::from_f64(f(a.to_f64(), b.to_f64()))
}

#[inline(always)]
fn via3<T: Real>(a: T, b: T, c: T, f: impl FnOnce(f64, f64, f64) -> f64) -> T {
    T::from_f64(f(a.to_f64(), b.to_f64(), c.to_f64()))
}

/// The complete elliptic integral of the first kind `K(m) = ∫₀^(π/2) dθ / √(1 − m sin²θ)`, for
/// `m < 1` (`∞` at 1).
pub fn ellip_k<T: Real>(m: T) -> T {
    super::via(m, ellip_k_f64)
}

/// The complete elliptic integral of the second kind `E(m) = ∫₀^(π/2) √(1 − m sin²θ) dθ`, for
/// `m ≤ 1`.
pub fn ellip_e<T: Real>(m: T) -> T {
    super::via(m, ellip_e_f64)
}

/// The complete elliptic integral of the third kind
/// `Π(n, m) = ∫₀^(π/2) dθ / ((1 − n sin²θ) √(1 − m sin²θ))` (its Cauchy principal value for
/// `n > 1`).
pub fn ellip_pi<T: Real>(n: T, m: T) -> T {
    via2(n, m, ellip_pi_f64)
}

/// The incomplete elliptic integral of the first kind `F(φ, m) = ∫₀^φ dθ / √(1 − m sin²θ)`.
pub fn ellip_f<T: Real>(phi: T, m: T) -> T {
    via2(phi, m, ellip_f_f64)
}

/// The incomplete elliptic integral of the second kind `E(φ, m) = ∫₀^φ √(1 − m sin²θ) dθ`.
pub fn ellip_e_inc<T: Real>(phi: T, m: T) -> T {
    via2(phi, m, ellip_e_inc_f64)
}

/// The incomplete elliptic integral of the third kind
/// `Π(n; φ, m) = ∫₀^φ dθ / ((1 − n sin²θ) √(1 − m sin²θ))`.
pub fn ellip_pi_inc<T: Real>(n: T, phi: T, m: T) -> T {
    via3(n, phi, m, ellip_pi_inc_f64)
}

/// Carlson's symmetric integral `R_F(x, y, z) = ½ ∫₀^∞ dt / √((t + x)(t + y)(t + z))`.
pub fn carlson_rf<T: Real>(x: T, y: T, z: T) -> T {
    via3(x, y, z, carlson_rf_f64)
}

/// Carlson's symmetric integral `R_D(x, y, z) = (3/2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))`.
pub fn carlson_rd<T: Real>(x: T, y: T, z: T) -> T {
    via3(x, y, z, carlson_rd_f64)
}

/// Carlson's symmetric integral
/// `R_J(x, y, z, p) = (3/2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))` (its Cauchy principal
/// value for `p < 0`).
pub fn carlson_rj<T: Real>(x: T, y: T, z: T, p: T) -> T {
    T::from_f64(carlson_rj_f64(
        x.to_f64(),
        y.to_f64(),
        z.to_f64(),
        p.to_f64(),
    ))
}

/// Carlson's degenerate integral `R_C(x, y) = R_F(x, y, y)` (its Cauchy principal value for
/// `y < 0`).
pub fn carlson_rc<T: Real>(x: T, y: T) -> T {
    via2(x, y, carlson_rc_f64)
}

/// The Jacobi elliptic functions `(sn(u, m), cn(u, m), dn(u, m))`.
pub fn jacobi_elliptic<T: Real>(u: T, m: T) -> (T, T, T) {
    let (s, c, d) = jacobi_f64(u.to_f64(), m.to_f64());
    (T::from_f64(s), T::from_f64(c), T::from_f64(d))
}
