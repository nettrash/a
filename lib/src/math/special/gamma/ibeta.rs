//! The regularised incomplete beta function `I_x(a, b)`, its complement and their inverses.
//!
//! The methods and their regions are those of DiDonato and Morris (ACM TOMS 708, 1992), in the
//! arrangement of Boost.Math's `ibeta_imp`: the power series (BPSER) where `x` is small or a
//! parameter is below 1, the continued fraction (BFRAC) for both parameters large, the
//! recurrence on `a` (BUP) and the expansion in incomplete gamma functions (BGRAT) for a small
//! `b` beside a large `a`, always on the side (`x` or `y = 1 − x`, `I` or `1 − I`) that does not
//! cancel. Unlike those, the power terms `x^a y^b / B(a, b)` are computed in double-double
//! from exact logarithms: directly for small parameters, and for large ones in the Stirling
//! ratio form `√(ab/(2π(a + b))) · e^(a·φ(r) + b·φ(s) − ΔS)`, `φ(r) = ln(1 + r) − r`, whose
//! exponent has no cancellation.

use super::super::dd::{Dd, two_prod, two_sum};
use super::ddm::*;
use super::igamma::{erfcx_pos_dd, gamma_p_f64, gamma_p_inv_f64, gamma_q_f64, gamma_q_inv_f64};
use super::real::*;
use super::tab::{LN_SQRT_2PI_HI, LN_SQRT_2PI_LO, STIRLING, STIRLING_C0_LO};

/// The relative size below which a series stops.
const EPS: f64 = 1.1e-17;

/// The distance from 1 of a continued fraction's last factor where it stops (its factors are
/// rounded: they reach 1 only to an ulp or so).
const CF_EPS: f64 = 2.3e-16;

/// `1/(2π)` as a double-double.
const INV_2PI: Dd = Dd {
    hi: 0.159_154_943_091_895_35,
    lo: -9.839_338_337_659_239e-18,
};

/// `ln √(2π)`.
const LN_SQRT_2PI: Dd = Dd {
    hi: LN_SQRT_2PI_HI,
    lo: LN_SQRT_2PI_LO,
};

/// A value `e^e · m`, kept apart until the end so that neither part over- or underflows.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    e: Dd,
    m: Dd,
}

impl Scaled {
    /// The value as a double-double (exactly zero below the subnormals' reach of `f64`; a
    /// value that underflows is returned rounded, its low part zero).
    #[inline]
    fn value(self) -> Dd {
        if self.m.hi == 0.0 || self.e.hi < -800.0 {
            return Dd::new(0.0);
        }
        let (d, k) = exp_parts(self.e);
        let p = d.mul(self.m);
        if k > -960 && k < 1000 {
            let s = pow2i(k);
            Dd {
                hi: p.hi * s,
                lo: p.lo * s,
            }
        } else {
            Dd::new(scale(p, k))
        }
    }
}

/// A computed `I` (or its complement): `s₀ + e^e·m`.
#[derive(Clone, Copy, Debug)]
struct Val {
    s0: Dd,
    sc: Scaled,
}

impl Val {
    /// `e^e·m` alone.
    #[inline]
    fn of(sc: Scaled) -> Val {
        Val {
            s0: Dd::new(0.0),
            sc,
        }
    }

    /// The constant `v`.
    #[inline]
    fn constant(v: Dd) -> Val {
        Val {
            s0: v,
            sc: Scaled {
                e: Dd::new(0.0),
                m: Dd::new(0.0),
            },
        }
    }

    /// The value as a double-double.
    #[inline]
    fn value(self) -> Dd {
        if self.sc.m.hi == 0.0 {
            return self.s0;
        }
        self.s0.add(self.sc.value())
    }

    /// The natural logarithm of the value (of a positive value), without its underflow.
    #[inline]
    fn ln(self) -> Dd {
        if self.s0.hi == 0.0 && self.sc.m.hi > 0.0 {
            return self.sc.e.add(ln_of_dd(self.sc.m));
        }
        ln_of_dd(self.value())
    }
}

/// `x/n` of a double-double by an `f64`.
trait DivF64 {
    fn div_f64(self, n: f64) -> Dd;
}

impl DivF64 for Dd {
    #[inline(always)]
    fn div_f64(self, n: f64) -> Dd {
        let q = self.hi / n;
        let r = rem(q, n, self.hi) + self.lo;
        let (hi, lo) = two_sum(q, r / n);
        Dd { hi, lo }
    }
}

/// `ln v` to about `10⁻⁶`, for a finite `v > 0` (normal or subnormal): the exponent and
/// `2 atanh((m − 1)/(m + 1))` of the significand `m ∈ [√½, √2)` to `s⁵`.
#[inline]
fn ln_crude(v: f64) -> f64 {
    let (v, k0) = if v < f64::MIN_POSITIVE {
        (v * 18_014_398_509_481_984.0, -54.0)
    } else {
        (v, 0.0)
    };
    let bits = v.to_bits();
    let mut e = ((bits >> 52) as i64 - 1023) as f64 + k0;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1.0;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    e * core::f64::consts::LN_2 + 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * 0.2))
}

/// `ψ(c)` to about 1% (the derivative that carries the low part of `c = a + b`).
#[inline]
fn psi_rough(c: f64) -> f64 {
    let (d, shift) = if c < 1.0 {
        (c + 1.0, 1.0 / c)
    } else {
        (c, 0.0)
    };
    ln_crude(d) - 0.5 / d - 1.0 / (12.0 * d * d) - shift
}

/// `ln v` roughly (for the choice of a method), `−∞` at 0.
#[inline]
fn ln_rough(v: f64) -> f64 {
    if v > 0.0 {
        ln_crude(v)
    } else {
        f64::NEG_INFINITY
    }
}

/// `ln Γ(c)` of the double-double `c > 0`.
#[inline]
pub(super) fn lgamma_of_dd(c: Dd) -> Dd {
    let l = lgamma_pos_dd(c.hi);
    if c.lo == 0.0 {
        l
    } else {
        l.add_f64(c.lo * psi_rough(c.hi))
    }
}

/// `S(z) = ln Γ(z) − ((z − 1/2) ln z − z + ln √(2π))`, the Stirling remainder, as a
/// double-double.
#[inline]
pub(super) fn stirling_rem(z: f64) -> Dd {
    if z >= STIRLING_X {
        let r = 1.0 / z;
        let u = r * r;
        let [c0, rest @ ..] = STIRLING;
        return Dd::new(r * (fma(u, poly(u, &rest), c0) + STIRLING_C0_LO));
    }
    let l = ln_dd(z);
    // (z − 1/2) ln z as z ln z − (ln z)/2: z − 1/2 is not exact for a small z.
    let main = l
        .mul_f64(z)
        .sub(l.mul_f64(0.5))
        .add_f64(-z)
        .add(LN_SQRT_2PI);
    lgamma_pos_dd(z).sub(main)
}

/// The coefficients `2/(2k + 1)`, `k = 2, 3, …`, of the tail of `φ` in `w²`.
static PHI_TAIL: [f64; 9] = [
    2.0 / 5.0,
    2.0 / 7.0,
    2.0 / 9.0,
    2.0 / 11.0,
    2.0 / 13.0,
    2.0 / 15.0,
    2.0 / 17.0,
    2.0 / 19.0,
    2.0 / 21.0,
];

/// `2/3` as a double-double.
const TWO_THIRDS: Dd = Dd {
    hi: 0.666_666_666_666_666_6,
    lo: 3.700_743_415_417_188e-17,
};

/// `φ(r) = ln(1 + r) − r` of `q = 1 + r` and `r` (both given, each accurate), as a
/// double-double with a relative error of about `2⁻⁶⁶`.
#[inline]
pub(super) fn phi(q: Dd, r: Dd) -> Dd {
    if r.hi.abs() < 0.125 {
        // ln(1 + r) = 2 atanh(w), w = r/(2 + r): φ = −r·w + (2/3)w³ + w⁵ T(w²).
        let w = r.div(r.add_f64(2.0));
        let w2 = w.mul(w);
        let w3 = w2.mul(w);
        let t = w3.hi * w2.hi * poly(w2.hi, &PHI_TAIL);
        return r.mul(w).neg().add(w3.mul(TWO_THIRDS)).add_f64(t);
    }
    ln_of_dd(q).sub(r)
}

/// `ln Γ(a) − ln Γ(a + b)` for `a ≥ 8`, as a double-double, without the cancellation of the
/// two logarithms.
#[inline]
pub(super) fn lgamma_delta(a: f64, b: f64) -> Dd {
    let (ch, cl) = two_sum(a, b);
    // ln(c/a) = ln(1 + b/a).
    let l1 = ln_of_dd(Dd::new(1.0).add(Dd::new(b).div(Dd::new(a))));
    let la = ln_dd(a);
    let sa = stirling_rem(a);
    // S′(c) ≈ −1/(12c²) carries the low part of c.
    let sc = stirling_rem(ch).add_f64(-cl / (12.0 * ch * ch));
    // −(a − 1/2) L₁ − b(ln a + L₁) + b + S(a) − S(c).
    l1.mul_f64(a - 0.5)
        .neg()
        .sub(la.add(l1).mul_f64(b))
        .add_f64(b)
        .add(sa)
        .sub(sc)
}

/// `1/Γ(x)` for `0 ≤ x < 16`, as a double-double: `x·(1/Γ(1 + x))` to ½, else
/// `(1/Γ(1 + t))/((t + 1)(t + 2)…(t + n − 1))` with `x = n + t`, `|t| ≤ ½`, the product compensated.
#[inline]
pub(super) fn rgamma_dd(x: f64) -> Dd {
    if x <= 0.5 {
        return rgamma1p(x).mul_f64(x);
    }
    let n = round_int(x);
    let t = x - n;
    let d = rgamma1p(t);
    let (mut p, mut e) = (1.0, 0.0);
    let mut k = 1.0;
    while k < n {
        let f = t + k;
        let q = p * f;
        e = fma(e, f, prod_err(p, f, q));
        p = q;
        k += 1.0;
    }
    if n <= 1.0 {
        d
    } else {
        d.div(Dd { hi: p, lo: e })
    }
}

/// The parts of the power terms that depend on `a` and `b` alone (symmetric in them), the
/// factor `e^ds·m`: for `max(a, b) < 8` `m = 1/B(a, b)` (`ds = −ln B(a, b)` instead where `1/B`
/// would underflow), else `m = √(ab/(2π(a + b)))` and `ds = S(a + b) − S(a) − S(b)`.
#[derive(Clone, Copy, Debug)]
struct Pre {
    small: bool,
    m: Dd,
    ds: Dd,
}

impl Pre {
    #[inline]
    fn new(a: f64, b: f64) -> Pre {
        let (ch, cl) = two_sum(a, b);
        let c = Dd { hi: ch, lo: cl };
        if a.max(b) < STIRLING_X && a.min(b) < 1e-100 {
            // 1/Γ(a)Γ(b) would underflow: in logarithms.
            return Pre {
                small: true,
                m: Dd::new(1.0),
                ds: lgamma_of_dd(c).sub(lgamma_pos_dd(a)).sub(lgamma_pos_dd(b)),
            };
        }
        if a.max(b) < STIRLING_X {
            // 1/B(a, b) = Γ(c)/(Γ(a)Γ(b)), with 1/Γ(hi + lo) = (1/Γ(hi))(1 − lo ψ(hi)).
            let mut rc = rgamma_dd(ch);
            if cl != 0.0 {
                rc = rc.add(rc.mul_f64(-cl * psi_rough(ch)));
            }
            let m = rgamma_dd(a).mul(rgamma_dd(b)).div(rc);
            return Pre {
                small: true,
                m,
                ds: Dd::new(0.0),
            };
        }
        Pre {
            small: false,
            m: Dd::new(a).div(c).mul_f64(b).mul(INV_2PI).sqrt(),
            ds: stirling_delta(a, b),
        }
    }
}

/// `x^a y^b / B(a, b)` (or `x^a / B(a, b)` without `y`), for the arguments as double-doubles
/// with `x + y = 1`.
fn power_terms(a: f64, b: f64, x: Dd, y: Dd, with_y: bool, pre: &Pre) -> Scaled {
    if pre.small {
        // e^(a ln x + b ln y)/B(a, b).
        let mut e = ln_of_dd(x).mul_f64(a);
        if with_y {
            e = e.add(ln_of_dd(y).mul_f64(b));
        }
        if pre.ds.hi != 0.0 {
            e = e.add(pre.ds);
        }
        return Scaled { e, m: pre.m };
    }
    // √(ab/(2πc)) · (xc/a)^a (yc/b)^b · e^(S(c) − S(a) − S(b)).
    if with_y {
        let e = power_exponent(a, b, x, y);
        return Scaled {
            e: e.add(pre.ds),
            m: pre.m,
        };
    }
    let (ch, cl) = two_sum(a, b);
    let c = Dd { hi: ch, lo: cl };
    let qa = x.mul(c).div(Dd::new(a));
    let e = ln_of_dd(qa)
        .mul_f64(a)
        .add(ln_of_dd(c.div(Dd::new(b))).mul_f64(b));
    Scaled {
        e: e.add(pre.ds),
        m: pre.m,
    }
}

/// `S(a + b) − S(a) − S(b)`, the Stirling corrections of `1/B(a, b)`.
#[inline]
pub(super) fn stirling_delta(a: f64, b: f64) -> Dd {
    let (ch, cl) = two_sum(a, b);
    // S′(c) ≈ −1/(12c²) carries the low part of c.
    stirling_rem(ch)
        .add_f64(-cl / (12.0 * ch * ch))
        .sub(stirling_rem(a))
        .sub(stirling_rem(b))
}

/// The exponent of the ratio form `a φ(r) + b φ(s)` with `r = (xc − a)/a`, `s = −(xc − a)/b`,
/// `c = a + b` (so that `a r + b s = 0` and neither term cancels the other).
fn power_exponent(a: f64, b: f64, x: Dd, _y: Dd) -> Dd {
    let (ch, cl) = two_sum(a, b);
    let c = Dd { hi: ch, lo: cl };
    let d = x.mul(c).add_f64(-a);
    // x c/a = 1 + r and y c/b = 1 − d/b: no further division.
    let r = d.div_f64(a);
    let s = d.neg().div_f64(b);
    phi(r.add_f64(1.0), r)
        .mul_f64(a)
        .add(phi(s.add_f64(1.0), s).mul_f64(b))
}

/// `I_x(a, b)` by the power series (BPSER): `x^a/B(a, b) · Σ (1 − b)ₙ xⁿ/(n! (a + n))`.
fn series(a: f64, b: f64, x: Dd, y: Dd, pre: &Pre) -> Scaled {
    let p = power_terms(a, b, x, y, false, pre);
    // The first three terms in double-double (the sum may be complemented, and they may cancel;
    // six were no more accurate on the survey);
    // the rest in f64 (their rounding errors no longer matter), summed without error.
    // The sum times a (1/a alone overflows for a subnormal a): 1 + a Σ tₙ/(a + n).
    let mut t = Dd::new(a);
    let mut s = Dd::new(1.0);
    let mut n = 1.0;
    while n <= 3.0 {
        t = t.mul(Dd::new(n).add_f64(-b).mul(x)).div_f64(n);
        let term = t.div(Dd::new(a).add_f64(n));
        s = s.add(term);
        if term.hi.abs() <= 3.5e-18 * s.hi.abs() {
            break;
        }
        n += 1.0;
    }
    if n > 3.0 {
        let xv = x.hi;
        // The terms by their ratio, one division each.
        let mut term = t.hi / (a + (n - 1.0));
        let (mut sh, mut sl) = (s.hi, s.lo);
        while n < 10_000.0 {
            term *= (n - b) * xv * (a + n - 1.0) / (n * (a + n));
            let (h, e) = two_sum(sh, term);
            sh = h;
            sl += e;
            if term.abs() <= 3.5e-18 * sh.abs() {
                break;
            }
            n += 1.0;
        }
        s = Dd { hi: sh, lo: sl };
    }
    Scaled {
        e: p.e.sub(ln_dd(a)),
        m: p.m.mul(s),
    }
}

/// `I_x(a, b)` by the continued fraction (BFRAC, Boost's second form) `b₀ + a₁/(b₁ + a₂/(b₂ + …))`,
/// in the equivalent form without divisions in its coefficients: `Bₘ = cₘbₘ`, `Aₘ = cₘ₋₁cₘaₘ` with
/// `cₘ = (a + 2m − 1)(a + 2m + 1)` (`c₀ = 1`). Forwards by the three-term recurrence of the
/// convergents (fused multiply-adds only) to find the depth, then backwards in projective form
/// `(N, D) ← (AₖD, BₖD + N)`, which does not accumulate the forward recurrence's rounding errors;
/// the last two steps and `b₀` in double-double (with exact coefficients the rounding of the
/// others no longer matters: five were needed with the divided ones).
fn fraction(a: f64, b: f64, x: Dd, y: Dd, pre: &Pre) -> Scaled {
    let p = power_terms(a, b, x, y, true, pre);
    let xv = x.hi;
    // λ = a y − b x = a − (a + b) x, which cancels near the mean: in double-double.
    let lam_dd = y.mul_f64(a).sub(x.mul_f64(b));
    let lam = lam_dd.value();
    let c = a + b;
    // (Aₘ, Bₘ) in f64, m ≥ 1.
    let coef = |m: f64| -> (f64, f64) {
        let den = a + 2.0 * m - 1.0;
        let den2 = a + 2.0 * m + 1.0;
        // ((c + m − 1)x)((b − m)x) first: (a + b)x and bx stay in range where b is huge.
        let num = m * (a + m - 1.0) * (((c + m - 1.0) * xv) * ((b - m) * xv));
        let am = if m == 1.0 {
            num * (a + 3.0) / (a + 1.0)
        } else {
            num * ((a + 2.0 * m - 3.0) * den2)
        };
        let bm =
            m * den * den2 + m * (b - m) * xv * den2 + (a + m) * (lam + 1.0 + m * (2.0 - xv)) * den;
        (am, bm)
    };
    // The same in double-double (every sum of a parameter and an integer exact), the constants
    // hoisted.
    let ad = Dd::new(a);
    let cd = ad.add_f64(b);
    let lam1 = lam_dd.add_f64(1.0);
    let two_x = Dd::new(2.0).sub(x);
    let coef_dd = |m: f64| -> (Dd, Dd) {
        let den = ad.add_f64(2.0 * m - 1.0);
        let den2 = ad.add_f64(2.0 * m + 1.0);
        let bm_ = Dd::new(b).add_f64(-m);
        let num = ad
            .add_f64(m - 1.0)
            .mul(cd.add_f64(m - 1.0).mul(x).mul(bm_.mul(x)))
            .mul_f64(m);
        let am = if m == 1.0 {
            num.mul(ad.add_f64(3.0)).div(ad.add_f64(1.0))
        } else {
            num.mul(ad.add_f64(2.0 * m - 3.0).mul(den2))
        };
        // m·den·den2 + m(b − m)x·den2 + (a + m)(λ + 1 + m(2 − x))·den
        //   = den2·m·(den + (b − m)x) + (a + m)(λ + 1 + m(2 − x))·den.
        let t12 = den.add(bm_.mul(x)).mul(den2).mul_f64(m);
        let t3 = ad.add_f64(m).mul(lam1.add(two_x.mul_f64(m))).mul(den);
        (am, t12.add(t3))
    };
    // b₀ = a(λ + 1)/(a + 1).
    let b0 = a * (lam + 1.0) / (a + 1.0);
    const BIG: f64 = 4.149_515_568_880_993e180; // 2^600
    const SMALL: f64 = 2.409_919_865_102_884_5e-181; // 2^−600
    // Forwards: the convergents Pₘ/Qₘ, two steps at a time, checked after each pair.
    const KEEP: usize = 128;
    let mut co = [(0.0f64, 0.0f64); KEEP];
    let (mut p1, mut p0, mut q1, mut q0) = (1.0, b0, 0.0, 1.0);
    let mut prev = b0;
    let mut i = 1usize;
    let mut stop = usize::MAX;
    while i < 100_000 {
        let m = i as f64;
        let (a1, b1) = coef(m);
        let (a2, b2) = coef(m + 1.0);
        if i + 1 < KEEP {
            co[i] = (a1, b1);
            co[i + 1] = (a2, b2);
        }
        let p2 = fma(b1, p0, a1 * p1);
        let q2 = fma(b1, q0, a1 * q1);
        let p3 = fma(b2, p2, a2 * p0);
        let q3 = fma(b2, q2, a2 * q0);
        (p1, p0, q1, q0) = (p2, p3, q2, q3);
        if q0.abs() > BIG || p0.abs() > BIG {
            (p1, p0, q1, q0) = (p1 * SMALL, p0 * SMALL, q1 * SMALL, q0 * SMALL);
        }
        i += 2;
        if stop == usize::MAX {
            let f = p0 / q0;
            if (f - prev).abs() <= CF_EPS * f.abs() {
                // A few more terms for the slowly converging tail.
                stop = i + 3 + i / 8;
            }
            prev = f;
        } else if i >= stop {
            break;
        }
    }
    let m = (i - 1) as f64;
    // Backwards in projective form, t = N/D; the last steps in double-double.
    const DD_STEPS: usize = 2;
    let (mut n, mut d) = (0.0f64, 1.0f64);
    let mut k = m as usize;
    while k > DD_STEPS {
        let (am, bm) = if k < KEEP { co[k] } else { coef(k as f64) };
        (n, d) = (am * d, fma(bm, d, n));
        if d.abs() > BIG {
            (n, d) = (n * SMALL, d * SMALL);
        } else if d.abs() < SMALL {
            (n, d) = (n * BIG, d * BIG);
        }
        k -= 1;
    }
    let (mut nd, mut dd) = (Dd::new(n), Dd::new(d));
    while k >= 1 {
        let (am, bm) = coef_dd(k as f64);
        (nd, dd) = (am.mul(dd), bm.mul(dd).add(nd));
        k -= 1;
    }
    // I = PT/(b₀ + N/D) = PT·(a + 1)D/(a(λ + 1)D + (a + 1)N): one division.
    let den = lam1.mul_f64(a).mul(dd).add(nd.mul(ad.add_f64(1.0)));
    Scaled {
        e: p.e,
        m: p.m.mul(ad.add_f64(1.0)).mul(dd).div(den),
    }
}

/// `I_x(a, b) − I_x(a + k, b)` (BUP): `x^a y^b/(a B(a, b)) · Σ_{i<k} (a + b)ᵢ/(a + 1)ᵢ xⁱ`.
fn a_step(a: f64, b: f64, x: Dd, y: Dd, k: u32, pre: &Pre) -> Dd {
    let p = power_terms(a, b, x, y, true, pre);
    // The terms in double-double: called with x above 0.7, they fall slowly (or grow first),
    // and an f64 recurrence would gather an ulp every few terms.
    let (mut s, mut term) = (Dd::new(1.0), Dd::new(1.0));
    let ab = Dd::new(a).add_f64(b);
    for i in 0..k.saturating_sub(1) {
        let i = i as f64;
        term = term
            .mul(ab.add_f64(i).mul(x))
            .div(Dd::new(a).add_f64(i + 1.0));
        s = s.add(term);
    }
    Scaled {
        e: p.e.sub(ln_dd(a)),
        m: p.m.mul(s),
    }
    .value()
}

/// `(Γ(1 + b) − 1)` for `0 < b ≤ 1`, as a double-double.
fn gamma1pm1(b: f64) -> Dd {
    if b <= 0.5 {
        let d = rgamma1p(b);
        // (1 − D)/D.
        Dd::new(1.0).sub(d).div(d)
    } else {
        // Γ(1 + b) = b Γ(b) = b/D(b − 1).
        let d = rgamma1p(b - 1.0);
        Dd::new(b).sub(d).div(d)
    }
}

/// `e^z − 1` of a double-double.
fn expm1_dd(z: Dd) -> Dd {
    if z.hi.abs() < 0.0625 {
        // z + z²/2 + z³ P(z).
        let h = z.hi;
        let (q, qe) = two_prod(h, 0.5 * h);
        let tail = h
            * h
            * h
            * poly(
                h,
                &[
                    1.0 / 6.0,
                    1.0 / 24.0,
                    1.0 / 120.0,
                    1.0 / 720.0,
                    1.0 / 5040.0,
                    1.0 / 40320.0,
                    1.0 / 362_880.0,
                    1.0 / 3_628_800.0,
                ],
            );
        return z.add(Dd {
            hi: q,
            lo: qe + h * z.lo + tail,
        });
    }
    if z.hi < -40.0 {
        return Dd::new(-1.0);
    }
    exp_dd(z).add_f64(-1.0)
}

/// `Q(b, u)/h` with `h = u^b e^(−u)/Γ(b)`, for `0 < b ≤ 1`, `u > 0`.
/// The double-double steps of `q_ratio`'s continued fraction.
const Q_DD_STEPS: f64 = 3.0;

fn q_ratio(b: f64, u: Dd) -> f64 {
    let uv = u.hi;
    if uv >= 1.5 {
        // Legendre's continued fraction 1/(u + 1 − b − 1(1 − b)/(u + 3 − b − …)), evaluated
        // backwards from a depth that suffices (forwards it loses ten ulps or so).
        let n = (8.0 + 90.0 / uv.sqrt()) as u32;
        // In projective form t = N/D, (N, D) ← (aₖD, bₖD + N): no division per step.
        const BIG: f64 = 4.149_515_568_880_993e180; // 2^600
        const SMALL: f64 = 2.409_919_865_102_884_5e-181; // 2^−600
        let (mut nn, mut dn) = (0.0f64, 1.0f64);
        let mut k = n as f64;
        while k > Q_DD_STEPS {
            (nn, dn) = (k * (b - k) * dn, fma(uv - b + 2.0 * k + 1.0, dn, nn));
            if dn.abs() > BIG {
                (nn, dn) = (nn * SMALL, dn * SMALL);
            }
            k -= 1.0;
        }
        // The last steps in double-double, on which the value depends most.
        let ub = u.add_f64(-b);
        let (mut nd, mut dd) = (Dd::new(nn), Dd::new(dn));
        while k >= 1.0 {
            let an = Dd::new(b).add_f64(-k).mul_f64(k);
            (nd, dd) = (an.mul(dd), ub.add_f64(2.0 * k + 1.0).mul(dd).add(nd));
            k -= 1.0;
        }
        // 1/(u + 1 − b + N/D) = D/((u + 1 − b)D + N).
        return dd.div(ub.add_f64(1.0).mul(dd).add(nd)).hi;
    }
    // Γ(b, u) = (Γ(1 + b) − 1 − (u^b − 1))/b − u^b Σ_{n≥1} (−u)ⁿ/(n! (b + n)).
    let lu = ln_of_dd(u);
    let pm1 = expm1_dd(lu.mul_f64(b));
    let g = gamma1pm1(b).sub(pm1).div(Dd::new(b));
    // The terms alternate and grow to about e^u/√u before they fall: the first ones in
    // double-double, the small rest in f64 (summed without error).
    let mut t = Dd::new(1.0);
    let mut s = Dd::new(0.0);
    let mut n = 1.0;
    while n <= 8.0 {
        t = t.mul(u).neg().div_f64(n);
        let term = t.div(Dd::new(b).add_f64(n));
        s = s.add(term);
        n += 1.0;
    }
    let mut tf = t.hi;
    let (mut sh, mut sl) = (s.hi, s.lo);
    while n < 1000.0 {
        tf *= -uv / n;
        let term = tf / (b + n);
        let (h, e) = two_sum(sh, term);
        sh = h;
        sl += e;
        if term.abs() <= 1e-20 * sh.abs() {
            break;
        }
        n += 1.0;
    }
    let s = Dd { hi: sh, lo: sl };
    let ub = pm1.add_f64(1.0);
    let big = g.sub(ub.mul(s));
    // ratio = Γ(b, u) e^u / u^b.
    let e = Dd::new(uv).add_f64(u.lo).sub(lu.mul_f64(b));
    big.mul(exp_dd(e)).hi
}

/// The coefficients `1/(2n + 1)!` of BGRAT's recurrence.
static INV_ODD_FACT: [f64; 31] = {
    let mut t = [0.0; 31];
    let mut f = 1.0f64;
    let mut i = 1;
    t[0] = 1.0;
    while i < 31 {
        // (2i + 1)! = (2i − 1)! (2i)(2i + 1).
        f *= (2 * i) as f64 * (2 * i + 1) as f64;
        t[i] = 1.0 / f;
        i += 1;
    }
    t
};

/// `s₀ + I_x(a, b)` for a small `b ≤ 1` and a large `a` (BGRAT, DiDonato and Morris's
/// equations 9 to 9.6).
fn bgrat(a: f64, b: f64, x: Dd, s0: Dd) -> Val {
    let bm1 = b - 1.0;
    // t = a + (b − 1)/2 and u = −t ln x, in double-double: u sits in the exponent.
    let (mh, ml) = two_sum(b, -1.0);
    let td = Dd::new(a).add(Dd {
        hi: 0.5 * mh,
        lo: 0.5 * ml,
    });
    let t = td.hi;
    let lx = ln_of_dd(x);
    let u = lx.mul(td).neg();
    // ln(prefix) = b ln(−ln x) − u − ln Γ(b) − (ln Γ(a) − ln Γ(a + b)).
    let lp = ln_of_dd(lx.neg())
        .mul_f64(b)
        .sub(u)
        .sub(lgamma_pos_dd(b))
        .sub(lgamma_delta(a, b));
    let mut j = q_ratio(b, u);
    // s₀ + prefix · Σ pₙ Jₙ, the sum relative to the prefix (which may be tiny).
    let mut inner = Dd::new(j);
    let mut p = [0.0f64; 30];
    p[0] = 1.0;
    let lx2 = {
        let h = lx.hi * 0.5;
        h * h
    };
    let mut lxp = 1.0;
    let t4 = 4.0 * t * t;
    let mut b2n = b;
    let uv = u.hi;
    for n in 1..30 {
        let nf = n as f64;
        let mut pn = 0.0;
        for m in 1..n {
            let mbn = m as f64 * b - nf;
            pn += mbn * p[n - m] * INV_ODD_FACT[m];
        }
        pn /= nf;
        pn += bm1 * INV_ODD_FACT[n];
        p[n] = pn;
        j = (b2n * (b2n + 1.0) * j + (uv + b2n + 1.0) * lxp) / t4;
        lxp *= lx2;
        b2n += 2.0;
        let r = pn * j;
        inner = inner.add_f64(r);
        if r.abs() <= EPS * inner.hi.abs() {
            break;
        }
    }
    Val {
        s0,
        sc: Scaled { e: lp, m: inner },
    }
}

/// `erfcx(z) = e^(z²) erfc(z)` for `z ≥ 0`, as an unrounded `hi + lo` (about half an ulp;
/// private to BASYM).
fn erfcx(z: f64) -> Dd {
    // The area's one double-double erfcx (in igamma.rs).
    erfcx_pos_dd(z)
}

/// `√π/4` as a double-double.
const SQRT_PI_4: Dd = Dd {
    hi: 0.443_113_462_726_379,
    lo: -1.916_646_624_956_449_7e-17,
};

/// `I_x(a, b)` for large `a` and `b` beside the mean (BASYM, DiDonato and Morris's asymptotic
/// expansion), with `λ = a − (a + b)x ≥ 0`: `e₀ e^(E) Σ`, `E` the exponent of the power
/// terms without their square root.
fn basym(a: f64, b: f64, x: Dd, y: Dd, pre: &Pre) -> Scaled {
    const E0: f64 = core::f64::consts::FRAC_2_SQRT_PI;
    const E1: f64 = 0.353_553_390_593_273_8;
    let (h, r0, r1, w0) = if a < b {
        let h = a / b;
        (
            h,
            1.0 / (1.0 + h),
            (b - a) / b,
            1.0 / (a * (1.0 + h)).sqrt(),
        )
    } else {
        let h = b / a;
        (
            h,
            1.0 / (1.0 + h),
            (b - a) / a,
            1.0 / (b * (1.0 + h)).sqrt(),
        )
    };
    let p = power_exponent(a, b, x, y);
    // f = −(a φ(r) + b φ(s)) ≥ 0, without the Stirling corrections.
    let f = (-p.hi).max(0.0);
    let z0 = f.sqrt();
    let z = 0.5 * (z0 / E1);
    let z2 = f + f;
    let mut a0 = [0.0f64; 21];
    let mut b0 = [0.0f64; 21];
    let mut c = [0.0f64; 21];
    let mut d = [0.0f64; 21];
    a0[0] = (2.0 / 3.0) * r1;
    c[0] = -0.5 * a0[0];
    d[0] = -c[0];
    // j₀ = (√π/4) erfcx(z₀), carried in double-double into the sum.
    let j0d = erfcx(z0).mul(SQRT_PI_4);
    let mut j0 = j0d.hi;
    let mut j1 = E1;
    let mut sum = j0d.add_f64(d[0] * w0 * j1);
    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;
    let mut n = 2;
    while n <= 20 {
        hn *= h2;
        a0[n - 1] = 2.0 * r0 * (1.0 + h * hn) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = 2.0 * r1 * s / (n as f64 + 3.0);
        for i in n..=np1 {
            let r = -0.5 * (i as f64 + 1.0);
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for j in 1..i {
                dsum += d[i - j - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }
        j0 = E1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = E1 * zn + n as f64 * j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[np1 - 1] * w * j1;
        sum = sum.add_f64(t0 + t1);
        if t0.abs() + t1.abs() <= EPS * sum.hi {
            break;
        }
        n += 2;
    }
    Scaled {
        e: p.add(pre.ds),
        m: sum.mul_f64(E0),
    }
}

/// The computed side of `I_x(a, b)` (or of its complement with `inv`), and whether the result is
/// its complement; the arguments as double-doubles, `x + y = 1`, both in `(0, 1)`.
fn ibeta_core(
    mut a: f64,
    mut b: f64,
    mut x: Dd,
    mut y: Dd,
    mut inv: bool,
    pre: &Pre,
) -> (Val, bool) {
    fn swap(a: &mut f64, b: &mut f64, x: &mut Dd, y: &mut Dd, inv: &mut bool) {
        core::mem::swap(a, b);
        core::mem::swap(x, y);
        *inv = !*inv;
    }
    if a == 1.0 {
        swap(&mut a, &mut b, &mut x, &mut y, &mut inv);
    }
    if b == 1.0 {
        // I_x(a, 1) = x^a.
        if a == 1.0 {
            return (Val::constant(if inv { y } else { x }), false);
        }
        let e = ln_of_dd(x).mul_f64(a);
        return if inv {
            (Val::constant(expm1_dd(e).neg()), false)
        } else {
            (Val::of(Scaled { e, m: Dd::new(1.0) }), false)
        };
    }
    // The value computed, and whether the result is its complement.
    let fract: Val;
    if a.min(b) <= 1.0 {
        if x.hi > 0.5 {
            swap(&mut a, &mut b, &mut x, &mut y, &mut inv);
        }
        if a.max(b) <= 1.0 {
            if a >= b.min(0.2) || a * ln_rough(x.hi) <= -0.105_360_515_657_826_3 {
                fract = Val::of(series(a, b, x, y, pre));
            } else {
                swap(&mut a, &mut b, &mut x, &mut y, &mut inv);
                if y.hi >= 0.3 {
                    fract = Val::of(series(a, b, x, y, pre));
                } else {
                    // I_x(a, b) = [I_x(a, b) − I_x(a + 20, b)] + I_x(a + 20, b).
                    let s = a_step(a, b, x, y, 20, pre);
                    fract = bgrat(a + 20.0, b, x, s);
                }
            }
        } else if b <= 1.0 || (x.hi < 0.1 && a * ln_rough(b * x.hi) <= -0.356_674_943_938_732_4) {
            fract = Val::of(series(a, b, x, y, pre));
        } else {
            swap(&mut a, &mut b, &mut x, &mut y, &mut inv);
            if y.hi >= 0.3 {
                fract = Val::of(series(a, b, x, y, pre));
            } else if a >= 15.0 {
                fract = bgrat(a, b, x, Dd::new(0.0));
            } else {
                let s = a_step(a, b, x, y, 20, pre);
                fract = bgrat(a + 20.0, b, x, s);
            }
        }
    } else {
        // Both above 1: the side where x is below the mean a/(a + b).
        let lambda = if a < b {
            a - (a + b) * x.hi
        } else {
            (a + b) * y.hi - b
        };
        if lambda < 0.0 {
            swap(&mut a, &mut b, &mut x, &mut y, &mut inv);
        }
        if b < 40.0 {
            // (TOMS 708 uses BUP and BGRAT where b x > 0.7; the backward fraction with its
            // first terms in double-double is as accurate there, and several times faster.)
            if b * x.hi <= 0.7 {
                fract = Val::of(series(a, b, x, y, pre));
            } else {
                fract = Val::of(fraction(a, b, x, y, pre));
            }
        } else {
            let lam = y.mul_f64(a).sub(x.mul_f64(b)).hi;
            if a.min(b) > 100.0 && lam <= 0.03 * a.min(b) {
                fract = Val::of(basym(a, b, x, y, pre));
            } else {
                fract = Val::of(fraction(a, b, x, y, pre));
            }
        }
    }
    (fract, inv)
}

/// `I_x(a, b)` (or its complement with `inv`) of the arguments as double-doubles, `x + y = 1`,
/// both in `(0, 1)`.
fn ibeta(a: f64, b: f64, x: Dd, y: Dd, inv: bool) -> f64 {
    let (v, inv) = ibeta_core(a, b, x, y, inv, &Pre::new(a, b));
    if inv {
        Dd::new(1.0).sub(v.value()).value()
    } else {
        v.value().value()
    }
}

/// `ln I_x(a, b)` (or of its complement), without the underflow of the value.
fn ibeta_ln(a: f64, b: f64, x: Dd, y: Dd, inv: bool, pre: &Pre) -> Dd {
    let (v, inv) = ibeta_core(a, b, x, y, inv, pre);
    if inv {
        ln_of_dd(Dd::new(1.0).sub(v.value()))
    } else {
        v.ln()
    }
}

/// Checks the arguments: `Some(result)` when it is decided without computing.
#[inline]
fn domain(a: f64, b: f64, x: f64, inv: bool) -> Option<f64> {
    if !(a > 0.0 && b > 0.0 && a < f64::INFINITY && b < f64::INFINITY && (0.0..=1.0).contains(&x)) {
        return Some(f64::NAN);
    }
    if x == 0.0 {
        return Some(if inv { 1.0 } else { 0.0 });
    }
    if x == 1.0 {
        return Some(if inv { 0.0 } else { 1.0 });
    }
    None
}

/// `I_x(a, b)` (or its complement) where `a + b` overflows: the distribution is a point at its
/// mean to far below an ulp, so the result is 0 or 1 by the side of `x` (½ at the mean).
#[inline]
fn point_mass(a: f64, b: f64, x: Dd, y: Dd, inv: bool) -> f64 {
    // λ = a y − b x in double-double, scaled down by 2⁻¹⁰⁰⁰ first (both parameters are above
    // 2¹⁰²³ together; Dekker's product, without a fused multiply-add, needs factors below 2⁹⁹⁶).
    const S: f64 = 9.332_636_185_032_189e-302;
    let l = y.mul_f64(a * S).sub(x.mul_f64(b * S)).hi;
    let i = if l > 0.0 {
        0.0
    } else if l < 0.0 {
        1.0
    } else {
        0.5
    };
    if inv { 1.0 - i } else { i }
}

/// `x` and `y = 1 − x` (exactly, as a double-double).
#[inline]
fn split(x: f64) -> (Dd, Dd) {
    let (hi, lo) = two_sum(1.0, -x);
    (Dd::new(x), Dd { hi, lo })
}

/// Parameters from `2⁹⁹⁵` on (where Dekker's product, the error-free product without a fused
/// multiply-add, no longer applies, and `a + b` may overflow).
const HUGE: f64 = 6.696_928_794_914_171e299;

/// `I_x(a, b)` (or its complement) where a parameter is at least `HUGE`: both, a point mass at the
/// mean; one, the incomplete gamma limit `I_x(a, b) → P(a, bx)` (`Q(b, ay)` for a huge `a`),
/// exact in `f64` there (its error is about `a/b`, and `bx` for any `x` but a tiny one is past the
/// range where `P` differs from 1).
fn huge(a: f64, b: f64, x: f64, inv: bool) -> f64 {
    let (xd, yd) = split(x);
    if a.min(b) >= HUGE || a + b == f64::INFINITY {
        return point_mass(a, b, xd, yd, inv);
    }
    let (p, q) = if b >= HUGE {
        let u = b * x;
        (gamma_p_f64(a, u), gamma_q_f64(a, u))
    } else {
        let u = a * yd.hi;
        (gamma_q_f64(b, u), gamma_p_f64(b, u))
    };
    if inv { q } else { p }
}

/// `I_x(a, b)` of `f64`s.
pub(super) fn beta_inc_f64(a: f64, b: f64, x: f64) -> f64 {
    if let Some(r) = domain(a, b, x, false) {
        return r;
    }
    if a.max(b) >= HUGE {
        return huge(a, b, x, false);
    }
    let (xd, yd) = split(x);
    ibeta(a, b, xd, yd, false)
}

/// `1 − I_x(a, b)` of `f64`s.
pub(super) fn beta_inc_c_f64(a: f64, b: f64, x: f64) -> f64 {
    if let Some(r) = domain(a, b, x, true) {
        return r;
    }
    if a.max(b) >= HUGE {
        return huge(a, b, x, true);
    }
    let (xd, yd) = split(x);
    ibeta(a, b, xd, yd, true)
}

/// `Φ⁻¹(p)` roughly (Acklam's rational approximation, about `10⁻⁹`), for `0 < p ≤ 1/2`: the
/// normal quantile of a first guess.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if p < 0.024_25 {
        let q = (-2.0 * ln_dd(p).hi).sqrt();
        let n = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let d = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        return n / d;
    }
    let q = p - 0.5;
    let r = q * q;
    let n = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
    let d = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
    n / d
}

/// `ln B(a, b)`, roughly (for a first guess).
fn lbeta_rough(a: f64, b: f64) -> f64 {
    if a.max(b) >= STIRLING_X {
        // ln B = ln Γ(a) − ln Γ(a + b) + ln Γ(b), the first difference without cancellation.
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        return lgamma_delta(big, small).add(lgamma_pos_dd(small)).hi;
    }
    lgamma_pos_dd(a)
        .add(lgamma_pos_dd(b))
        .sub(lgamma_pos_dd(a + b))
        .hi
}

/// The `x` with `I_x(a, b) = t` (`upper` false) or `1 − I_x(a, b) = t` (`upper` true), for
/// `0 < t ≤ 1/2`.
///
/// A first guess (the normal approximation for large parameters, else the leading term of the
/// series of the tail), then Newton's method on `ln T` in the logarithm of the smaller of `x`
/// and `1 − x` (the variable `s`, the other exact beside it), safeguarded by bisection.
/// The size of a Halley step after which the next is below an ulp.
const ACCEPT: f64 = 1e-6;

fn inverse(a: f64, b: f64, t: f64, upper: bool) -> f64 {
    let ltd = ln_dd(t);
    let lt = ltd.hi;
    let lb = lbeta_rough(a, b);
    let pre = Pre::new(a, b);
    // ln of the power terms' factor (constant): the derivative needs it every step.
    let lnm = ln_of_dd(pre.m);
    // The first guess, as (s, on_y): s = x or y = 1 − x, whichever is the smaller.
    let (mut s, on_y) = {
        let m = a / (a + b);
        let sd = (a * b / ((a + b) * (a + b) * (a + b + 1.0))).sqrt();
        // The normal approximation with Cornish and Fisher's skewness term.
        let z = normal_quantile(t);
        let z = if upper { -z } else { z };
        let g1 = 2.0 * (b - a) * (a + b + 1.0).sqrt() / ((a + b + 2.0) * (a * b).sqrt());
        // (Only where the correction is small: far in the tails it overshoots.)
        let corr = g1 * (z * z - 1.0) / 6.0;
        let corr = if corr.abs() <= 0.25 * (1.0 + z.abs()) {
            corr
        } else {
            0.0
        };
        let xn = m + sd * (z + corr);
        if a.min(b) >= 2.0 && xn > 0.0 && xn < 1.0 {
            if xn > 0.5 {
                (1.0 - xn, true)
            } else {
                (xn, false)
            }
        } else {
            // The leading term of the tail: I ≈ x^a/(a B(a, b)), 1 − I ≈ y^b/(b B(a, b)) (valid
            // where it is small); else the mean.
            let (p, own) = if upper { (b, true) } else { (a, false) };
            let v = exp_f64(Dd::new(((lt + ln_dd(p).hi + lb) / p).min(0.0)));
            if v < 0.9 {
                if v > 0.5 { (1.0 - v, !own) } else { (v, own) }
            } else if m > 0.5 {
                (1.0 - m, true)
            } else {
                (m, false)
            }
        }
    };
    s = s.clamp(f64::MIN_POSITIVE, 1.0);
    // h = ln T(s) − ln t is increasing in ln s when T rises with s.
    let rises = on_y == upper;
    // The bracket of s, and the geometric bisection of it.
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mid = |lo: f64, hi: f64| {
        if lo < 1e-300 {
            (hi * 1e-30).max(f64::from_bits(1))
        } else {
            (lo.sqrt() * hi.sqrt()).clamp(lo, hi)
        }
    };
    let mut halley = false;
    let mut last = f64::INFINITY;
    for _ in 0..200 {
        let (sh, sl) = two_sum(1.0, -s);
        let other = Dd { hi: sh, lo: sl };
        let (x, y) = if on_y {
            (other, Dd::new(s))
        } else {
            (Dd::new(s), other)
        };
        if !(s > 0.0 && x.hi > 0.0 && y.hi > 0.0) {
            lo = lo.max(s);
            s = mid(lo, hi);
            continue;
        }
        // T in logarithms (a tiny t is often subnormal: its value would lose digits).
        let lnt = ibeta_ln(a, b, x, y, upper, &pre);
        if !lnt.hi.is_finite() {
            // T underflowed even in logarithms (zero): s is too small.
            if rises {
                lo = lo.max(s)
            } else {
                hi = hi.min(s)
            }
            s = mid(lo, hi);
            continue;
        }
        let hd = lnt.sub(ltd);
        let h = hd.hi;
        if h == 0.0 && hd.lo == 0.0 {
            break;
        }
        if (h > 0.0) == rises {
            hi = hi.min(s);
        } else {
            lo = lo.max(s);
        }
        // Halley on h(w): h′ = ±s F′/T with F′ = x^(a−1) y^(b−1)/B(a, b), and
        // h″ = h′ (1 + s (ln F′)′_s) − h′², (ln F′)′_x = (a − 1)/x − (b − 1)/y.
        let pt = power_terms(a, b, x, y, true, &pre);
        let lr = pt.e.add(lnm).sub(lnt).hi;
        // (A non-finite logarithm, as for a T that underflowed, leaves a bisection step.)
        let g = if lr.is_finite() {
            s / (x.hi * y.hi) * exp_f64(Dd::new(lr.min(700.0)))
        } else {
            f64::NAN
        };
        let dw = if g > 0.0 && g.is_finite() {
            let sg = if rises { 1.0 } else { -1.0 };
            let h1 = sg * g;
            let dl = (a - 1.0) / x.hi - (b - 1.0) / y.hi;
            let dls = if on_y { -dl } else { dl };
            let h2 = h1 * (1.0 + s * dls) - h1 * h1;
            let newton = -h / h1;
            let den = 1.0 + 0.5 * newton * h2 / h1;
            if den > 0.5 && den < 2.0 {
                halley = true;
                newton / den
            } else {
                halley = false;
                newton
            }
        } else {
            f64::NAN
        };
        // At most a factor e¹⁰ a step: far from the root (where T is near 1) the logarithm's
        // slope is small and Newton's step would overshoot by many decades.
        let dw = dw.clamp(-10.0, 10.0);
        let next = if dw.abs() < 0.01 {
            // s·e^dw = s + s·(e^dw − 1), the series to dw⁶ (below 2⁻⁵⁸ for |dw| < 0.01).
            let em1 = dw
                * (1.0
                    + dw * (0.5
                        + dw * (1.0 / 6.0
                            + dw * (1.0 / 24.0 + dw * (1.0 / 120.0 + dw * (1.0 / 720.0))))));
            fma(s, em1, s)
        } else {
            s * exp_f64(Dd::new(dw.clamp(-700.0, 700.0)))
        };
        // Converged: the step is below half an ulp of s, or (Halley's step, cubic) so small that
        // the next one would be (its error is about the cube of this step's size, and the last
        // step was already small: inside the region of convergence).
        if dw.abs() <= 1.1e-16 || (halley && dw.abs() <= ACCEPT && last.abs() <= 0.05) {
            s = next;
            break;
        }
        last = dw;
        s = if next > lo && next < hi {
            next
        } else {
            mid(lo, hi)
        };
        if hi - lo <= 2.3e-16 * hi {
            break;
        }
    }
    if lo == 0.0 && hi <= f64::from_bits(1) {
        // The root is below the least subnormal.
        s = 0.0;
    }
    if on_y {
        let (sh, _) = two_sum(1.0, -s);
        sh
    } else {
        s
    }
}

/// The inverse of `I` (`upper` false) or of `1 − I` (`upper` true) at `t`.
fn inverse_of(a: f64, b: f64, t: f64, upper: bool) -> f64 {
    if !(a > 0.0 && b > 0.0 && a < f64::INFINITY && b < f64::INFINITY && (0.0..=1.0).contains(&t)) {
        return f64::NAN;
    }
    if t == 0.0 {
        return if upper { 1.0 } else { 0.0 };
    }
    if t == 1.0 {
        return if upper { 0.0 } else { 1.0 };
    }
    if a.min(b) >= HUGE || a + b == f64::INFINITY {
        // A point mass at the mean (see `point_mass`).
        return (0.5 * a) / (0.5 * a + 0.5 * b);
    }
    if a.max(b) >= HUGE {
        // The incomplete gamma limit (see `huge`): P(a, bx) = I, or Q(b, ay) = I.
        let (lower, small, big) = if b >= HUGE {
            (!upper, a, b)
        } else {
            (upper, b, a)
        };
        let u = if lower {
            gamma_p_inv_f64(small, t)
        } else {
            gamma_q_inv_f64(small, t)
        };
        let s = u / big;
        return if b >= HUGE { s } else { 1.0 - s };
    }
    // The tail of T or of its complement that is at most 1/2 (1 − t is exact above 1/2).
    if t <= 0.5 {
        inverse(a, b, t, upper)
    } else {
        inverse(a, b, 1.0 - t, !upper)
    }
}

/// The `x` with `I_x(a, b) = p`.
pub(super) fn beta_inc_inv_f64(a: f64, b: f64, p: f64) -> f64 {
    inverse_of(a, b, p, false)
}

/// The `x` with `1 − I_x(a, b) = q`.
pub(super) fn beta_inc_c_inv_f64(a: f64, b: f64, q: f64) -> f64 {
    inverse_of(a, b, q, true)
}
