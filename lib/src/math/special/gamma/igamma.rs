//! The regularised incomplete gamma functions `P(a, x)` and `Q(a, x)` and their inverses.
//!
//! The structure is DiDonato and Morris's (ACM TOMS 12, 1986) as Boost.Math arranges it: the
//! smaller of `P` and `Q` is computed and the other is its complement.
//!
//! * `P` by its power series `P = x^a e^(−x)/Γ(a + 1) · Σ x^k/((a + 1)…(a + k))` when `x` is
//!   below about `a`;
//! * `Q` by the continued fraction of Legendre (modified Lentz) when `x` is above;
//! * `Q` for small `a` and `x < 3` (`x < 1.1` as `a` nears 1) from `Γ(a, x) = (Γ(1 + a) − x^a)/a − x^a Σ (−x)ⁿ/(n!(a + n))`
//!   with `Γ(1 + a) − 1` and `x^a − 1` computed without cancellation;
//! * Temme's uniform asymptotic expansion for `a ≥ 20` with `|z| ≤ 1/2` (`x` within about
//!   `−42 %` and `+58 %` of `a`), written with `e^(−y) erfcx(√y)` so that the large exponent is the
//!   only exponential; its coefficients `c_k(z)` are derived anew (series arithmetic after Temme
//!   1979) and fitted minimax on `|z| ≤ 1/2`.
//!
//! The prefix `x^a e^(−x)/Γ(a)` is computed in double-double: `e^(a ln x − x)/Γ(a)` for `a < 8`,
//! and `√(a/2π) e^(a ln(x/a) + a − x − S(a))` with Stirling's correction `S` above, so that it is
//! accurate to a few units of `2⁻⁶⁰` wherever it does not underflow.

use super::super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::ddm::{
    FMA, exp_parts, fma, ln_dd, ln_of_dd, poly, pow2i, prod_err, rem, round_int, scale,
};
use super::igamma_tab::*;
use super::real::{STIRLING_X, rgamma1p};
use super::tab::{
    EULER_HI, EULER_LO, FACT_HI, FACT_LO, LN2_HI, LN2_LO, RGAMMA, RGAMMA_C2, STIRLING,
    STIRLING_C0_LO,
};

/// `2π` as a double-double.
const TWO_PI: Dd = Dd {
    hi: core::f64::consts::TAU,
    lo: 2.449_293_598_294_706_4e-16,
};

/// `2/√π`.
const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;

/// `2⁹⁰⁰`: beyond it Dekker's product (the products' error-free step without a fused
/// multiply-add) would overflow in its split, so the operands are scaled first.
const BIG: f64 = 8.452_712_498_170_644e270;
/// `2⁻¹²⁸` and `2¹²⁸`.
const DOWN: f64 = 2.938_735_877_055_719e-39;
const UP: f64 = 3.402_823_669_209_385e38;

/// `a·b = p + e` exactly for any finite `a`, `b` whose product does not overflow.
#[inline(always)]
fn tp(a: f64, b: f64) -> (f64, f64) {
    if FMA || (a.abs() < BIG && b.abs() < BIG) {
        two_prod(a, b)
    } else if a.abs() >= b.abs() {
        let (p, e) = two_prod(a * DOWN, b);
        (p * UP, e * UP)
    } else {
        let (p, e) = two_prod(a, b * DOWN);
        (p * UP, e * UP)
    }
}

/// `c − a·b` exactly where it is representable, for any finite operands (`b` the large one).
#[inline(always)]
fn rem_s(a: f64, b: f64, c: f64) -> f64 {
    if FMA || b.abs() < BIG {
        rem(a, b, c)
    } else {
        rem(a, b * DOWN, c * DOWN) * UP
    }
}

/// `n/d` of double-doubles, `d` possibly beyond the range of Dekker's split (`|n/d|` not huge).
#[inline(always)]
fn div_s(n: Dd, d: Dd) -> Dd {
    if FMA || d.hi.abs() < BIG {
        n.div(d)
    } else {
        let q = n.div(Dd {
            hi: d.hi * DOWN,
            lo: d.lo * DOWN,
        });
        Dd {
            hi: q.hi * DOWN,
            lo: q.lo * DOWN,
        }
    }
}

/// A value `d·2^m`: the prefixes, kept apart from their scale until the last product.
#[derive(Clone, Copy)]
struct Scaled {
    d: Dd,
    m: i32,
}

impl Scaled {
    const ZERO: Scaled = Scaled {
        d: Dd { hi: 0.0, lo: 0.0 },
        m: 0,
    };

    /// `self·v`.
    #[inline]
    fn mul_f64(self, v: f64) -> Scaled {
        Scaled {
            d: self.d.mul_f64(v),
            m: self.m,
        }
    }
}

/// `e^e` as a scaled double-double (zero below `e^(−1500)`).
#[inline]
fn exp_scaled(e: Dd) -> Scaled {
    if e.hi.is_nan() || e.hi <= -1500.0 {
        return Scaled::ZERO;
    }
    let (d, m) = exp_parts(e);
    Scaled { d, m }
}

/// The compensated product `Π (t + k)` for `k` in `lo..=hi` (the factors are exact).
#[inline]
fn product(t: f64, lo: i32, hi: i32) -> Dd {
    let mut p = 1.0;
    let mut e = 0.0;
    let mut k = lo;
    while k <= hi {
        let f = t + k as f64;
        let q = p * f;
        e = fma(e, f, prod_err(p, f, q));
        p = q;
        k += 1;
    }
    Dd { hi: p, lo: e }
}

/// `1/Γ(1 + a)` for `0 ≤ a < 9`, as a double-double.
#[inline]
fn rgamma1p_any(a: f64) -> Dd {
    let n = round_int(a);
    let t = a - n;
    let d = rgamma1p(t);
    if n == 0.0 {
        return d;
    }
    // Γ(1 + a) = Γ(1 + t)(t + 1)(t + 2)…(t + n).
    d.div(product(t, 1, n as i32))
}

/// `a ln(x/a) + a − x` (never positive) as a double-double, for `a > 0`, `x > 0`; its absolute
/// error is about `2⁻¹⁰⁰ |x − a|`.
#[inline]
fn log_term(a: f64, x: f64) -> Dd {
    // x/a = q + ql exactly but for the last rounding of ql.
    let q = x / a;
    let ql = rem_s(q, a, x) / a;
    log_term_q(a, x, Dd { hi: q, lo: ql })
}

/// `a ln q + a − x` for `q = x/a` given as a double-double.
#[inline(always)]
fn log_term_q(a: f64, x: f64, q: Dd) -> Dd {
    let l = ln_of_dd(q);
    let (ph, pe) = tp(l.hi, a);
    let pl = fma(l.lo, a, pe);
    let (dh, dl) = two_sum(a, -x);
    let (s, e) = two_sum(ph, dh);
    let (hi, lo) = fast_two_sum(s, e + (pl + dl));
    Dd { hi, lo }
}

/// `1/√(2πa)` as a double-double: the rounded root corrected by one Newton step,
/// `r(1 + (1 − 2πa r²)/2)`, without a double-double square root or division.
#[inline]
fn rsqrt_2pi_a(a: f64) -> Dd {
    let (a, f) = if FMA || a < BIG {
        (a, 1.0)
    } else {
        // 1/√(2πa) = 2⁻⁶⁴/√(2π·a·2⁻¹²⁸).
        (a * DOWN, 5.421_010_862_427_522e-20)
    };
    let w = TWO_PI.mul_f64(a);
    let r = 1.0 / w.hi.sqrt();
    // 1 − w r² in double-double: w r² = (w.hi + w.lo)(p + e), p + e = r².
    let (p, e) = two_prod(r, r);
    let (t, te) = two_prod(w.hi, p);
    let d = (1.0 - t) - (te + fma(w.hi, e, w.lo * p));
    let (hi, lo) = fast_two_sum(r, 0.5 * r * d);
    Dd {
        hi: hi * f,
        lo: lo * f,
    }
}

/// The prefix of `P`: `x^a e^(−x)/Γ(a + 1)`, for `a > 0`, finite `x > 0`.
#[inline]
fn prefix_p(a: f64, x: f64) -> Scaled {
    if a <= 170.0 && a == a.floor() {
        // An integer a: e^(a ln x − x)/a! with a! from the table (the exponent stays below
        // 710).
        let s = exp_scaled(ln_dd(x).mul_f64(a).add_f64(-x));
        let f = Dd {
            hi: FACT_HI[a as usize],
            lo: FACT_LO[a as usize],
        };
        return Scaled {
            d: s.d.div(f),
            m: s.m,
        };
    }
    if a < STIRLING_X {
        let e = ln_dd(x).mul_f64(a).add_f64(-x);
        let s = exp_scaled(e);
        return Scaled {
            d: s.d.mul(rgamma1p_any(a)),
            m: s.m,
        };
    }
    // x^a e^(−x)/Γ(a + 1) = e^(a ln(x/a) + a − x − S(a))/√(2πa), with one division for 1/a
    // (x/a as x·(1/a) in double-double).
    let ia = 1.0 / a;
    let ial = rem(ia, a, 1.0) * ia;
    let (qh, qe) = tp(x, ia);
    let (qh, ql) = fast_two_sum(qh, fma(x, ial, qe));
    let u = ia * ia;
    let [c0, rest @ ..] = STIRLING;
    let st = ia * (fma(u, poly(u, &rest), c0) + STIRLING_C0_LO);
    let s = exp_scaled(log_term_q(a, x, Dd { hi: qh, lo: ql }).add_f64(-st));
    Scaled {
        d: s.d.mul(rsqrt_2pi_a(a)),
        m: s.m,
    }
}

/// `Σ x^k/((a + 1)…(a + k))`, `k ≥ 0`, as a double-double: the terms are carried with their
/// rounding errors (`t·(1 + e)`) and summed in double-double while they matter, then plainly.
fn p_sum(a: f64, x: f64) -> Dd {
    let (mut sh, mut sl) = (1.0, 0.0);
    let (mut t, mut te) = (1.0, 0.0);
    let mut k = 1.0;
    // (Where 1/x overflows the terms beyond the first vanish and need no compensation.)
    let ix = if x > 1e-300 { 1.0 / x } else { 0.0 };
    loop {
        // r = x/(a + k) = rh + rl, rl = (x − rh(a + k))/(a + k) ≈ residual·rh/x.
        let (dh, dl) = two_sum(a, k);
        let rh = x / dh;
        let rl = fma(-rh, dl, rem_s(rh, dh, x)) * (rh * ix);
        let p = t * rh;
        te = fma(te, rh, fma(t, rl, prod_err(t, rh, p)));
        t = p;
        let (h, e) = fast_two_sum(sh, t);
        sh = h;
        sl += e + te;
        k += 1.0;
        if t <= sh * 2.4e-4 {
            break;
        }
    }
    // The tail, where the rounding of a term no longer shows.
    let mut tail = 0.0;
    loop {
        t *= x / (a + k);
        tail += t;
        if t <= sh * 1e-17 {
            break;
        }
        k += 1.0;
    }
    let (hi, lo) = fast_two_sum(sh, sl + tail);
    Dd { hi, lo }
}

/// `P(a, x)` by its power series, for `x` below about `a + 1` (and above 1.1 or with `a` not
/// small).
fn p_series(a: f64, x: f64) -> Scaled {
    let pre = prefix_p(a, x);
    if pre.d.hi == 0.0 {
        return Scaled::ZERO;
    }
    Scaled {
        d: pre.d.mul(p_sum(a, x)),
        m: pre.m,
    }
}

/// `Q(a, x)` by Legendre's continued fraction
/// `Q = x^a e^(−x)/Γ(a) / (b₀ + 1(a − 1)/(b₁ + 2(a − 2)/(b₂ + …)))`, `b_k = x − a + 1 + 2k`, for
/// `x` above about `a`: evaluated from the bottom (stable), the last three steps in
/// double-double.
fn q_fraction(a: f64, x: f64) -> Scaled {
    let pre = prefix_p(a, x).mul_f64(a);
    if pre.d.hi == 0.0 {
        return Scaled::ZERO;
    }
    let (h, l) = two_sum(x, -a);
    let (h, l2) = two_sum(h, 1.0);
    let b0 = Dd { hi: h, lo: l + l2 };
    // Enough terms for 2⁻⁶²: the error of the n-th approximant falls like e^(−4√(nx)) for x
    // near a (about 125/x terms), and needs about 20 terms where x ≈ 1.6a (measured over the
    // region of the method, with a margin).
    let n = (160.0 / x + 4.0 + 24.0 * a / x) as u32;
    // T_j = n_j/(d_j + T_(j+1)), n_j = j(a − j), d_j = b₀ + 2j, from T = 0 below the bottom.
    // Four steps a division: T_j = (c₁₁T + c₁₂)/(c₂₁T + c₂₂) with T = T_(j+4) and the
    // coefficients of the product of the four maps (independent of T, so they are computed
    // ahead and only one division a block is on the chain). The bottom steps' errors shrink by
    // the time they reach the top.
    let nn = |j: f64| j * (a - j);
    let dd = |j: f64| b0.hi + 2.0 * j;
    let mut v = 0.0;
    let mut k = n.max(4) as f64;
    if x < 1e60 {
        while k >= 8.0 {
            let j = k - 3.0;
            let (n0, n1, n2, n3) = (nn(j), nn(j + 1.0), nn(j + 2.0), nn(j + 3.0));
            let (d0, d1, d2, d3) = (dd(j), dd(j + 1.0), dd(j + 2.0), dd(j + 3.0));
            // A = M_j M_(j+1), B = M_(j+2) M_(j+3), M = [[0, n], [1, d]].
            let (a11, a12, a21, a22) = (n0, n0 * d1, d0, fma(d0, d1, n1));
            let (b11, b12, b21, b22) = (n2, n2 * d3, d2, fma(d2, d3, n3));
            let c11 = fma(a11, b11, a12 * b21);
            let c12 = fma(a11, b12, a12 * b22);
            let c21 = fma(a21, b11, a22 * b21);
            let c22 = fma(a21, b12, a22 * b22);
            v = fma(c11, v, c12) / fma(c21, v, c22);
            k -= 4.0;
        }
    }
    while k >= 4.0 {
        v = nn(k) / (dd(k) + (b0.lo + v));
        k -= 1.0;
    }
    let mut k = 3;
    // The last three steps in double-double.
    let mut vd = Dd::new(v);
    while k >= 1 {
        let kf = k as f64;
        let (ah, al) = two_sum(a, -kf);
        let num = Dd { hi: ah, lo: al }.mul_f64(kf);
        let den = b0.add_f64(2.0 * kf).add(vd);
        vd = div_s(num, den);
        k -= 1;
    }
    Scaled {
        d: div_s(pre.d, b0.add(vd)),
        m: pre.m,
    }
}

/// `h = 1 + r₁(1 + r₂(1 + … ))` from the inside out, `r_k = (k + off)/x` for `k = 1..n − 1`
/// (all at most 1 here), compensated: the terms of the finite sums of `Q` at integer and
/// half-integer `a`, relative to their largest.
#[inline]
fn horner_ratio(n: u32, off: f64, x: f64) -> Dd {
    let q = 1.0 / x;
    let ql = rem(q, x, 1.0) * q;
    let (mut h, mut hl) = (1.0, 0.0);
    let mut k = 1;
    while k < n {
        let c = k as f64 + off;
        let (rh, re) = two_prod(c, q);
        let rl = fma(c, ql, re);
        let p = rh * h;
        let pe = prod_err(rh, h, p);
        let (s, se) = fast_two_sum(1.0, p);
        hl = se + fma(rh, hl, fma(rl, h, pe));
        h = s;
        k += 1;
    }
    Dd { hi: h, lo: hl }
}

/// `Q(n, x) = e^(−x) Σ_{k<n} x^k/k!` for an integer `1 ≤ n ≤ 30` and `x ≥ n`, as
/// `e^(−x) x^(n−1)/(n − 1)! · (1 + (n − 1)/x (1 + (n − 2)/x (…)))`.
fn q_int(n: f64, x: f64) -> Scaled {
    let m = n - 1.0;
    let s = exp_scaled(ln_dd(x).mul_f64(m).add_f64(-x));
    if s.d.hi == 0.0 {
        return Scaled::ZERO;
    }
    let f = Dd {
        hi: FACT_HI[m as usize],
        lo: FACT_LO[m as usize],
    };
    Scaled {
        d: s.d.mul(horner_ratio(n as u32, 0.0, x)).div(f),
        m: s.m,
    }
}

/// `d·2^m + e·2^k` as one scaled value.
#[inline]
fn add_scaled(u: Scaled, v: Scaled) -> Scaled {
    let (b, s) = if u.m >= v.m { (u, v) } else { (v, u) };
    let k = s.m - b.m;
    if k < -1000 || s.d.hi == 0.0 {
        return b;
    }
    let f = pow2i(k.max(-1022));
    Scaled {
        d: b.d.add(Dd {
            hi: s.d.hi * f,
            lo: s.d.lo * f,
        }),
        m: b.m,
    }
}

/// `Q(n + 1/2, x) = erfc(√x) + e^(−x) Σ_{k=1}^n x^(k−1/2)/Γ(k + 1/2)` for an integer
/// `0 ≤ n ≤ 19` and `x ≥ n + 1/2`: the sum relative to its last term
/// `x^(a−1) e^(−x)/Γ(a)`, `erfc(√x) = e^(−x) erfcx(√x)`.
fn q_half(a: f64, x: f64) -> Scaled {
    let n = (a - 0.5) as u32;
    let sum = if n == 0 {
        Scaled::ZERO
    } else {
        let pre = prefix_p(a, x).mul_f64(a / x);
        Scaled {
            d: pre.d.mul(horner_ratio(n, 0.5, x)),
            m: pre.m,
        }
    };
    let t = Dd::new(x).sqrt();
    let ex = erfcx_pos_dd(t.hi);
    let ex = ex.add_f64(t.lo * (2.0 * t.hi * ex.hi - TWO_OVER_SQRT_PI));
    let e = exp_scaled(Dd::new(-x));
    add_scaled(
        sum,
        Scaled {
            d: e.d.mul(ex),
            m: e.m,
        },
    )
}

/// `eᵛ − 1` of a double-double, for `|v| < 700`.
#[inline]
fn expm1_dd(v: Dd) -> Dd {
    if v.hi.abs() > 0.5 {
        let (d, m) = exp_parts(v);
        let s = pow2i(m);
        return Dd {
            hi: d.hi * s,
            lo: d.lo * s,
        }
        .add_f64(-1.0);
    }
    let h = v.hi;
    let (qh, ql) = two_prod(h, 0.5 * h);
    let (s, e) = fast_two_sum(h, qh);
    let lo = e + ql + fma(v.lo, h, v.lo) + h * h * h * poly(h, &EXPM1_TAIL);
    let (hi, lo) = fast_two_sum(s, lo);
    Dd { hi, lo }
}

/// `Q(a, x)` for small `a` (below about 0.83) and `x < 3`:
/// `Q = (Γ(1 + a) − 1 − (x^a − 1) − a x^a Σ (−x)ⁿ/(n!(a + n)))/Γ(1 + a)`.
fn q_small_a(a: f64, x: f64) -> Scaled {
    let n = round_int(a);
    let t = a - n;
    // Γ(1 + a) = Π/d with Π = 1 (n = 0) or a (n = 1); Γ(1 + a) − 1 without cancellation.
    let (p, d, g1) = if n == 0.0 {
        // 1/Γ(1 + a) = 1 + u, u = γa + c₂a² + a²R(a): Γ(1 + a) − 1 = −u/(1 + u).
        let (th, tl) = two_prod(a, a);
        let (gh, gl) = two_prod(EULER_HI, a);
        let (ch, cl) = two_prod(RGAMMA_C2, th);
        let (uh, ue) = two_sum(gh, ch);
        let ul = ue + fma(EULER_LO, a, gl) + fma(RGAMMA_C2, tl, cl) + th * poly(a, &RGAMMA);
        let (uh, ul) = fast_two_sum(uh, ul);
        let u = Dd { hi: uh, lo: ul };
        let d = u.add_f64(1.0);
        (1.0, d, u.neg().div(d))
    } else {
        let d = rgamma1p(t);
        (a, d, Dd::new(a).sub(d).div(d))
    };
    let pm1 = expm1_dd(ln_dd(x).mul_f64(a));
    // Σ (−x)ⁿ/(n!(a + n)), n ≥ 1, in double-double (it cancels against the rest).
    let (mut sh, mut sl) = (0.0, 0.0);
    let (mut u, mut ue) = (1.0, 0.0);
    let mut k = 1.0;
    loop {
        // u = (−x)ᵏ/k! with its error; term = u/(a + k).
        let r = -x / k;
        let rl = rem(r, k, -x) / k;
        let p = u * r;
        ue = fma(ue, r, fma(u, rl, prod_err(u, r, p)));
        u = p;
        let (dh, dl) = two_sum(a, k);
        let q = u / dh;
        // The remainder u − q·dh exactly (a fused multiply-add, or Dekker's product).
        let ql = (rem(q, dh, u) + fma(-q, dl, ue)) / dh;
        let (h, e) = two_sum(sh, q);
        sh = h;
        sl += e + ql;
        if q.abs() <= sh.abs() * 1e-18 {
            break;
        }
        k += 1.0;
    }
    let s = Dd { hi: sh, lo: sl };
    let xa = pm1.add_f64(1.0);
    let core = g1.sub(pm1).sub(xa.mul(s).mul_f64(a));
    Scaled {
        d: core.mul(d).div(Dd::new(p)),
        m: 0,
    }
}

/// `erfcx(t) = e^(t²) erfc(t)` for `t ≥ 0`, as a double-double (relative error about `2⁻⁶⁰`):
/// eight minimax pieces `Y + R(s)` of width 1/2 below 4, then `(1/t)·g(1/t²)`.
pub(super) fn erfcx_pos_dd(t: f64) -> Dd {
    if t < 4.0 {
        let k = (2.0 * t) as usize;
        let c = 0.5 * k as f64 + 0.25;
        return Dd::new(ERFCX_Y[k]).add_f64(poly(t - c, &ERFCX_R[k]));
    }
    // 1/t = r + rl.
    let r = 1.0 / t;
    let rl = rem(r, t, 1.0) / t;
    let u = r * r;
    let g = if t < 8.0 {
        Dd::new(ERFCX_U_MID_Y).add_f64(poly(u - 5.0 / 128.0, &ERFCX_U_MID))
    } else {
        Dd::new(ERFCX_U_TAIL_Y).add_f64(poly(u, &ERFCX_U_TAIL))
    };
    g.mul(Dd { hi: r, lo: rl })
}

/// Temme's expansion for large `a` with `x` near `a`: the smaller of `P` and `Q` (`Q` when
/// `x ≥ a`), as `e^(−y)(erfcx(√y)/2 ± W/√(2πa))`, `y = x − a − a ln(x/a)`.
fn temme(a: f64, x: f64) -> Scaled {
    let e = log_term(a, x);
    let y = e.neg();
    let phi = y.hi / a;
    let z = if x < a {
        -(2.0 * phi).sqrt()
    } else {
        (2.0 * phi).sqrt()
    };
    let w = [
        poly(z, &TEMME_C0),
        poly(z, &TEMME_C1),
        poly(z, &TEMME_C2),
        poly(z, &TEMME_C3),
        poly(z, &TEMME_C4),
        poly(z, &TEMME_C5),
        poly(z, &TEMME_C6),
        poly(z, &TEMME_C7),
        poly(z, &TEMME_C8),
        poly(z, &TEMME_C9),
        poly(z, &TEMME_C10),
        poly(z, &TEMME_C11),
        poly(z, &TEMME_C12),
    ];
    let w = poly(1.0 / a, &w);
    // erfcx(√y), with the low part of √y as a first-order correction.
    let t = if y.hi > 0.0 { y.sqrt() } else { Dd::new(0.0) };
    let ex = erfcx_pos_dd(t.hi);
    let ex = ex.add_f64(t.lo * (2.0 * t.hi * ex.hi - TWO_OVER_SQRT_PI));
    let r = rsqrt_2pi_a(a).hi;
    let sw = if x < a { -w } else { w };
    let bracket = ex.mul_f64(0.5).add_f64(sw * r);
    let s = exp_scaled(e);
    Scaled {
        d: s.d.mul(bracket),
        m: s.m,
    }
}

/// Whether Temme's expansion is the method at `(a, x)`: `a ≥ 20` and `|z| ≤ 1/2`, i.e.
/// `−0.42 < (x − a)/a < 0.58` (`z² = 2((x − a)/a − ln(x/a))`), where its 13 terms reach `10⁻¹⁸`.
#[inline]
fn use_temme(a: f64, x: f64) -> bool {
    let sigma = (x - a) / a;
    a >= 20.0 && sigma > -0.42 && sigma < 0.58
}

/// The smaller of `P(a, x)` and `Q(a, x)` (nearly: the methods change over near `P = Q`), by
/// the method of the region, and whether it is `Q`; for `a > 0` and finite `x > 0`.
fn smaller(a: f64, x: f64) -> (Scaled, bool) {
    if x < 0.5 {
        if -0.4 / ln_dd(x).hi < a {
            (p_series(a, x), false)
        } else {
            (q_small_a(a, x), true)
        }
    } else if x < 1.1 {
        if 0.75 * x < a {
            (p_series(a, x), false)
        } else {
            (q_small_a(a, x), true)
        }
    } else if a < 0.75 && x < 3.0 {
        // Still Q's side; its series beats the slowly converging fraction here.
        (q_small_a(a, x), true)
    } else if use_temme(a, x) {
        (temme(a, x), x >= a)
    } else if a < 20.0 && x >= a && a == a.floor() {
        // Integer a: the finite sum of Q.
        (q_int(a, x), true)
    } else if a < 20.0 && x >= a && (a - 0.5) == (a - 0.5).floor() {
        (q_half(a, x), true)
    } else if x < a || (a < 20.0 && x - 1.0 / (3.0 * x) < a) {
        (p_series(a, x), false)
    } else {
        (q_fraction(a, x), true)
    }
}

/// `1 − v` of a scaled value at most about 1, as a double-double.
#[inline]
fn complement(v: Scaled) -> Dd {
    if v.m <= -120 {
        return Dd::new(1.0);
    }
    let s = pow2i(v.m);
    Dd::new(1.0).sub(Dd {
        hi: v.d.hi * s,
        lo: v.d.lo * s,
    })
}

/// `(P(a, x), Q(a, x))`.
fn pq(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 || a == f64::INFINITY {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x == f64::INFINITY {
        return (1.0, 0.0);
    }
    let (v, is_q) = smaller(a, x);
    let small = scale(v.d, v.m).clamp(0.0, 1.0);
    // The complement in double-double.
    let big = complement(v).value().clamp(0.0, 1.0);
    if is_q { (big, small) } else { (small, big) }
}

/// `P(a, x)` of `f64`s.
pub(super) fn gamma_p_f64(a: f64, x: f64) -> f64 {
    pq(a, x).0
}

/// `Q(a, x)` of `f64`s.
pub(super) fn gamma_q_f64(a: f64, x: f64) -> f64 {
    pq(a, x).1
}

// ---------------------------------------------------------------------------------- inverses

/// `ln 2` as a double-double.
const LN2: Dd = Dd {
    hi: LN2_HI,
    lo: LN2_LO,
};

/// `ln(d·2^m)` as a double-double (−∞ for zero).
#[inline]
fn ln_scaled(v: Scaled) -> Dd {
    if v.d.hi.is_nan() || v.d.hi <= 0.0 {
        return Dd::new(f64::NEG_INFINITY);
    }
    ln_of_dd(v.d).add(LN2.mul_f64(v.m as f64))
}

/// `ln S(a, x)`, `S = Q` if `upper` else `P`, for `a > 0`, finite `x > 0`: from the
/// double-double result of the method (accurate to about `2⁻⁵⁸`), so that the inverses keep
/// their accuracy where `x` is badly conditioned in `S`.
fn ln_side(a: f64, x: f64, upper: bool) -> Dd {
    let (v, is_q) = smaller(a, x);
    if is_q == upper {
        ln_scaled(v)
    } else {
        let c = complement(v);
        if c.hi > 0.0 {
            ln_of_dd(c)
        } else {
            Dd::new(f64::NEG_INFINITY)
        }
    }
}

/// `ln x` as an `f64`, for finite `x > 0`.
#[inline]
fn ln(x: f64) -> f64 {
    ln_dd(x).hi
}

/// `eˣ` as an `f64`.
#[inline]
fn exp(x: f64) -> f64 {
    super::ddm::exp_f64(Dd::new(x))
}

/// The quantile of the standard normal distribution at `t ≤ 1/2`, by Acklam's rational
/// approximations (relative error below `1.2·10⁻⁹`: a first guess only).
fn normal_quantile_lower(t: f64) -> f64 {
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
    if t < 0.02425 {
        let q = (-2.0 * ln(t)).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = t - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// `λ` with `λ − 1 − ln λ = η²/2` on the side of the sign of `η`: its series for `|η| ≤ 1/2`,
/// else Newton's method from the leading terms (a first guess only).
fn lambda_of_eta(eta: f64) -> f64 {
    if eta.abs() <= 0.5 {
        // λ = 1 + η + η²/3 + η³/36 − η⁴/270 + η⁵/4320 + η⁶/17010 − 139η⁷/5443200 (to 10⁻⁷).
        let c = [
            1.0,
            1.0 / 3.0,
            1.0 / 36.0,
            -1.0 / 270.0,
            1.0 / 4320.0,
            1.0 / 17010.0,
            -139.0 / 5_443_200.0,
        ];
        return 1.0 + eta * poly(eta, &c);
    }
    let h = 0.5 * eta * eta;
    let mut l = if eta > 0.0 {
        if eta < 2.0 {
            1.0 + eta + eta * eta / 3.0
        } else {
            h + 1.0 + ln(h + 1.0)
        }
    } else if eta > -1.2 {
        (1.0 + eta + eta * eta / 3.0).max(0.05)
    } else {
        exp(-1.0 - h)
    };
    for _ in 0..3 {
        let f = l - 1.0 - ln(l) - h;
        l -= f / (1.0 - 1.0 / l);
        if eta > 0.0 {
            l = l.max(1.0 + 1e-300);
        } else {
            l = l.clamp(1e-300, 1.0);
        }
    }
    l
}

/// A first guess of the `x` with `S(a, x) = t`, `t ≤ 1/2` (`S = Q` if `upper`), given
/// `lg1 = ln Γ(a + 1)`. For `a ≥ 1` the inverse of Temme's expansion to its first order
/// (DiDonato and Morris 1986, §5): `η₀ = ∓z/√a` from the normal quantile `z` of `t`, then
/// `η = η₀ + ε₁(η₀)/a` with `ε₁ = ln(η/(λ − 1))/η`, and `x = aλ(η)`. For `a < 1` the leading
/// terms of the tails, `P ≈ x^a/Γ(a + 1)` and `Q ≈ x^(a−1) e^(−x)/Γ(a)`.
fn first_guess(a: f64, t: f64, upper: bool, lg1: f64, lt: f64) -> f64 {
    // The tail of P: x = (t Γ(a + 1))^(1/a), corrected by e^x/(1 + x/(a + 1)) (deep in the
    // lower tail, where it is closer and cheaper than Temme's form).
    let p_tail = || {
        let mut x = exp((lt + lg1) / a);
        for _ in 0..3 {
            x = exp((lt + lg1 + x - ln(1.0 + x / (a + 1.0))) / a);
        }
        x
    };
    if !upper && (a < 1.0 || lt < -12.0 - 0.5 * a) {
        return p_tail();
    }
    if a >= 1.0 {
        let z = normal_quantile_lower(t);
        let e0 = if upper { -z } else { z } / a.sqrt();
        let l0 = lambda_of_eta(e0);
        // ε₁ = ln(η₀/(λ₀ − 1))/η₀ (DiDonato and Morris's form of Temme's c₀; −1/3 at 0).
        let e1 = if e0.abs() < 1e-3 {
            -1.0 / 3.0 + e0 / 36.0
        } else {
            ln(e0 / (l0 - 1.0)) / e0
        };
        // λ(η₀ + δ) to second order: λ′ = ηλ/(λ − 1), λ″ = λ/(λ − 1) − ηλ′/(λ − 1)².
        let d = e1 / a;
        let lm = l0 - 1.0;
        if lm.abs() < 1e-3 {
            return a * lambda_of_eta(e0 + d);
        }
        let l1 = e0 * l0 / lm;
        let l2 = l0 / lm - e0 * l1 / (lm * lm);
        return a * (l0 + d * (l1 + 0.5 * d * l2)).max(1e-300);
    }
    let lg = lg1 - ln(a);
    if -lt - lg > 1.0 {
        // The tail of Q: x = −ln t − ln Γ(a) + (a − 1) ln x + ln(1 + (a − 1)/x).
        let mut x = -lt - lg;
        for _ in 0..4 {
            x = (-lt - lg + (a - 1.0) * ln(x) + ln((1.0 + (a - 1.0) / x).max(1e-3))).max(1.0);
        }
        return x;
    }
    // Else P = 1 − Q ≈ x^a/Γ(a + 1) (ln(1 − t) without rounding 1 − t).
    let l1 = if t < 1e-4 {
        -t * (1.0 + 0.5 * t)
    } else {
        ln(1.0 - t)
    };
    exp((l1 + lg1) / a)
}

/// The `x` with `S(a, x) = t` (`S = Q` if `upper`, else `P`), for `0 < t ≤ 1/2`: Halley's method
/// on `ln S` in `ln x` from a first guess, kept in a bracket (halved geometrically when a step
/// leaves it).
fn inverse(a: f64, t: f64, upper: bool) -> f64 {
    let lt = ln_dd(t);
    // ln Γ(a + 1), and ln Γ(a) for the slope ln(x^a e^(−x)/Γ(a)) = a ln x − x − ln Γ(a).
    let lg1 = if a < 1e-3 {
        // ln Γ(1 + a) = −γa + (π²/12)a² − (ζ(3)/3)a³ + … (1 + a would round away from 1).
        a * (-EULER_HI + a * (0.822_467_033_424_113_2 - a * 0.400_685_634_386_531_4))
    } else {
        super::real::lgamma_f64(a + 1.0)
    };
    let lga = lg1 - ln(a);
    let mut x = first_guess(a, t, upper, lg1, lt.hi);
    if x.is_nan() || x <= f64::MIN_POSITIVE {
        // The root is below the normal range (tiny a, P ≈ x^a/Γ(a + 1)): zero when even the
        // least subnormal is past it.
        let ls = ln_side(a, f64::from_bits(1), upper);
        let h = ls.sub(lt).hi;
        if (h >= 0.0) != upper || h == 0.0 {
            return 0.0;
        }
        x = f64::MIN_POSITIVE;
    }
    if x.is_nan() || x <= 0.0 || x == f64::INFINITY {
        x = a.max(1e-300);
    }
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    let mut last = f64::INFINITY;
    // ln x, carried along with x (only the slope uses it).
    let mut u = ln(x);
    for _ in 0..100 {
        let ls = ln_side(a, x, upper);
        let ld = fma(a, u, -x) - lga;
        let h = if ls.hi == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            ls.sub(lt).value()
        };
        if h == 0.0 {
            return x;
        }
        // S is increasing in x for P, decreasing for Q.
        if (h > 0.0) != upper {
            hi = hi.min(x);
        } else {
            lo = lo.max(x);
        }
        // g = d ln S/d ln x = ±x S′/S, and d²ln S/d(ln x)² = g(a − x) − g²: Halley's step.
        let slope = exp(ld - ls.hi);
        let g = if upper { -slope } else { slope };
        let newton = -h / g;
        let c = 1.0 + newton * ((a - x) - g) * 0.5;
        let du = if c > 0.5 && c < 2.0 {
            newton / c
        } else {
            newton
        };
        if du.abs() < 1e-15 {
            // Converged: the last step is below the resolution of x.
            return x + x * du;
        }
        let mut nx = if du.abs() < 1e-3 {
            x + x * (du + 0.5 * du * du)
        } else {
            x * exp(du.clamp(-700.0, 700.0))
        };
        let mut nu = u + du;
        if nx.is_nan() || nx <= lo || nx >= hi || !du.is_finite() {
            nx = if lo == 0.0 {
                hi * 1e-3
            } else if hi == f64::INFINITY {
                lo * 1e3
            } else {
                (lo.sqrt() * hi.sqrt()).clamp(lo, hi)
            };
            if nx == x {
                return x;
            }
            nu = ln(nx);
        } else if (a >= 1.0 && du.abs() < 1e-6) || (du.abs() < 3e-6 && last.abs() < 1e-2) {
            // Cubic convergence: the next step would be below 10⁻¹⁷ (for a ≥ 1 already from a
            // first step that small; below 1 the slope is too small to trust it so early).
            return nx;
        }
        last = du;
        u = nu;
        x = nx;
    }
    x
}

/// The `x` with `P(a, x) = p`.
pub(super) fn gamma_p_inv_f64(a: f64, p: f64) -> f64 {
    if a.is_nan() || p.is_nan() || a <= 0.0 || a == f64::INFINITY || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p <= 0.5 {
        inverse(a, p, false)
    } else {
        inverse(a, 1.0 - p, true)
    }
}

/// The `x` with `Q(a, x) = q`.
pub(super) fn gamma_q_inv_f64(a: f64, q: f64) -> f64 {
    if a.is_nan() || q.is_nan() || a <= 0.0 || a == f64::INFINITY || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return 0.0;
    }
    if q <= 0.5 {
        inverse(a, q, true)
    } else {
        inverse(a, 1.0 - q, false)
    }
}
