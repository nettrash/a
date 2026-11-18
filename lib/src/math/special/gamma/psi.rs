//! The digamma function `ψ` and the polygamma functions `ψ⁽ⁿ⁾`.
//!
//! * `ψ(x)`: `ln x − 1/(2x) − u·T(u)` (`u = 1/x²`, `T` minimax) for `x ≥ 8`; the recurrence
//!   `ψ(x) = ψ(2 + t) + Σ 1/(t + k)` over exact `t + k` down to two minimax pieces of `ψ(2 + t)`;
//!   `(x − x₀)·Q(x)` on `[1, 2]` in four pieces, with the positive zero `x₀` in double-double;
//!   `ψ(x) = ψ(1 + x) − 1/x` below 1; and for `x < 0` the reflection
//!   `ψ(−u) = ψ(u) + 1/u + π cot(πu)`, with `ψ(u) + 1/u = ψ(1 + u)` for `u < 1`.
//! * `ψ⁽ⁿ⁾(x) = (−1)ⁿ⁺¹ n! ζ(n + 1, x)`: the Hurwitz zeta function by direct terms up to where
//!   its Euler–Maclaurin tail converges (or the rest is negligible); for `x < 0` the sum over all
//!   the terms `(x + j)^(−n−1)` split into three Hurwitz zeta functions of exact arguments. Large
//!   orders and arguments are carried as `v·e^l` so that nothing overflows before the end.

use super::super::dd::{Dd, fast_two_sum, two_prod, two_sum};
use super::ddm::*;
use super::real::lgamma_pos_dd;
use super::tab::*;

const PI_DD: Dd = Dd {
    hi: PI_HI,
    lo: PI_LO,
};

/// `1/a` of an `f64` as a double-double.
#[inline(always)]
fn recip(a: f64) -> Dd {
    let q = 1.0 / a;
    let (hi, lo) = fast_two_sum(q, q * rem(q, a, 1.0));
    Dd { hi, lo }
}

/// `a + b` of two double-doubles, lightly (one `two_sum`; for values that need no more).
#[inline(always)]
fn add_light(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.hi, b.hi);
    let (hi, lo) = fast_two_sum(s, e + (a.lo + b.lo));
    Dd { hi, lo }
}

/// `ψ(x)` on `[2, 8)` from its eight pieces, as a double-double `Y + R`.
#[inline(always)]
fn psi_piece(x: f64) -> Dd {
    let (i, c) = if x < 4.0 {
        let i = ((x - 2.0) * 2.0) as usize;
        (i, 2.25 + 0.5 * i as f64)
    } else {
        let f = x.floor();
        (f as usize, f + 0.5)
    };
    // Y + c₁s + (c₀ + s²R(s)), the first two terms without error.
    let s = x - c;
    let [y, c0, c1, r @ ..] = PSI_PIECES[i.min(7)];
    let (p, e) = two_prod(c1, s);
    let (h, e2) = fast_two_sum(y, p);
    let (hi, lo) = fast_two_sum(h, e2 + (e + fma(s * s, poly(s, &r), c0)));
    Dd { hi, lo }
}

/// `ψ(1 + w)` for `0 ≤ w ≤ 1` (`d = 1 + w − x₀` as a double-double), as a double-double:
/// `(Y + R)·d`.
#[inline]
fn psi1p(w: f64, d: Dd) -> Dd {
    let i = ((w * 4.0) as usize).min(3);
    let s = w - (0.125 + 0.25 * i as f64);
    let [y, r @ ..] = PSI_Q[i];
    let r = poly(s, &r);
    let (p, e) = two_prod(y, d.hi);
    let (hi, lo) = fast_two_sum(p, e + fma(r, d.hi, (y + r) * d.lo));
    Dd { hi, lo }
}

/// `1 + w − x₀ = w − (x₀ − 1)` as a double-double (`x₀ − 1` is exact).
#[inline(always)]
fn shifted(w: f64) -> Dd {
    let (h, e) = two_sum(w, 1.0 - PSI_ZERO_HI);
    let (hi, lo) = fast_two_sum(h, e - PSI_ZERO_LO);
    Dd { hi, lo }
}

/// `ψ(x)` for `x > 0`, as a double-double.
pub(super) fn psi_pos_dd(x: f64) -> Dd {
    if x >= 8.0 {
        // ln x − 1/(2x) − u T(u).
        let r = 1.0 / x;
        let u = r * r;
        return ln_dd(x).add_f64(-0.5 * r).add_f64(-u * poly(u, &PSI_TAIL));
    }
    if x >= 2.0 {
        return psi_piece(x);
    }
    if x >= 1.0 {
        // x − x₀ is exact in its first step on [1, 2].
        // (x − x₀ is at least an ulp of x₀ apart from 0, so the sum is a double-double.)
        let (hi, lo) = fast_two_sum(x - PSI_ZERO_HI, -PSI_ZERO_LO);
        let d = Dd { hi, lo };
        return psi1p(x - 1.0, d);
    }
    // ψ(x) = ψ(1 + x) − 1/x; 1 + x − x₀ = x − (x₀ − 1), with x₀ − 1 exact.
    let d = shifted(x);
    let r = recip(x);
    if r.hi.is_infinite() {
        return Dd::new(-r.hi);
    }
    add_light(psi1p(x, d), r.neg())
}

/// `ψ(x)` of an `f64`: `∓∞` at `±0`, NaN at the negative integers and at `−∞`.
pub(super) fn digamma_f64(x: f64) -> f64 {
    if x > 0.0 {
        if x == f64::INFINITY {
            return x;
        }
        if x >= 8.0 {
            // ln x + (its low part − 1/(2x) − u T(u)), rounded once.
            let r = 1.0 / x;
            let u = r * r;
            let l = ln_dd(x);
            return l.hi + (l.lo - fma(u, poly(u, &PSI_TAIL), 0.5 * r));
        }
        return psi_pos_dd(x).value();
    }
    if x == 0.0 {
        return -1.0 / x;
    }
    if x.is_nan() || x == f64::NEG_INFINITY || x == x.floor() {
        return f64::NAN;
    }
    // ψ(−u) = ψ(u) + 1/u + π cot(πu).
    let u = -x;
    let base = if u < 1.0 {
        let d = shifted(u);
        psi1p(u, d)
    } else {
        add_light(psi_pos_dd(u), recip(u))
    };
    add_light(base, PI_DD.mul(cot_pi_dd(u))).value()
}

/// A value `v·e^l` (`l` zero when unscaled).
#[derive(Clone, Copy, Debug)]
struct Scaled {
    v: Dd,
    l: Dd,
}

/// `y^(−s)` of a double-double `y` for an integer `2 ≤ s ≤ 8`, as a double-double.
#[inline]
fn pow_neg(y: Dd, s: u32) -> Dd {
    let q = Dd::new(1.0).div(y);
    let q2 = q.mul(q);
    let q4 = q2.mul(q2);
    match s {
        2 => q2,
        3 => q2.mul(q),
        4 => q4,
        5 => q4.mul(q),
        6 => q4.mul(q2),
        7 => q4.mul(q2).mul(q),
        _ => q4.mul(q4),
    }
}

/// `2⁻⁶⁰`.
const NEGLIGIBLE: f64 = 8.673_617_379_884_035e-19;

/// `ζ(s, a) = Σ_{j ≥ 0} (a + j)^(−s)` for an integer `s ≥ 2` and `a > 0`: unscaled for `s ≤ 8`
/// (`scaled` false; the caller has checked the range), else as `v·e^l` with `l = −s ln a`.
fn hurwitz(s: u32, a: Dd, scaled: bool) -> Scaled {
    let sf = s as f64;
    let start = 0.38 * (sf + 20.0) + 0.5;
    let la = if scaled { ln_of_dd(a) } else { Dd::new(0.0) };
    // (a + j)^(−s), or relative to a^(−s).
    let term = |y: Dd| -> Dd {
        if scaled {
            let e = ln_of_dd(y).sub(la).mul_f64(-sf);
            if e.hi < -700.0 {
                Dd::new(0.0)
            } else {
                exp_dd(e)
            }
        } else {
            pow_neg(y, s)
        }
    };
    let mut sum = Dd::new(0.0);
    let mut j = 0.0;
    loop {
        let y = a.add_f64(j);
        if y.hi >= start {
            sum = sum.add(tail(s, y, term(y)));
            break;
        }
        let t = term(y);
        sum = sum.add(t);
        // The rest is below t·(1 + y/(s − 1)).
        if t.hi * (1.0 + y.hi / (sf - 1.0)) < NEGLIGIBLE * sum.hi {
            break;
        }
        j += 1.0;
    }
    Scaled {
        v: sum,
        l: la.mul_f64(-sf),
    }
}

/// The Euler–Maclaurin tail `Σ_{j ≥ 0} (y + j)^(−s)` given `p = y^(−s)` (possibly relative):
/// `p·(y/(s − 1) + 1/2 + Σ B₂ₖ/(2k)! (s)₂ₖ₋₁ y^(1−2k))`, for `y` past the start of convergence.
fn tail(s: u32, y: Dd, p: Dd) -> Dd {
    let sf = s as f64;
    // The leading terms and the first Bernoulli term in double-double.
    let lead = y.div(Dd::new(sf - 1.0)).add_f64(0.5);
    let t1 = Dd::new(sf).div(y.mul_f64(12.0));
    let mut b = lead.add(t1);
    // The rest in f64, summed without error: t_k = B₂ₖ/(2k)! (s)₂ₖ₋₁ y^(1−2k).
    let yi = 1.0 / y.hi;
    let u = yi * yi;
    let mut t = BERN_FACT[0] * sf * yi;
    let mut hi = 0.0;
    let mut lo = 0.0;
    for k in 2..=BERN_FACT.len() {
        let kf = k as f64;
        t *=
            BERN_FACT[k - 1] / BERN_FACT[k - 2] * (sf + 2.0 * kf - 3.0) * (sf + 2.0 * kf - 2.0) * u;
        let (h, e) = two_sum(hi, t);
        hi = h;
        lo += e;
        if t.abs() < NEGLIGIBLE * 0.25 * b.hi {
            break;
        }
    }
    b = b.add(Dd { hi, lo });
    b.mul(p)
}

/// `n!` as a double-double, `n ≤ 22` exactly.
#[inline]
fn factorial_small(n: u32) -> f64 {
    let mut f = 1.0;
    let mut k = 2.0;
    while k <= n as f64 {
        f *= k;
        k += 1.0;
    }
    f
}

/// `⌊log₂ |a|⌋` of a nonzero finite `a`.
#[inline]
fn ilogb(a: f64) -> i32 {
    let a = a.abs();
    if a < f64::MIN_POSITIVE {
        ((a * 18_446_744_073_709_551_616.0).to_bits() >> 52) as i32 - 1023 - 64
    } else {
        (a.to_bits() >> 52) as i32 - 1023
    }
}

/// `±e^l·v` rounded, `sign` the sign.
#[inline]
fn finish(sign: f64, x: Scaled) -> f64 {
    if x.v.hi == 0.0 {
        return 0.0 * sign;
    }
    let sign = sign * x.v.hi.signum();
    if x.l.hi > 1500.0 {
        return sign * f64::INFINITY;
    }
    if x.l.hi < -1500.0 {
        return sign * 0.0;
    }
    // e^l·v with the exponent of v taken out first.
    let k = ilogb(x.v.hi);
    let s = pow2i(-k);
    let v = Dd {
        hi: x.v.hi.abs() * s,
        lo: x.v.lo * x.v.hi.signum() * s,
    };
    let (d, m) = exp_parts(x.l);
    sign * scale(d.mul(v), m + k)
}

/// `ψ′(x)` for `x > 0`: `1/x + 1/(2x²) + T(u)/x³` from 10 on, ten minimax pieces on
/// `[2, 10)`, and `ψ′(x) = Σ 1/(x + j)² + ψ′(x + m)` below 2.
fn trigamma_pos(x: f64) -> f64 {
    if x >= 10.0 {
        // 1/x + (its rounding error + 1/(2x²) + T(u)/x³): the rest is below a twentieth of 1/x.
        let r = recip(x);
        let u = r.hi * r.hi;
        return r.hi + (r.lo + u * fma(r.hi, poly(u, &TRIG_TAIL), 0.5));
    }
    // The piece of y ∈ [2, 10) and its centre.
    let piece = |y: f64| -> Dd {
        let (i, c) = if y < 4.0 {
            let i = ((y - 2.0) * 2.0) as usize;
            (i, 2.25 + 0.5 * i as f64)
        } else {
            let f = y.floor();
            (f as usize, f + 0.5)
        };
        let [yc, r @ ..] = TRIG_PIECES[i.min(9)];
        Dd::new(yc).add_f64(poly(y - c, &r))
    };
    if x >= 2.0 {
        return piece(x).value();
    }
    // Below 2: the terms 1/(x + j)², each x + j a double-double, then ψ′ at y = x + m ∈ [2, 3)
    // with the rounding of y corrected through ψ″(y) ≈ −1/y² − 1/y³.
    if x < 1e-150 {
        // 1/x², rounded once (ψ′(1 + x) is below its last bit).
        if x < 7.458_340_731_200_207e-155 {
            return f64::INFINITY;
        }
        let q = recip(x);
        let (p, e) = two_prod(q.hi, q.hi);
        return p + (e + 2.0 * q.hi * q.lo);
    }
    let m = if x < 1.0 { 2.0 } else { 1.0 };
    // 1/(x + j)² for j < m, x + j = h + l exactly: q = 1/h with its residual, then q².
    let mut hi = 0.0;
    let mut lo = 0.0;
    let mut j = 0.0;
    while j < m {
        let (h, l) = two_sum(x, j);
        let q = 1.0 / h;
        let ql = q * (rem(q, h, 1.0) - l * q);
        let (p, e) = two_prod(q, q);
        let (s2, e2) = two_sum(hi, p);
        hi = s2;
        lo += e2 + fma(2.0 * q, ql, e);
        j += 1.0;
    }
    let s = Dd { hi, lo };
    let (y, d) = two_sum(x, m);
    let yi = 1.0 / y;
    add_light(s, piece(y))
        .add_f64(-d * yi * yi * (1.0 + yi))
        .value()
}

/// `ψ⁽ⁿ⁾(x)` of an `f64` for `n ≥ 1`.
pub(super) fn polygamma_f64(n: u32, x: f64) -> f64 {
    if n == 0 {
        return digamma_f64(x);
    }
    let odd = n % 2 == 1;
    // (−1)ⁿ⁺¹
    let sign = if odd { 1.0 } else { -1.0 };
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return 0.0 * sign;
    }
    if x == 0.0 {
        // n!/x^(n+1) at ±0: (−1)ⁿ⁺¹ ∞ at +0, +∞ at −0.
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            sign * f64::INFINITY
        };
    }
    if x < 0.0 && (x == f64::NEG_INFINITY || x == x.floor()) {
        return f64::NAN;
    }
    if n == 1 && x > 0.0 {
        return trigamma_pos(x);
    }
    let s = n + 1;
    // ln n! (or n! itself) and whether the direct, unscaled sum is safe.
    let small = s <= 8;
    if x > 0.0 {
        let safe = small && (s as i32) * ilogb(x).abs() < 900;
        let z = hurwitz(s, Dd::new(x), !safe);
        return with_factorial(sign, n, z, safe);
    }
    // x < 0: Σ_{j ≥ 0} (x + j)^(−s) = (−1)^s [ζ(s, r) − ζ(s, 1 − x)] + ζ(s, p), with K the number
    // of negative terms, r = −(x + K − 1) and p = x + K (both exact, in (0, 1)).
    let k = (-x).ceil();
    let r = -(x + (k - 1.0));
    let p = x + k;
    let (h, l) = two_sum(1.0, -x);
    let far = Dd { hi: h, lo: l };
    let safe = small && (s as i32) * ilogb(r.min(p)).abs() < 900 && ilogb(h) < 100;
    let zr = hurwitz(s, Dd::new(r), !safe);
    let zf = hurwitz(s, far, !safe);
    let zp = hurwitz(s, Dd::new(p), !safe);
    let neg = if s.is_multiple_of(2) { 1.0 } else { -1.0 };
    // The three in one scale (the largest).
    let top = if zr.l.hi >= zp.l.hi { zr.l } else { zp.l };
    let rel = |z: Scaled| -> Dd {
        if safe {
            return z.v;
        }
        let e = z.l.sub(top);
        if e.hi < -700.0 {
            Dd::new(0.0)
        } else {
            z.v.mul(exp_dd(e))
        }
    };
    let v = rel(zr).sub(rel(zf)).mul_f64(neg).add(rel(zp));
    let z = Scaled {
        v,
        l: if safe { Dd::new(0.0) } else { top },
    };
    with_factorial(sign, n, z, safe)
}

/// `sign·n!·z`, rounded.
#[inline]
fn with_factorial(sign: f64, n: u32, z: Scaled, safe: bool) -> f64 {
    if safe {
        return sign * z.v.mul_f64(factorial_small(n)).value();
    }
    let lf = lgamma_pos_dd(n as f64 + 1.0);
    finish(
        sign,
        Scaled {
            v: z.v,
            l: z.l.add(lf),
        },
    )
}
