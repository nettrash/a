//! Carlson's symmetric elliptic integrals by the duplication algorithm (Carlson 1995) with series
//! of the thirteenth (`R_F`) and eleventh (`R_D`, `R_J`) orders, and `R_C` in closed form.
//!
//! Each function takes arguments already checked by its caller (finite, non-negative, at most one
//! zero where the integral needs it). The duplication carries every argument with its rounding
//! error (`h + l`, the square roots with their residuals and `λ` with the errors of its products
//! and sums), so that its steps add errors of second order only, and is arranged so that it cannot
//! overflow (`λ/4` from half square roots); tiny arguments are scaled up first, the integrals
//! being homogeneous: `R_F(λx, λy, λz) = λ^(−1/2) R_F(x, y, z)`, `R_D` and `R_J` of degree −3/2.

use super::eft::{mla, res, two_prod, two_sum};
use core::f64::consts::FRAC_PI_2;

/// The duplication of `R_F` stops when every argument is within this factor of their mean: the
/// omitted terms of its thirteenth-order series are then below 2^−56.
const TOL_F: f64 = 0.07;

/// The same for `R_D` and `R_J` and their eleventh-order series.
const TOL_DJ: f64 = 0.035;

/// The factor `4^k` that brings `max < 2^−600` to `[1, 4)`, or 1 for a larger `max` (the
/// iterations below cannot overflow, but subnormal arguments would lose bits).
#[inline(always)]
fn scale_of(max: f64) -> f64 {
    if max >= 2.4e-181 || max == 0.0 {
        return 1.0;
    }
    let e = if max < f64::MIN_POSITIVE {
        ((max * 2f64.powi(600)).to_bits() >> 52) as i32 - 1023 - 600
    } else {
        (max.to_bits() >> 52) as i32 - 1023
    };
    let k = e.div_euclid(2);
    pow2(-2 * k)
}

/// The factor `4^−k` that brings `max > 2^600` towards `[2^600, 2^602)`, but no further than
/// keeps the smallest nonzero argument `min` above `2^−1000` (the integrals are singular as some
/// arguments go to 0, so those must keep their bits; arguments spanning more than `2^1600` lose
/// some at the extremes of the range).
#[inline(always)]
fn down_to_600(max: f64, min: f64) -> f64 {
    let e = (max.to_bits() >> 52) as i32 - 1023;
    let k = -(e - 600).div_euclid(2);
    let em = if min < f64::MIN_POSITIVE {
        -1023
    } else {
        (min.to_bits() >> 52) as i32 - 1023
    };
    let kmin = (-1000 - em + 1).div_euclid(2) + 1;
    let k = k.max(kmin).min(0);
    pow2(2 * k)
}

/// The smallest nonzero value of a few (or `f64::MAX`).
#[inline(always)]
fn min_nonzero(v: &[f64]) -> f64 {
    v.iter()
        .fold(f64::MAX, |m, &x| if x > 0.0 && x < m { x } else { m })
}

/// `2^e` for `e` in `−1022..=1023`.
#[inline(always)]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// A value `h + l`, `|l|` about an ulp of `h` or less: the duplication carries its arguments so,
/// to first order in the rounding errors.
#[derive(Clone, Copy)]
struct V {
    h: f64,
    l: f64,
}

impl V {
    #[inline(always)]
    fn new(h: f64) -> V {
        V { h, l: 0.0 }
    }

    /// `(s, e)` with `s + e = √(h + l) / 2`.
    #[inline(always)]
    fn half_sqrt(self) -> (f64, f64) {
        let s = self.h.sqrt();
        if s == 0.0 {
            return (0.0, 0.0);
        }
        let r = res(s, s, self.h) + self.l;
        (0.5 * s, r * (0.25 * approx_inv(s)))
    }

    /// `self/4 + q`.
    #[inline(always)]
    fn quarter_plus(self, q: V) -> V {
        let (h, l) = two_sum(0.25 * self.h, q.h);
        V {
            h,
            l: l + mla(0.25, self.l, q.l),
        }
    }
}

/// `1/s` to about 2^−6 for a positive normal `s` (one Newton step from a guess of the bits): the
/// residuals it scales are of the order of an ulp, so their own error is of second order.
#[inline(always)]
fn approx_inv(s: f64) -> f64 {
    let y = f64::from_bits(0x7FDE_6238_DA3C_2118u64.wrapping_sub(s.to_bits()));
    y * mla(-s, y, 2.0)
}

/// `λ/4` of the duplication, `λ = √x√y + √x√z + √y√z`, from the half square roots.
#[inline(always)]
fn lambda_q((hx, ex): (f64, f64), (hy, ey): (f64, f64), (hz, ez): (f64, f64)) -> V {
    let (p1, e1) = two_prod(hx, hy);
    let (p2, e2) = two_prod(hx, hz);
    let (p3, e3) = two_prod(hy, hz);
    let (s, t) = two_sum(p1, p2);
    let (s2, t2) = two_sum(s, p3);
    let lo = (t + t2) + (e1 + e2 + e3) + ex * (hy + hz) + ey * (hx + hz) + ez * (hx + hy);
    V { h: s2, l: lo }
}

/// `(a + b + c)/k` as a `V`.
#[inline(always)]
fn mean(vs: &[V], k: f64) -> V {
    let mut s = V::new(0.0);
    for v in vs {
        let (h, l) = two_sum(s.h, v.h);
        s = V {
            h,
            l: l + s.l + v.l,
        };
    }
    let q = s.h / k;
    let r = res(q, k, s.h) + s.l;
    V { h: q, l: r / k }
}

/// `(1 + t) / √A` with one rounding at the end.
#[inline(always)]
fn over_sqrt(t: f64, a: V) -> f64 {
    let s = a.h.sqrt();
    let e = res(s, s, a.h) + a.l; // A − s²
    let r0 = 1.0 / s;
    let rho = res(r0, s, 1.0); // 1 − r0·s
    mla(r0, rho + t - 0.5 * e / a.h, r0)
}

/// `(1 + t) / A^(3/2)` as a double-double `(q, lo)`.
#[inline(always)]
fn over_sqrt3(t: f64, a: V) -> (f64, f64) {
    let s = a.h.sqrt();
    let e = res(s, s, a.h) + a.l;
    let r0 = 1.0 / s;
    let rho = res(r0, s, 1.0);
    let (p, pe) = two_prod(r0, r0);
    let (q, qe) = two_prod(p, r0);
    let qe = qe + pe * r0;
    (q, mla(q, t + 3.0 * rho - 1.5 * e / a.h, qe))
}

/// `R_F(x, y, z)` of finite non-negative arguments, at most one of them zero.
pub(super) fn rf(x: f64, y: f64, z: f64) -> f64 {
    let sc = scale_of(x.max(y).max(z));
    if sc != 1.0 {
        return rf_core(x * sc, y * sc, z * sc) * sc.sqrt();
    }
    rf_core(x, y, z)
}

#[inline(always)]
fn rf_core(x0: f64, y0: f64, z0: f64) -> f64 {
    let a0 = x0 / 3.0 + y0 / 3.0 + z0 / 3.0;
    let (dx, dy) = (a0 - x0, a0 - y0);
    let q = (dx.abs()).max(dy.abs()).max((a0 - z0).abs());
    let (mut x, mut y, mut z) = (V::new(x0), V::new(y0), V::new(z0));
    let mut f = 1.0; // 4^−n
    let mut a = a0;
    while q * f > TOL_F * a {
        let lq = lambda_q(x.half_sqrt(), y.half_sqrt(), z.half_sqrt());
        x = x.quarter_plus(lq);
        y = y.quarter_plus(lq);
        z = z.quarter_plus(lq);
        a = 0.25 * a + lq.h;
        f *= 0.25;
    }
    let am = mean(&[x, y, z], 3.0);
    let xx = dx * f / am.h;
    let yy = dy * f / am.h;
    let zz = -(xx + yy);
    let e2 = mla(xx, yy, -zz * zz);
    let e3 = xx * yy * zz;
    let t = rf_series(e2, e3);
    over_sqrt(t, am)
}

/// The thirteenth-order series of `R_F` less its leading 1 (DLMF 19.19.7 expanded in the elementary
/// symmetric functions `E2`, `E3` of the deviations).
#[inline(always)]
fn rf_series(e2: f64, e3: f64) -> f64 {
    mla(
        mla(
            mla(
                mla(mla(231.0 / 25600.0, e2, -3.0 / 256.0), e2, 35.0 / 2176.0),
                e2,
                -5.0 / 208.0,
            ),
            e2,
            1.0 / 24.0,
        ),
        e2,
        -1.0 / 10.0,
    ) * e2
        + mla(
            mla(
                mla(
                    mla(mla(-77.0 / 1536.0, e2, 315.0 / 5888.0), e2, -35.0 / 608.0),
                    e2,
                    1.0 / 16.0,
                ),
                e2,
                -3.0 / 44.0,
            ),
            e2,
            1.0 / 14.0,
        ) * e3
        + mla(
            mla(mla(-63.0 / 640.0, e2, 5.0 / 64.0), e2, -15.0 / 272.0),
            e2,
            3.0 / 104.0,
        ) * e3
            * e3
        + mla(mla(35.0 / 384.0, e2, -35.0 / 736.0), e2, 5.0 / 304.0) * e3 * e3 * e3
        + (7.0 / 640.0) * e3 * e3 * e3 * e3
}

/// `R_D(x, y, z)` of finite non-negative arguments, `z > 0` and at most one of `x, y` zero.
pub(super) fn rd(x: f64, y: f64, z: f64) -> f64 {
    let big = x.max(y).max(z);
    // A result near the bottom of the range would lose its bits in the sum: bring max to 2^600.
    let sc = if big > 4.15e180 {
        down_to_600(big, min_nonzero(&[x, y, z]))
    } else {
        scale_of(big)
    };
    if sc != 1.0 {
        let s = sc.sqrt();
        return rd_core(x * sc, y * sc, z * sc) * (sc * s);
    }
    rd_core(x, y, z)
}

/// The eleventh-order series of `R_D` and `R_J` less its leading 1 (DLMF 19.19.7 in the elementary
/// symmetric functions of the five deviations `X, Y, Z, P, P`).
#[inline(always)]
fn series_dj(e2: f64, e3: f64, e4: f64, e5: f64) -> f64 {
    mla(
        mla(
            mla(mla(-189.0 / 5888.0, e2, 105.0 / 2432.0), e2, -1.0 / 16.0),
            e2,
            9.0 / 88.0,
        ),
        e2,
        -3.0 / 14.0,
    ) * e2
        + mla(
            mla(
                mla(mla(189.0 / 1280.0, e2, -5.0 / 32.0), e2, 45.0 / 272.0),
                e2,
                -9.0 / 52.0,
            ),
            e2,
            1.0 / 6.0,
        ) * e3
        + mla(
            mla(mla(105.0 / 736.0, e2, -45.0 / 304.0), e2, 3.0 / 20.0),
            e2,
            -3.0 / 22.0,
        ) * e4
        + mla(
            mla(mla(-21.0 / 160.0, e2, 15.0 / 112.0), e2, -9.0 / 68.0),
            e2,
            3.0 / 26.0,
        ) * e5
        + mla(mla(-63.0 / 160.0, e2, 15.0 / 56.0), e2, -9.0 / 68.0) * e3 * e4
        + mla(-45.0 / 184.0, e2, 9.0 / 76.0) * e3 * e5
        + mla(mla(315.0 / 1472.0, e2, -45.0 / 304.0), e2, 3.0 / 40.0) * e3 * e3
        + mla(9.0 / 40.0, e2, -3.0 / 28.0) * e4 * e5
        + mla(-45.0 / 368.0, e2, 9.0 / 152.0) * e4 * e4
        + (9.0 / 184.0) * e5 * e5
        + (9.0 / 80.0) * e3 * e4 * e4
        + (-45.0 / 368.0) * e3 * e3 * e4
        + (9.0 / 80.0) * e3 * e3 * e5
        + mla(-21.0 / 160.0, e2, 5.0 / 112.0) * e3 * e3 * e3
}

/// `w / (s · v)` to first order in the low parts of `s` and `v`, as `(q, lo)`, `w` a power of two.
#[inline(always)]
fn quot(w: f64, (s, es): (f64, f64), v: V) -> (f64, f64) {
    let (d, de) = two_prod(s, v.h);
    if d.is_infinite() {
        // Below 2^−1000 of the integral's other part: R_D of such arguments is about
        // 3/(√z_k z_k) at most.
        return (0.0, 0.0);
    }
    let dl = de + mla(s, v.l, es * v.h);
    let q = w / d;
    let r = res(q, d, w);
    // 1/d = q/w to first order, and w is a power of two: no second division.
    (q, (r - q * dl) * (q / w))
}

#[inline(always)]
fn rd_core(x0: f64, y0: f64, z0: f64) -> f64 {
    let a0 = 0.2 * x0 + 0.2 * y0 + 0.6 * z0;
    let (dx, dy) = (a0 - x0, a0 - y0);
    let q = (dx.abs()).max(dy.abs()).max((a0 - z0).abs());
    let (mut x, mut y, mut z) = (V::new(x0), V::new(y0), V::new(z0));
    let mut f = 1.0;
    let mut a = a0;
    let (mut sh, mut sl) = (0.0, 0.0); // Σ 4^−k / (√z_k (z_k + λ_k)), z_k + λ_k = 4 z_(k+1)
    while q * f > TOL_DJ * a {
        let hz = z.half_sqrt();
        let lq = lambda_q(x.half_sqrt(), y.half_sqrt(), hz);
        x = x.quarter_plus(lq);
        y = y.quarter_plus(lq);
        z = z.quarter_plus(lq);
        a = 0.25 * a + lq.h;
        // 4^−k / (2 (√z/2) · 4 z') = (4^−k / 8) / ((√z/2) z').
        let (th, tl) = quot(0.125 * f, hz, z);
        let (s, e) = two_sum(sh, th);
        sh = s;
        sl += e + tl;
        f *= 0.25;
    }
    let am = mean(&[x, y, z, z, z], 5.0);
    let xx = dx * f / am.h;
    let yy = dy * f / am.h;
    let zz = -(xx + yy) / 3.0;
    let xy = xx * yy;
    let z2 = zz * zz;
    let e2 = xy - 6.0 * z2;
    let e3 = (3.0 * xy - 8.0 * z2) * zz;
    let e4 = 3.0 * (xy - z2) * z2;
    let e5 = xy * z2 * zz;
    let t = series_dj(e2, e3, e4, e5);
    let (qh, ql) = over_sqrt3(t, am);
    let (s3, e3) = two_prod(3.0, sh);
    let (h, l) = two_sum(s3, f * qh);
    h + (l + (e3 + 3.0 * sl + f * ql))
}

/// `R_C(1, 1 + e) − 1` for `|e| < 1/16`: `Σ_(k≥1) (−e)^k / (2k + 1)` to `e^13`
/// (`|e|^14/29 < 2^−58`).
#[inline(always)]
fn rc1m1(e: f64) -> f64 {
    let w = -e;
    let mut p: f64 = 1.0 / 27.0;
    let mut k = 12;
    while k >= 1 {
        p = mla(p, w, 1.0 / (2 * k + 1) as f64);
        k -= 1;
    }
    p * w
}

/// `R_C(1, 1 + e)` for `e` near −1 from `ope = 1 + e` given accurately:
/// `atanh(v)/v = (ln(1 + v) − ½ ln(1 + e))/v`, `v = √(1 − ope)`.
#[inline(always)]
fn rc1_near_m1(ope: f64) -> f64 {
    let (v, vl) = sqrt_r(1.0 - ope, 0.0);
    let n = v.ln_1p() - 0.5 * ope.ln();
    div_r(n, vl / (1.0 + v), (v, vl))
}

/// `R_C(1, 1 + e)` for `|e| ≥ 1/16`, `e > −1`, of `e + el`, about correctly rounded but for the
/// rounding of `atan` or `ln_1p`.
#[inline(always)]
fn rc1_dd(e: f64, el: f64) -> f64 {
    if e > 0.0 {
        let (v, vl) = sqrt_r(e, el);
        div_r(v.atan(), vl / mla(v, v, 1.0), (v, vl))
    } else {
        // atanh(v) = ½ ln_1p(2v/(1 − v)), v = √−e.
        let (v, vl) = sqrt_r(-e, -el);
        let (d, dl) = two_sum(1.0, -v);
        let dl = dl - vl;
        let t = 2.0 * v / d;
        let tl = (res(t, d, 2.0 * v) + 2.0 * vl - t * dl) / d;
        div_r(0.5 * t.ln_1p(), 0.5 * tl / (1.0 + t), (v, vl))
    }
}

/// `atanh(v)` for `0 ≤ v < 1`.
#[inline(always)]
fn atanh(v: f64) -> f64 {
    0.5 * (2.0 * v / (1.0 - v)).ln_1p()
}

/// `(s, e)` with `s + e = √(h + l)` to first order, `h > 0`.
#[inline(always)]
fn sqrt_r(h: f64, l: f64) -> (f64, f64) {
    let s = h.sqrt();
    if s == 0.0 {
        return (0.0, 0.0);
    }
    (s, (res(s, s, h) + l) / (2.0 * s))
}

/// `(n + nl) / (s + es)` rounded once.
#[inline(always)]
fn div_r(n: f64, nl: f64, (s, es): (f64, f64)) -> f64 {
    let q = n / s;
    let r = res(q, s, n);
    q + (r + nl - q * es) / s
}

/// `(h + l)` of the quotient `a / b`, `b > 0`.
#[inline(always)]
fn quot_r(a: f64, al: f64, b: f64) -> (f64, f64) {
    let q = a / b;
    (q, (res(q, b, a) + al) / b)
}

/// `R_C(x, y)` for finite `x ≥ 0`, `y > 0`, its arguments and the functions of them carried with
/// their rounding errors to first order, so that the result is rounded about once more than the
/// elementary function it calls.
pub(super) fn rc_pos(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        return div_r(FRAC_PI_2, 6.123_233_995_736_766e-17, sqrt_r(y, 0.0));
    }
    if x < y {
        // R_C = acos(√(x/y)) / √(y − x) = atan(√((y − x)/x)) / √(y − x).
        let (w, wl) = two_sum(y, -x);
        if w < 0.0625 * x {
            return over_sqrt(rc1m1(w / x), V::new(x));
        }
        if x < 0.5 * y {
            let (q, ql) = quot_r(x, 0.0, y);
            let (v, vl) = sqrt_r(q, ql);
            let nl = -vl / ((1.0 - v) * (1.0 + v)).sqrt();
            div_r(v.acos(), nl, sqrt_r(w, wl))
        } else {
            let (q, ql) = quot_r(w, wl, x);
            let (v, vl) = sqrt_r(q, ql);
            div_r(v.atan(), vl / mla(v, v, 1.0), sqrt_r(w, wl))
        }
    } else {
        // R_C = atanh(√((x − y)/x)) / √(x − y) = ln((√x + √(x − y))/√y) / √(x − y).
        let (w, wl) = two_sum(x, -y);
        if w < 0.0625 * x {
            return over_sqrt(rc1m1(-w / x), V::new(x));
        }
        if x < 2.0 * y {
            let (q, ql) = quot_r(w, wl, x);
            let (v, vl) = sqrt_r(q, ql);
            div_r(atanh(v), vl / ((1.0 - v) * (1.0 + v)), sqrt_r(w, wl))
        } else {
            let (sx, ex) = sqrt_r(x, 0.0);
            let (sw, ew) = sqrt_r(w, wl);
            let (sy, ey) = sqrt_r(y, 0.0);
            let (t, tl) = two_sum(sx, sw);
            let tl = tl + ex + ew;
            let q = t / sy;
            let ql = (res(q, sy, t) + tl - q * ey) / sy;
            div_r(q.ln(), ql / q, (sw, ew))
        }
    }
}

/// `R_C(x, y)` for finite `x ≥ 0`, `y ≠ 0` (the principal value for `y < 0`).
pub(super) fn rc(x: f64, y: f64) -> f64 {
    let sc = scale_of(x.max(y.abs()));
    let (x, y, f) = if sc != 1.0 {
        (x * sc, y * sc, sc.sqrt())
    } else {
        (x, y, 1.0)
    };
    let r = if y > 0.0 {
        rc_pos(x, y)
    } else {
        // R_C(x, y) = √(x/(x − y)) R_C(x − y, −y).
        let w = x - y;
        (x / w).sqrt() * rc_pos(w, -y)
    };
    r * f
}

/// `R_J(x, y, z, p)` of finite non-negative `x, y, z`, at most one zero, and `p > 0`.
pub(super) fn rj(x: f64, y: f64, z: f64, p: f64) -> f64 {
    let big = x.max(y).max(z).max(p);
    // 1/d_m ≈ max^(−3/2) and A^(−3/2) would underflow: bring max to 2^600 (no further, so that a
    // small p keeps its bits).
    let sc = if big > 4.15e180 {
        down_to_600(big, min_nonzero(&[x, y, z, p]))
    } else {
        scale_of(big)
    };
    if sc != 1.0 {
        let s = sc.sqrt();
        return rj_core(x * sc, y * sc, z * sc, p * sc) * (sc * s);
    }
    rj_core(x, y, z, p)
}

#[inline(always)]
fn rj_core(x0: f64, y0: f64, z0: f64, p0: f64) -> f64 {
    let a0 = 0.2 * x0 + 0.2 * y0 + 0.2 * z0 + 0.4 * p0;
    let (dx, dy, dz) = (a0 - x0, a0 - y0, a0 - z0);
    let q = (dx.abs()).max(dy.abs()).max(dz.abs()).max((a0 - p0).abs());
    // e_m = δ 4^−3m / d_m², δ = (p − x)(p − y)(p − z), as a product of bounded ratios.
    let (dpx, dpy, dpz) = (p0 - x0, p0 - y0, p0 - z0);
    let (mut x, mut y, mut z, mut p) = (V::new(x0), V::new(y0), V::new(z0), V::new(p0));
    let mut f = 1.0;
    let mut a = a0;
    let (mut sh, mut sl) = (0.0, 0.0);
    while q * f > TOL_DJ * a {
        let (hx, hy, hz, hp) = (x.half_sqrt(), y.half_sqrt(), z.half_sqrt(), p.half_sqrt());
        let lq = lambda_q(hx, hy, hz);
        // d = (√p + √x)(√p + √y)(√p + √z) = 8 (hp + hx)(hp + hy)(hp + hz).
        let sum2 = |(a, ea): (f64, f64), (b, eb): (f64, f64)| {
            let (s, e) = two_sum(a, b);
            (s, e + ea + eb)
        };
        let (gx, ex) = sum2(hp, hx);
        let (gy, ey) = sum2(hp, hy);
        let (gz, ez) = sum2(hp, hz);
        // 1/d = (1/8)(1/gx)(1/gy)(1/gz), with the first-order errors of the inverses.
        let (ix, iy, iz) = (1.0 / gx, 1.0 / gy, 1.0 / gz);
        let rho =
            res(ix, gx, 1.0) + res(iy, gy, 1.0) + res(iz, gz, 1.0) - (ex * ix + ey * iy + ez * iz);
        let (pp, pe) = two_prod(ix, iy);
        let (inv, pe2) = two_prod(pp, iz);
        let inv_lo = mla(inv, rho, mla(pe, iz, pe2));
        // e = (1/64) Π (4^−m (p₀ − x₀))/g², each ratio bounded (4^−3m alone would underflow
        // while p ≫ x, y, z and e stays near 1).
        let e = (1.0 / 64.0)
            * (((dpx * f) * ix) * ix)
            * (((dpy * f) * iy) * iy)
            * (((dpz * f) * iz) * iz);
        let rc = if e.abs() < 0.0625 {
            // The series is insensitive to the rounding of e there.
            1.0 + rc1m1(e)
        } else if e < -0.5 {
            // e = Π r_i, r_i = (√p − √x_i)/(√p + √x_i), all near ±1: 1 + e = 1 − Π (1 − δ_i)
            // with δ_i = 1 − |r_i| from the half roots, without the cancellation.
            let dl = |dp: f64, h: f64, i: f64| {
                if dp > 0.0 {
                    2.0 * h * i
                } else {
                    2.0 * hp.0 * i
                }
            };
            let (d1, d2, d3) = (dl(dpx, hx.0, ix), dl(dpy, hy.0, iy), dl(dpz, hz.0, iz));
            let ope = (d1 + d2 + d3) - (d1 * d2 + d1 * d3 + d2 * d3) + d1 * d2 * d3;
            rc1_near_m1(ope)
        } else {
            // e = Π r_i again, each ratio and the product carried with its error.
            let r = |(h, eh): (f64, f64), (g, eg): (f64, f64), i: f64| {
                let (n, en) = two_sum(hp.0, -h);
                let nl = en + hp.1 - eh;
                let q = n * i;
                (q, (res(q, g, n) + nl - q * eg) * i)
            };
            let (r1, l1) = r(hx, (gx, ex), ix);
            let (r2, l2) = r(hy, (gy, ey), iy);
            let (r3, l3) = r(hz, (gz, ez), iz);
            let (p12, f12) = two_prod(r1, r2);
            let l12 = f12 + mla(r1, l2, l1 * r2);
            let (e, fe) = two_prod(p12, r3);
            rc1_dd(e, fe + mla(p12, l3, l12 * r3))
        };
        let (t1, e1) = two_prod(rc, inv);
        let w = 0.125 * f;
        let (th, tl) = (w * t1, w * mla(rc, inv_lo, e1));
        let (s, t) = two_sum(sh, th);
        sh = s;
        sl += t + tl;
        x = x.quarter_plus(lq);
        y = y.quarter_plus(lq);
        z = z.quarter_plus(lq);
        p = p.quarter_plus(lq);
        a = 0.25 * a + lq.h;
        f *= 0.25;
    }
    let am = mean(&[x, y, z, p, p], 5.0);
    let xx = dx * f / am.h;
    let yy = dy * f / am.h;
    let zz = dz * f / am.h;
    let pp = -(xx + yy + zz) * 0.5;
    let p2 = pp * pp;
    let xyz = xx * yy * zz;
    let e2 = xx * yy + xx * zz + yy * zz - 3.0 * p2;
    let e3 = xyz + 2.0 * e2 * pp + 4.0 * p2 * pp;
    let e4 = (2.0 * xyz + e2 * pp + 3.0 * p2 * pp) * pp;
    let e5 = xyz * p2;
    let t = series_dj(e2, e3, e4, e5);
    let (qh, ql) = over_sqrt3(t, am);
    let (s3, e3) = two_prod(6.0, sh);
    let (h, l) = two_sum(s3, f * qh);
    h + (l + (e3 + 6.0 * sl + f * ql))
}

/// `R_J(x, y, z, p)` for `p < 0`: its Cauchy principal value (Carlson 1995, (2.22); the form of
/// Boost.Math), of finite non-negative `x, y, z` with at most one zero:
/// `(z + q) R_J(x, y, z, p) = (p' − z) R_J(x, y, z, p') − 3 R_F(x, y, z) + 3 √(xyz/W) R_C(W, p'q)`,
/// `q = −p`, `p' = (z(x + y + q) − xy)/(z + q)`, `W = xy + p'q`, `x ≤ y ≤ z`; the products are
/// formed relative to `max(y, q)` so that they cannot overflow or underflow.
pub(super) fn rj_pv(x: f64, y: f64, z: f64, p: f64) -> f64 {
    let (mut x, mut y, mut z) = (x, y, z);
    if x > y {
        core::mem::swap(&mut x, &mut y);
    }
    if y > z {
        core::mem::swap(&mut y, &mut z);
    }
    if x > y {
        core::mem::swap(&mut x, &mut y);
    }
    let q = -p;
    let zq = z + q;
    let pn = (x + y + q) * (z / zq) - x * (y / zq);
    let pz = -((z - x) * ((z - y) / zq)); // p' − z
    let s = y.max(q);
    let w = (x / s) * (y / s) + (pn / s) * (q / s); // W / s²
    let r = ((x / s) * (y / s) / w) * z; // xyz / W
    let rc = rc(w, (pn / s) * (q / s)) / s;
    let v = pz * rj(x, y, z, pn) - 3.0 * rf(x, y, z) + 3.0 * r.sqrt() * rc;
    v / zq
}
