//! `ln Γ(z)` and `Γ(z)` of complex arguments.
//!
//! The principal branch of `ln Γ`, carried in complex double-double: Stirling's series for
//! `|z| ≥ 10` in the right half-plane, the recurrence `ln Γ(z) = ln Γ(z + n) − ln Π(z + k)` below
//! it (the branch of the product's logarithm kept by counting its crossings of the negative axis,
//! as in Hare's method and SciPy's `loggamma`), Taylor series at the zeros 1 and 2, and for
//! `Re z < 0` the reflection formula
//! `ln Γ(z) = ln 2π − iπ/2 + iπz − ln(1 − e^{2πiz}) − ln Γ(1 − z)` (`Im z > 0`; the lower
//! half-plane by conjugation). `Γ(z) = e^{ln Γ(z)}` with the exponential and the sine and cosine
//! of the double-double logarithm, so that its relative error stays near ε even where `|ln Γ|` is
//! large.

use super::super::dd::{Dd, two_sum};
use super::ddm::*;
use super::real::{gamma_f64, lgamma_f64};
use super::tab::*;
use crate::math::complex::Complex;

const PI_DD: Dd = Dd {
    hi: PI_HI,
    lo: PI_LO,
};

const LN_SQRT_2PI: Dd = Dd {
    hi: LN_SQRT_2PI_HI,
    lo: LN_SQRT_2PI_LO,
};

/// `ln 2π`.
const LN_2PI: Dd = Dd {
    hi: 1.8378770664093456,
    lo: -7.756588316134483e-17,
};

/// `ln 2`.
const LN2: Dd = Dd {
    hi: core::f64::consts::LN_2,
    lo: 2.3190468138462996e-17,
};

/// A complex double-double.
#[derive(Clone, Copy, Debug)]
struct Cdd {
    re: Dd,
    im: Dd,
}

impl Cdd {
    #[inline]
    fn mul(self, b: Cdd) -> Cdd {
        Cdd {
            re: self.re.mul(b.re).sub(self.im.mul(b.im)),
            im: self.re.mul(b.im).add(self.im.mul(b.re)),
        }
    }

    #[inline]
    fn sub(self, b: Cdd) -> Cdd {
        Cdd {
            re: self.re.sub(b.re),
            im: self.im.sub(b.im),
        }
    }

    #[inline]
    fn add_re(self, k: f64) -> Cdd {
        Cdd {
            re: self.re.add_f64(k),
            im: self.im,
        }
    }
}

/// `θ/π` with `θ = arg(x + iy)`, as a double-double, for a finite nonzero `x + iy`: the rounded
/// `atan2` corrected by one Newton step with the double-double sine and cosine.
fn arg_over_pi(x: Dd, y: Dd) -> Dd {
    let t0 = y.hi.atan2(x.hi);
    let qh = t0 / PI_HI;
    let r = rem(qh, PI_HI, t0) - qh * PI_LO;
    let ql = r / PI_HI;
    let s = sin_pi_dd(qh);
    let c = cos_pi_dd(qh);
    let d = PI_HI * ql;
    let s1 = s.add_f64(d * c.hi);
    let c1 = c.add_f64(-d * s.hi);
    // tan(θ − θ₀) = (y cos θ₀ − x sin θ₀)/(x cos θ₀ + y sin θ₀).
    let num = y.mul(c1).sub(x.mul(s1));
    let den = x.hi * c1.hi + y.hi * s1.hi;
    Dd::new(qh).add_f64(ql + num.value() / den / PI_HI)
}

/// `2^e` for any `e` in `−2044..=2046`, as two factors in range.
#[inline]
fn pow2_pair(e: i32) -> (f64, f64) {
    let a = e / 2;
    (pow2i(a), pow2i(e - a))
}

/// The principal `ln w` of a nonzero finite complex double-double.
fn ln_c(w: Cdd) -> Cdd {
    let m = w.re.hi.abs().max(w.im.hi.abs());
    // ⌊log₂ m⌋, then x² + y² of the scaled parts (no overflow or underflow).
    let k = if m < f64::MIN_POSITIVE {
        ((m * 18_446_744_073_709_551_616.0).to_bits() >> 52) as i32 - 1023 - 64
    } else {
        (m.to_bits() >> 52) as i32 - 1023
    };
    let (s1, s2) = pow2_pair(-k);
    let sc = |d: Dd| Dd {
        hi: d.hi * s1 * s2,
        lo: d.lo * s1 * s2,
    };
    let (xs, ys) = (sc(w.re), sc(w.im));
    let n2 = xs.mul(xs).add(ys.mul(ys));
    let l = ln_of_dd(n2);
    let re = Dd {
        hi: 0.5 * l.hi,
        lo: 0.5 * l.lo,
    }
    .add(LN2.mul_f64(k as f64));
    let im = PI_DD.mul(arg_over_pi(w.re, w.im));
    Cdd { re, im }
}

/// Stirling's series of `ln Γ(w)` for `|w| ≥ 10`, `Re w ≥ 0`.
fn stirling_c(w: Cdd) -> Cdd {
    let l = ln_c(w);
    let xm = w.re.add_f64(-0.5);
    let re = xm.mul(l.re).sub(w.im.mul(l.im)).sub(w.re).add(LN_SQRT_2PI);
    let im = w.im.mul(l.re).add(xm.mul(l.im)).sub(w.im);
    // Σ B₂ₖ/(2k(2k − 1) w^{2k−1}) in complex f64.
    let r = Complex::new(1.0, 0.0) / Complex::new(w.re.hi, w.im.hi);
    let u = r * r;
    let mut s = Complex::new(STIRLING_B[STIRLING_B.len() - 1], 0.0);
    for &b in STIRLING_B.iter().rev().skip(1) {
        s = s * u + Complex::new(b, 0.0);
    }
    let s = s * r;
    Cdd {
        re: re.add_f64(s.re),
        im: im.add_f64(s.im),
    }
}

/// `ln Γ(w)` for `Re w ≥ 0`, `Im w > 0`, by Stirling's series after shifting `|w|` past 10.
fn lgamma_right(w: Cdd) -> Cdd {
    let (x, y) = (w.re.hi, w.im.hi);
    if x * x + y * y >= 100.0 {
        return stirling_c(w);
    }
    let n = ((100.0 - y * y).sqrt() - x).ceil().max(1.0) as i32;
    // P = w (w + 1) … (w + n − 1); arg P passes π (Im P turns negative) at most once a step.
    let mut p = w;
    let mut neg = false;
    let mut flips = 0;
    for k in 1..n {
        p = p.mul(w.add_re(k as f64));
        let now = p.im.hi < 0.0;
        if now && !neg {
            flips += 1;
        }
        neg = now;
    }
    let g = stirling_c(w.add_re(n as f64)).sub(ln_c(p));
    Cdd {
        re: g.re,
        im: g.im.sub(PI_DD.mul_f64(2.0 * flips as f64)),
    }
}

/// `Σ c[k] w^(k+1)` in complex f64.
fn taylor(w: Complex<f64>, c: &[f64]) -> Complex<f64> {
    let mut s = Complex::new(c[c.len() - 1], 0.0);
    for &ck in c.iter().rev().skip(1) {
        s = s * w + Complex::new(ck, 0.0);
    }
    s * w
}

/// `ln Γ(z)` for `Im z > 0`, as a complex double-double.
fn lgamma_upper(z: Complex<f64>) -> Cdd {
    let (x, y) = (z.re, z.im);
    let d1 = Complex::new(x - 1.0, y);
    if d1.norm_sqr() < 0.0625 {
        let t = taylor(d1, &LGAMMA1_TAYLOR);
        return Cdd {
            re: Dd::new(t.re),
            im: Dd::new(t.im),
        };
    }
    let d2 = Complex::new(x - 2.0, y);
    if d2.norm_sqr() < 0.0625 {
        let t = taylor(d2, &LGAMMA2_TAYLOR);
        return Cdd {
            re: Dd::new(t.re),
            im: Dd::new(t.im),
        };
    }
    if x >= 0.0 {
        return lgamma_right(Cdd {
            re: Dd::new(x),
            im: Dd::new(y),
        });
    }
    // Reflection: ln Γ(z) = ln 2π − iπ/2 + iπz − ln(1 − e^{2πiz}) − ln Γ(1 − z), where
    // ln Γ(1 − z) is the conjugate of ln Γ(1 − x + iy).
    let (h, l) = two_sum(1.0, -x);
    let g = lgamma_right(Cdd {
        re: Dd { hi: h, lo: l },
        im: Dd::new(y),
    });
    // 1 − e^{2πiz} = 1 − e^{−2πy}(cos 2πx + i sin 2πx).
    let e = {
        let a = PI_DD.mul_f64(-2.0 * y);
        if a.hi < -708.0 {
            Dd::new(0.0)
        } else {
            exp_dd(a)
        }
    };
    let one_minus = Cdd {
        re: Dd::new(1.0).sub(e.mul(cos_pi_dd(2.0 * x))),
        im: e.mul(sin_pi_dd(2.0 * x)).neg(),
    };
    let lw = ln_c(one_minus);
    let re = LN_2PI.sub(PI_DD.mul_f64(y)).sub(lw.re).sub(g.re);
    // −π/2 + πx = π(x − 1/2).
    let im = PI_DD.mul(Dd::new(x).add_f64(-0.5)).sub(lw.im).add(g.im);
    Cdd { re, im }
}

/// The principal branch of `ln Γ(z)` of a `Complex<f64>`.
pub(crate) fn lgamma_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if !(x.is_finite() && y.is_finite()) {
        if x.is_nan() || y.is_nan() {
            return Complex::new(f64::NAN, f64::NAN);
        }
        if y.is_infinite() && x.is_finite() {
            return Complex::new(f64::NEG_INFINITY, y);
        }
        if x == f64::INFINITY {
            return Complex::new(
                f64::INFINITY,
                if y == 0.0 {
                    y
                } else {
                    f64::INFINITY.copysign(y)
                },
            );
        }
        return Complex::new(f64::NAN, f64::NAN);
    }
    if y == 0.0 {
        if x > 0.0 {
            return Complex::new(lgamma_f64(x), y);
        }
        if x == x.floor() {
            return Complex::new(f64::INFINITY, f64::NAN);
        }
        // On the cut: the limit from the side of the sign of the zero, −π⌈−x⌉ from above.
        let k = (-x).ceil();
        let im = -PI_HI * k;
        return Complex::new(lgamma_f64(x), if y.is_sign_negative() { -im } else { im });
    }
    if y > 0.0 {
        let g = lgamma_upper(z);
        Complex::new(g.re.value(), g.im.value())
    } else {
        let g = lgamma_upper(z.conj());
        Complex::new(g.re.value(), -g.im.value())
    }
}

/// `Γ(z)` of a `Complex<f64>`.
pub(crate) fn gamma_c64(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    if !(x.is_finite() && y.is_finite()) {
        if x.is_nan() || y.is_nan() {
            return Complex::new(f64::NAN, f64::NAN);
        }
        if y.is_infinite() && x.is_finite() {
            return Complex::new(0.0, 0.0);
        }
        if x == f64::INFINITY && y == 0.0 {
            return Complex::new(f64::INFINITY, 0.0);
        }
        return Complex::new(f64::NAN, f64::NAN);
    }
    if y == 0.0 {
        return Complex::new(gamma_f64(x), y);
    }
    let g = lgamma_upper(if y > 0.0 { z } else { z.conj() });
    let im = if y > 0.0 { g.im } else { g.im.neg() };
    let (s, c) = (sin_of(im), cos_of(im));
    if g.re.hi > 1500.0 {
        // Far beyond the range: the infinities in the directions of the phase.
        return Complex::new(f64::INFINITY * c.hi, f64::INFINITY * s.hi);
    }
    if g.re.hi < -1500.0 {
        return Complex::new(0.0 * c.hi, 0.0 * s.hi);
    }
    let (d, m) = exp_parts(g.re);
    Complex::new(scale(d.mul(c), m), scale(d.mul(s), m))
}

/// `θ/π` of a double-double angle `θ`, as a double-double.
fn over_pi(t: Dd) -> Dd {
    let qh = t.hi / PI_HI;
    let r = rem(qh, PI_HI, t.hi) - qh * PI_LO + t.lo;
    Dd::new(qh).add_f64(r / PI_HI)
}

/// `sin θ` of a double-double angle, to about `2⁻⁶⁰` relative.
fn sin_of(t: Dd) -> Dd {
    let q = over_pi(t);
    sin_pi_dd(q.hi).add_f64(PI_HI * q.lo * cos_pi_dd(q.hi).hi)
}

/// `cos θ` of a double-double angle, to about `2⁻⁶⁰` relative.
fn cos_of(t: Dd) -> Dd {
    let q = over_pi(t);
    cos_pi_dd(q.hi).add_f64(-PI_HI * q.lo * sin_pi_dd(q.hi).hi)
}
