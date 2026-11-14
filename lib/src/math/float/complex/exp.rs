//! The exponential, the logarithms and the argument, each part correctly rounded, with MPC's
//! special values (`mpc_exp`, `mpc_log`, `mpc_log10`, `mpc_arg`, `mpc_proj`).
//!
//! - `e^(x + yi) = e^x cos y + i·e^x sin y`: the products of the real functions of
//!   [`real`](super::real), each within a few units of `2^-w` relatively, of an exponent that may
//!   leave the range (a part then overflows or underflows by its mode). For `|x|, |y| < 2^-(p +
//!   8)`, `p` a part's precision, the parts lie within less than their last place of `1` and `y`:
//!   they are `1 + (e^x - 1) cos y - 2 sin²(y/2)` and `y + (e^x - 1) sin y + (sin y - y)`, each
//!   difference computed on its own, relatively, and rounded beside its lead
//!   ([`lead`](super::real::lead)).
//! - `ln z = ln|z| + i·arg z`: the argument is [`BigFloat::atan2_round`] (and the arctangent of a
//!   tiny dyadic ratio rounded beside it); `ln|z| = ln(x² + y²)/2`, and for a larger part of the
//!   exponent 0 or 1, where `|z|` may be near 1, `ln(1 + d)/2` with `d = x² + y² - 1` exact (or cut
//!   to the working precision when the parts lie far apart), so that nothing cancels.
//! - `log10 z = ln z / ln 10`, its real part exact when `x² + y²` is a power of 10 (`x` and `y`
//!   integers then).
//!
//! No other part is exact: by the Lindemann–Weierstrass theorem `e^z`'s parts are transcendental
//! for `z` not real and not imaginary, `ln|z| = q` would make `|z|² = e^(2q)` rational, and so on.

use super::BigComplex;
use super::raw::{self, Exact};
use super::real::{
    self, Part, add, add_exact, check, div, exactly, inf, ln, ln1p, ln1p_minus, ln10, mul,
    mul_exact, nan, nans, of, one, pair, range_step, set, shl, sin_cos, sqr, step, sub, top,
    zero_of,
};
use crate::math::float::exp::pow_u64;
use crate::math::float::sincos::Fx;
use crate::math::float::trig::near;
use crate::math::float::{BigFloat, Round, limbs};
use crate::math::numbers::Number;
use core::cmp::Ordering;

impl BigComplex {
    /// The exponential, each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_exp`. The special values: `e^(x ± 0i) = e^x ± 0i` for any
    /// `x`, NaN included; `e^(±0 + yi) = cos y + i sin y`; `e^(-∞ + yi) = ±0 ± 0i` and `e^(+∞ +
    /// yi) = ±∞ ± ∞i` with the signs of `cos y` and `sin y` for a finite `y` not zero; `e^(-∞ ±
    /// ∞i) = e^(-∞ + NaN·i) = +0 + 0i`, `e^(+∞ ± ∞i) = e^(+∞ + NaN·i) = +∞ + NaN·i`; NaN in both
    /// parts for the other numbers with a NaN or an infinite imaginary part. A part beyond the
    /// exponent range overflows or underflows by its mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (e, o) = z.exp_round(53, Round::Nearest);
    /// assert_eq!(*e.re(), BigFloat::from(1.4686939399158851)); // e cos 1
    /// assert_eq!(*e.im(), BigFloat::from(2.2873552871788423)); // e sin 1
    /// assert_eq!(o, (Ordering::Less, Ordering::Less));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn exp_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.exp_round_parts((prec, prec), (round, round))
    }

    /// [`exp_round`](BigComplex::exp_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::zero(53), BigFloat::pi(53));
    /// let (e, o) = z.exp_round_parts((24, 53), (Round::Down, Round::Up));
    /// assert_eq!(*e.re(), BigFloat::from(-1)); // cos π rounded down
    /// assert_eq!(o.0, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn exp_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(exp(self, check(prec), round))
    }

    /// The exponential, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(2), BigFloat::zero(53));
    /// assert_eq!(z.exp(), BigComplex::new(BigFloat::from(2).exp(), BigFloat::zero(53)));
    /// ```
    pub fn exp(&self) -> BigComplex {
        self.exp_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// The natural logarithm, principal branch, each part rounded to `prec` bits in the mode
    /// `round`, and the directions of the roundings, as MPC's `mpc_log`: `ln|z| + i·arg z`, the
    /// imaginary part in `[-π, π]`, its sign the sign of `self`'s imaginary part, signed zeros
    /// included, so that the branch cut along the negative real axis has `ln(-1 + 0i) = πi` and
    /// `ln(-1 - 0i) = -πi`. The special values: `ln(±0 ± 0i) = -∞ + i·arg`; a number with an
    /// infinite part gives `+∞ + i·arg` (`arg` of the infinities as `atan2`'s), and `+∞ + NaN·i`
    /// when its other part is NaN; NaN in both parts for the other numbers with a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(-1), BigFloat::zero(53));
    /// let (l, o) = z.ln_round(53, Round::Nearest);
    /// assert_eq!(l, BigComplex::new(BigFloat::zero(53), BigFloat::pi(53)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Less));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn ln_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.ln_round_parts((prec, prec), (round, round))
    }

    /// [`ln_round`](BigComplex::ln_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// let (l, _) = z.ln_round_parts((53, 24), (Round::Nearest, Round::Nearest));
    /// assert_eq!(*l.re(), BigFloat::from(5f64.ln()));
    /// assert_eq!(*l.im(), BigFloat::from(4f32.atan2(3.0)));
    /// ```
    #[track_caller]
    pub fn ln_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(log(self, check(prec), round, false))
    }

    /// The natural logarithm, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1.0));
    /// let half_pi = BigFloat::pi(53).mul_round(&BigFloat::from(0.5), 53, a::math::float::Round::Nearest).0;
    /// assert_eq!(z.ln(), BigComplex::new(BigFloat::zero(53), half_pi));
    /// ```
    pub fn ln(&self) -> BigComplex {
        self.ln_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// The decimal logarithm, `ln z / ln 10`, each part rounded to `prec` bits in the mode
    /// `round`, and the directions of the roundings, as MPC's `mpc_log10`: the branch cut and the
    /// special values of [`ln_round`](BigComplex::ln_round); the real part exact when `|z|²` is a
    /// power of 10.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(3)); // |z|² = 10
    /// let (l, o) = z.log10_round(53, Round::Nearest);
    /// assert_eq!(*l.re(), BigFloat::from(0.5));
    /// assert_eq!(o.0, Ordering::Equal);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn log10_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.log10_round_parts((prec, prec), (round, round))
    }

    /// [`log10_round`](BigComplex::log10_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(-100), BigFloat::zero(53));
    /// let (l, o) = z.log10_round_parts((10, 53), (Round::Nearest, Round::Up));
    /// assert_eq!(*l.re(), BigFloat::from(2));
    /// ```
    #[track_caller]
    pub fn log10_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(log(self, check(prec), round, true))
    }

    /// The decimal logarithm, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(6), BigFloat::from(8));
    /// assert_eq!(*z.log10().re(), BigFloat::from(1));
    /// ```
    pub fn log10(&self) -> BigComplex {
        self.log10_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// The argument `atan2(im, re)`, in `[-π, π]`, rounded to `prec` bits in the mode `round`,
    /// and the direction of the rounding, as MPC's `mpc_arg` (MPFR's `atan2` of the parts, with its
    /// special values).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(-1), BigFloat::neg_zero(53));
    /// let (a, o) = z.arg_round(53, Round::Nearest);
    /// assert_eq!(a, -BigFloat::pi(53));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn arg_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        arg_part(&self.im, &self.re, BigFloat::check_prec(prec), round)
    }

    /// The argument, to nearest at the larger precision of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1.0), BigFloat::from(1.0));
    /// assert_eq!(z.arg(), BigFloat::from(core::f64::consts::FRAC_PI_4));
    /// ```
    pub fn arg(&self) -> BigFloat {
        self.arg_round(self.max_prec(), Round::Nearest).0
    }

    /// The projection on the Riemann sphere, each part rounded to `prec` bits in the mode
    /// `round`, as MPC's `mpc_proj` and C99's `cproj`: a number with an infinite part, even with a
    /// NaN, goes to `+∞ ± 0i` with the sign of its imaginary part; any other is itself, rounded.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::nan(53), BigFloat::neg_infinity(53));
    /// let (p, _) = z.proj_round(53, Round::Nearest);
    /// assert_eq!(p, BigComplex::new(BigFloat::infinity(53), BigFloat::neg_zero(53)));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn proj_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.proj_round_parts((prec, prec), (round, round))
    }

    /// [`proj_round`](BigComplex::proj_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(1.75));
    /// let (p, o) = z.proj_round_parts((2, 2), (Round::Up, Round::Down));
    /// assert_eq!(p, BigComplex::new(BigFloat::from(3), BigFloat::from(1.5)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Less));
    /// ```
    #[track_caller]
    pub fn proj_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = check(prec);
        if self.re.is_infinite() || self.im.is_infinite() {
            let neg = self.im.is_sign_negative();
            return pair((exactly(inf(prec.0, false)), exactly(zero_of(prec.1, neg))));
        }
        pair((
            set(&self.re, prec.0, round.0),
            set(&self.im, prec.1, round.1),
        ))
    }

    /// The projection on the Riemann sphere, the parts as they are unless one is infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// assert_eq!(z.proj(), z);
    /// ```
    pub fn proj(&self) -> BigComplex {
        self.proj_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }
}

/// The exponential (see the [module](self)).
fn exp(z: &BigComplex, p: (u64, u64), r: (Round, Round)) -> (Part, Part) {
    let (x, y) = (&z.re, &z.im);
    if x.is_nan() || y.is_nan() {
        if y.is_zero() {
            return (exactly(nan(p.0)), set(y, p.1, r.1));
        }
        if x.is_infinite() {
            return if x.is_sign_negative() {
                (exactly(zero_of(p.0, false)), exactly(zero_of(p.1, false)))
            } else {
                (exactly(inf(p.0, false)), exactly(nan(p.1)))
            };
        }
        return nans(p);
    }
    if y.is_zero() {
        return (x.exp_round(p.0, r.0), set(y, p.1, r.1));
    }
    if x.is_zero() {
        return (y.cos_round(p.0, r.0), y.sin_round(p.1, r.1));
    }
    if x.is_infinite() {
        let minus = x.is_sign_negative();
        let n = |prec: u64, negative: bool| {
            if minus {
                zero_of(prec, negative)
            } else {
                inf(prec, negative)
            }
        };
        if y.is_infinite() {
            return if minus {
                (exactly(zero_of(p.0, false)), exactly(zero_of(p.1, false)))
            } else {
                (exactly(inf(p.0, false)), exactly(nan(p.1)))
            };
        }
        // The signs of `cos y` and `sin y`, neither zero for a finite `y` not zero.
        let ((s, _), (c, _)) = y.sin_cos_round(2, Round::Nearest);
        return (
            exactly(n(p.0, c.is_sign_negative())),
            exactly(n(p.1, s.is_sign_negative())),
        );
    }
    if y.is_infinite() {
        return nans(p);
    }
    let (fx, fy) = (of(x), of(y));
    real::ziv2(p, r, |w, need| {
        let tiny = |prec: u64| {
            let t = -i128::from(prec) - 8;
            top(&fx) < t && 2 * top(&fy) < t
        };
        let mut out = [None, None];
        if (need[0] && tiny(p.0)) || (need[1] && tiny(p.1)) {
            // `e^x - 1`, and `sin y`, `1 - cos y = 2 sin²(y/2)` from the half angle.
            let em = real::expm1(&fx, w).expect("a tiny exponent");
            let (sh, ch) = sin_cos(&shl(fy.clone(), -1), w);
            let s = shl(mul(&sh, &ch, w), 1);
            let omc = shl(sqr(&sh, w), 1);
            let c = sub(&one(), &omc, w);
            if need[0] {
                out[0] = Some(if tiny(p.0) {
                    let d = sub(&mul(&em, &c, w), &omc, w);
                    real::lead(&one(), &d, w, p.0, r.0)
                } else {
                    step(&add(&mul(&em, &c, w), &c, w))
                });
            }
            if need[1] {
                out[1] = Some(if tiny(p.1) {
                    let d = add(&mul(&em, &s, w), &real::odd_series(&fy, w, true, false), w);
                    real::lead(&fy, &d, w, p.1, r.1)
                } else {
                    step(&add(&mul(&em, &s, w), &s, w))
                });
            }
            return out;
        }
        let (s, c) = sin_cos(&fy, w);
        match real::exp(&fx, w) {
            Ok(e) => {
                if need[0] {
                    out[0] = Some(step(&mul(&e, &c, w)));
                }
                if need[1] {
                    out[1] = Some(step(&mul(&e, &s, w)));
                }
            }
            Err(range) => {
                // `|cos y|`, `|sin y|` are far above `2^-(2^61)` for any `y` whose reduction
                // can be computed, so the product lies beyond the range too.
                for (i, f) in [&c, &s].into_iter().enumerate() {
                    if need[i] && real::sign_known(f) {
                        out[i] = Some(range_step(range, f.negative, [p.0, p.1][i], [r.0, r.1][i]));
                    }
                }
            }
        }
        out
    })
}

/// `atan2(y, x)` rounded, as [`BigFloat::atan2_round`], and for `x > 0` and a tiny ratio `y/x`
/// that is a dyadic number `q`, `atan q` rounded beside `q` (`0 < q - atan q < q³/3 < 2^(3e - 1)`
/// for `|q| < 2^e`, toward zero), which Ziv's rounding would only reach at about `-2e` bits.
fn arg_part(y: &BigFloat, x: &BigFloat, prec: u64, round: Round) -> Part {
    if x.is_normal() && y.is_normal() && !x.is_sign_negative() {
        let (ex, ey) = (
            i128::from(x.exponent().expect("finite")),
            i128::from(y.exponent().expect("finite")),
        );
        if ey - ex < -i128::from(prec / 2) - 8 {
            let (Exact::Val(vy), Exact::Val(vx)) = (raw::exact(y), raw::exact(x)) else {
                unreachable!("finite and not zero")
            };
            if let Some(q) = raw::div_exact(vy.negative, &vy, &vx) {
                let e = q.top();
                if let Some(r) = near(q.negative, &q.m, q.lsb, false, 3 * e - 1, prec, round) {
                    return r;
                }
            }
        }
    }
    y.atan2_round(x, prec, round)
}

/// The exponent of a part, finite and not zero.
#[inline]
fn expo(x: &BigFloat) -> i64 {
    x.exponent().expect("finite and not zero")
}

/// `ln z`, or `log10 z` when `ten`.
fn log(z: &BigComplex, p: (u64, u64), r: (Round, Round), ten: bool) -> (Part, Part) {
    let (x, y) = (&z.re, &z.im);
    if !z.is_finite() {
        if x.is_nan() || y.is_nan() {
            let re = if x.is_infinite() || y.is_infinite() {
                inf(p.0, false)
            } else {
                nan(p.0)
            };
            return (exactly(re), exactly(nan(p.1)));
        }
        return (exactly(inf(p.0, false)), arg_of(y, x, p.1, r.1, ten));
    }
    if y.is_zero() {
        if x.is_zero() {
            return (exactly(inf(p.0, true)), arg_of(y, x, p.1, r.1, ten));
        }
        let re = real_log(&x.abs(), p.0, r.0, ten);
        if !x.is_sign_negative() {
            return (re, set(y, p.1, r.1));
        }
        return (re, arg_of(y, x, p.1, r.1, ten));
    }
    if x.is_zero() {
        return (
            real_log(&y.abs(), p.0, r.0, ten),
            arg_of(y, x, p.1, r.1, ten),
        );
    }
    let re = log_abs(x, y, p.0, r.0, ten);
    let im = arg_of(y, x, p.1, r.1, ten);
    (re, im)
}

/// `ln x` or `log10 x` of a real `x > 0`.
fn real_log(x: &BigFloat, prec: u64, round: Round, ten: bool) -> Part {
    if ten {
        x.log10_round(prec, round)
    } else {
        x.ln_round(prec, round)
    }
}

/// The imaginary part of the logarithm: the argument, over `ln 10` when `ten` (a zero is
/// itself).
fn arg_of(y: &BigFloat, x: &BigFloat, prec: u64, round: Round, ten: bool) -> Part {
    if !ten {
        return arg_part(y, x, prec, round);
    }
    let (a, _) = y.atan2_round(x, 2, Round::Nearest);
    if a.is_zero() || a.is_nan() {
        return exactly(a.prec_round(prec, round).0);
    }
    // The special points have a multiple `num/4` of π (`atan2`'s table), which the rounding to
    // 2 bits tells: 0.75, 1.5, 2 and 3 for `num` 1 to 4.
    let negative = a.is_sign_negative();
    let special = !(x.is_finite() && y.is_finite() && !x.is_zero() && !y.is_zero());
    let num = special.then(|| {
        let f = a.abs();
        if f < 1 {
            1u64
        } else if f < 2 {
            2
        } else if f < 3 {
            3
        } else {
            4
        }
    });
    let (fx, fy) = (
        (!x.is_zero() && x.is_finite()).then(|| of(x)),
        (!y.is_zero() && y.is_finite()).then(|| of(y)),
    );
    real::ziv1(prec, round, |w| {
        let angle = match num {
            Some(n) => shl(real::mul_int(&real::pi(w + 8), n), -2),
            None => match real::atan2(fy.as_ref().expect("y"), fx.as_ref().expect("x"), w + 8) {
                Some(a) => real::abs(a),
                None => return Ok(real::undecided()),
            },
        };
        let mut q = match div(&angle, &ln10(w + 8), w + 8) {
            Some(q) => q,
            None => return Ok(real::undecided()),
        };
        q.negative = negative;
        step(&q)
    })
}

/// `ln|z|` (or `log10|z|`) for `x`, `y` finite and not zero (see the [module](self)).
fn log_abs(x: &BigFloat, y: &BigFloat, prec: u64, round: Round, ten: bool) -> Part {
    if ten && let Some(m) = power_of_ten(x, y) {
        // `log10|z| = m/2`, exactly.
        let (v, _) = BigFloat::from_parts(&Number::from(m), -1, 64, Round::Nearest);
        return v.prec_round(prec, round);
    }
    let (ax, ay) = (real::abs(of(x)), real::abs(of(y)));
    let (big, small) = if expo(x) >= expo(y) {
        (ax, ay)
    } else {
        (ay, ax)
    };
    let k = expo(x).max(expo(y));
    // Near 1: `d = A² - 1 + B²`, the first sum exact.
    let near_one = k == 0 || k == 1;
    let d1 = near_one.then(|| add_exact(&mul_exact(&big, &big), &real::neg(one())));
    let b2 = mul_exact(&small, &small);
    real::ziv1(prec, round, |w| {
        let attempt = || -> Option<real::Step> {
            let half_ln = if let Some(d1) = &d1 {
                let d = if d1.m.is_empty() {
                    b2.clone()
                } else {
                    add(d1, &b2, w + 16)
                };
                let exact = d.err.0 == f64::NEG_INFINITY;
                if !ten && exact && 2 * top(&d) < -i128::from(prec) - 8 {
                    // `ln(1 + d)/2 = d/2 + (ln(1 + d) - d)/2`, beside `d/2`.
                    let delta = shl(ln1p_minus(&d, w)?, -1);
                    return Some(real::lead(&shl(d, -1), &delta, w, prec, round));
                }
                shl(ln1p(&d, w)?, -1)
            } else {
                let n = add(&sqr(&big, w + 8), &b2_cut(&b2, w), w + 8);
                shl(ln(&n, w)?, -1)
            };
            let v = if ten {
                div(&half_ln, &ln10(w + 8), w + 8)?
            } else {
                half_ln
            };
            Some(step(&v))
        };
        attempt().unwrap_or(Ok(real::undecided()))
    })
}

/// `B²` cut to the working precision.
#[inline]
fn b2_cut(b2: &Fx, w: u64) -> Fx {
    b2.clone().cut(w + 32)
}

/// `m` when `x² + y² = 10^m` (`x`, `y` finite, not both zero): then `x` and `y` are integers
/// (`x = u/2^e`, `y = v/2^e` would need `u² + v² = 10^m·4^e`, so `u` and `v` even, down to `e =
/// 0`), `m` is the exponent of 2 in `x² + y²` (of `4^min(a, b)` for the exponents `a ≠ b` of 2 in
/// `x` and `y`, `2^(2a + 1)` for `a = b`, as the odd parts' squares add to 2 modulo 4), and `10^m`
/// has the length of `x² + y²`; and when these hold, it is tested exactly.
fn power_of_ten(x: &BigFloat, y: &BigFloat) -> Option<u64> {
    let v2 = |f: &BigFloat| -> Option<(i128, i128)> {
        if f.is_zero() {
            return None;
        }
        let (_, m, lsb) = raw::significant(f);
        Some((
            lsb + i128::from(limbs::trailing_zeros(m)),
            i128::from(expo(f)),
        ))
    };
    let (a, b) = (v2(x), v2(y));
    for (low, _) in [a, b].into_iter().flatten() {
        if low < 0 {
            return None;
        }
    }
    let t = match (a, b) {
        (Some((a, _)), Some((b, _))) if a == b => 2 * a + 1,
        (Some((a, _)), Some((b, _))) => 2 * a.min(b),
        (Some((a, _)), None) | (None, Some((a, _))) => 2 * a,
        (None, None) => return None,
    };
    let e = a
        .map_or(i128::MIN, |v| v.1)
        .max(b.map_or(i128::MIN, |v| v.1));
    // `x² + y²` lies in `[2^(2e - 2), 2^(2e + 1))`, `10^t` in `[2^(3.32t - 1), 2^(3.33t)]`.
    let len = t as f64 * core::f64::consts::LOG2_10;
    if (len - 2.0 * e as f64).abs() > 4.0 || t > 1 << 40 {
        return None;
    }
    let n = match raw::sum_exact(&raw::square(x), &raw::square(y), Round::Nearest) {
        Exact::Val(v) => v,
        Exact::Zero(_) => return None,
    };
    // `n = 10^t = 5^t·2^t`: its odd part `5^t` and the exponent `t` of its 2s.
    let tz = limbs::trailing_zeros(&n.m);
    if n.lsb + i128::from(tz) != t {
        return None;
    }
    let mut odd = limbs::shr(&n.m, tz);
    limbs::trim(&mut odd);
    let mut five = limbs::of_number(&pow_u64(&Number::from(5), t as u64));
    limbs::trim(&mut five);
    (odd == five).then_some(t as u64)
}
