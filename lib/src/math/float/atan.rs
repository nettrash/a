//! The inverse circular functions, correctly rounded, with MPFR's special values: `atan`, `asin`
//! and `acos`, and `atan2` with the whole table of C99's Annex F for zeros and infinities.
//!
//! Each comes to the arctangent of a `z` in `(0, 1]` (see [`sincos::atan`]) and a multiple of
//! `π/2`: `atan x = π/2 - atan(1/x)` for `x > 1`; `asin x = atan(x/s)` or `π/2 - atan(s/x)` with
//! `s = sqrt(1 - x²)`, whichever ratio is at most 1, and `acos x` likewise, `atan(s/x)` near 1;
//! `1 - x²` is exact, so that no cancellation is left near `±1`. No sum of a multiple of `π/2`
//! and the arctangent cancels: the sums that are not the arctangent alone are above `π/4`.

use super::arith::round_for;
use super::consts::pi_approx;
use super::limbs;
use super::short;
use super::sincos::{self, Fx, Lg};
use super::trig::{approx, near, ziv_fast as ziv};
use super::ziv::Approx;
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// A result that is exact: `x` and [`Ordering::Equal`].
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// `±(num/4)·π` rounded, `num` 1, 2, 4 or 3: π rounded and its exponent moved, exactly, or for
/// `3π/4` Ziv's rounding of `3·P·2^lp` within three times the error of π.
fn pi_part(num: u64, negative: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let r = round_for(round, negative);
    let (mut y, o) = if num == 3 {
        ziv(prec, r, |w| {
            let p = pi_approx(w);
            Approx {
                negative: false,
                m: sincos::mul_small(&p.m, 3),
                lsb: p.lsb - 2,
                err: p.err,
            }
        })
    } else {
        let (mut p, o) = BigFloat::pi_round(prec, r);
        if let Kind::Finite { exp, .. } = &mut p.kind {
            *exp += match num {
                1 => -2,
                2 => -1,
                _ => 0,
            };
        }
        (p, o)
    };
    if negative {
        y.set_sign(true);
        (y, o.reverse())
    } else {
        (y, o)
    }
}

/// π times `2^shift` as an [`Fx`] of about `w` bits after the point.
fn pi_fx(w: u64, shift: i128) -> Fx {
    let p = pi_approx(w);
    Fx {
        negative: false,
        m: p.m,
        lsb: p.lsb + shift,
        err: Lg::pow2(p.err - p.lsb),
    }
}

/// `atan(z)` for `0 < z <= 1` to about `w` bits relatively: the fixed-point limbs of the kernel
/// `w + 24` bits for the series of a small `z`, and `2K` more for Newton's iteration, whose sine
/// loses `K` bits of the absolute error and whose result may be as small as `2^-K`.
fn atan_z(z: &Fx, w: u64) -> Fx {
    let f1 = w + 24;
    let k = sincos::halvings(f1.div_ceil(64) * 64);
    // The short sine is within a few ulps, absolutely: `K` bits suffice for a result above
    // `2^-K`.
    let f = if z.exponent() <= -k {
        f1
    } else if (f1 + k as u64).div_ceil(64) as usize <= short::MAX_LIMBS {
        f1 + k as u64
    } else {
        f1 + 2 * k as u64
    };
    sincos::atan(z, f.div_ceil(64) as usize)
}

/// `atan(z)` for `0 < z <= 1`, an [`Fx`] whose error passes through (`|atan'| <= 1`), to about `w`
/// bits relatively: the kernel of [`BigFloat::atan2_round`], for the complex functions.
pub(super) fn atan_unit(z: &Fx, w: u64) -> Fx {
    atan_z(z, w)
}

/// `π/2 - a`, `π - a` or `π/2 + a` for `part` 1, 2 or 3, `a` itself for 0, for an arctangent `0 <=
/// a <= π/4` (see [`with_pi`]): the quadrants of [`BigFloat::atan2_round`], for the complex
/// functions.
pub(super) fn plus_pi_part(a: Fx, part: u8, w: u64) -> Fx {
    with_pi(a, part, w)
}

/// The order of the values of two [`Fx`]s that are not negative.
fn cmp(a: &Fx, b: &Fx) -> Ordering {
    let l = a.lsb.min(b.lsb);
    let x = limbs::shl(&a.m, (a.lsb - l) as u64);
    let y = limbs::shl(&b.m, (b.lsb - l) as u64);
    limbs::cmp(&x, &y)
}

/// `π/2 - a`, `π - a`, `π/2 + a` or `a` alone, for `part` 1, 2, 3 and 0: the arctangent `a` and
/// π at the bits after the point it has, plus 8.
fn with_pi(a: Fx, part: u8, w: u64) -> Fx {
    if part == 0 {
        return a;
    }
    // The sum is above `π/4`: `a` is needed to `w + 72` bits after the point at most.
    let low = -(i128::from(w) + 72);
    let mut a = if a.lsb < low {
        let keep = (a.exponent() - low).max(0) as u64;
        a.cut(keep)
    } else {
        a
    };
    let p = pi_fx(
        w + 80 + (-a.lsb).max(0) as u64,
        if part == 2 { 0 } else { -1 },
    );
    a.negative = part != 3;
    sincos::sum(&[&p, &a])
}

/// `a/b` for magnitudes held as [`Fx`]s, to about `w + 64` bits, or `None` when nothing is known.
fn ratio(a: &Fx, b: &Fx, w: u64) -> Option<Fx> {
    sincos::quotient(a, b, w + 64)
}

/// `s = sqrt(1 - x²)` for `x = M·2^lsb`, `0 < |x| < 1`, as an [`Fx`] of about `w + 8` bits: 1 with
/// an error below `x² < 2^(2e)` when that is below `2^-(w + 8)`; otherwise the integer square root
/// of `1 - x² = (2^(-2lsb) - M²)·2^(2lsb)`, exact, shifted to the bits, within a unit of its last
/// place. For `|x| < 1/2`, `x` is cut to `w + 64` bits first, which moves `s` by at most as much
/// as `x` (`|ds/dx| = |x|/s < 0.58`).
fn sqrt_one_minus_square(m: &[u64], lsb: i128, w: u64) -> Fx {
    let e = lsb + i128::from(limbs::bit_len(m));
    let bits = w + 8;
    if 2 * e < -i128::from(bits) {
        return Fx {
            negative: false,
            m: vec![1],
            lsb: 0,
            err: Lg::pow2(2 * e),
        };
    }
    let mut x = Fx::exact(m.to_vec(), lsb);
    if e <= -1 {
        x = x.cut(w + 64);
    }
    let l = x.lsb;
    debug_assert!(l < 0);
    let d = limbs::sub(&limbs::shl(&[1], (-2 * l) as u64), &limbs::mul(&x.m, &x.m));
    let db = limbs::bit_len(&d) as i128;
    let t = ((2 * i128::from(bits) - db).max(0) + 1) / 2;
    let s = sincos::isqrt(&limbs::shl(&d, (2 * t) as u64));
    let lsb_s = l - t;
    // The cut of `x`, in units of `2^lsb_s`, plus the floor of the root.
    let err = x.err.scale(x.lsb - lsb_s).add(Lg(0.0));
    Fx {
        negative: false,
        m: s,
        lsb: lsb_s,
        err,
    }
}

/// The inverse functions of one argument.
#[derive(Clone, Copy, PartialEq, Eq)]
enum G {
    Atan,
    Asin,
    Acos,
}

impl G {
    /// The approximation of the function of `x = M·2^lsb` (`x` above zero, but below zero for
    /// `acos`, whose `negative` says so) to about `w` bits relatively.
    fn of(self, negative: bool, m: &[u64], lsb: i128, w: u64) -> Approx {
        let x = Fx::exact(m.to_vec(), lsb);
        let e = x.exponent();
        let (z, part) = match self {
            G::Atan => {
                if e <= 0 || (e == 1 && limbs::trailing_zeros(m) == limbs::bit_len(m) - 1) {
                    (Some(x), 0)
                } else {
                    (ratio(&Fx::exact(vec![1], 0), &x, w), 1)
                }
            }
            G::Asin | G::Acos => {
                let s = sqrt_one_minus_square(m, lsb, w + 64);
                let big = cmp(&x, &s) != Ordering::Less;
                let z = if big {
                    ratio(&s, &x, w)
                } else {
                    ratio(&x, &s, w)
                };
                let part = match (self, negative, big) {
                    (G::Asin, _, false) => 0,
                    (G::Asin, _, true) => 1,
                    (_, false, true) => 0,
                    (_, false, false) => 1,
                    (_, true, true) => 2,
                    (_, true, false) => 3,
                };
                (z, part)
            }
        };
        let Some(z) = z else {
            return undecided();
        };
        if z.m.is_empty() {
            return undecided();
        }
        let a = atan_z(&z, w);
        let r = with_pi(a, part, w);
        approx(r.negative, &r)
    }
}

/// An approximation that decides nothing.
fn undecided() -> Approx {
    Approx {
        negative: false,
        m: Vec::new(),
        lsb: 0,
        err: 0,
    }
}

impl BigFloat {
    /// The arctangent rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: NaN for NaN, `±0` for `±0`, `±π/2` rounded for `±∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (a, o) = BigFloat::from(1).atan_round(53, Round::Nearest);
    /// assert_eq!(a, BigFloat::from(core::f64::consts::FRAC_PI_4));
    /// assert_eq!(o, Ordering::Less);
    /// assert_eq!(BigFloat::infinity(24).atan(), BigFloat::from(core::f32::consts::FRAC_PI_2));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn atan_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => exact(BigFloat::nan(prec)),
            Kind::Zero { negative } => exact(BigFloat::special(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Inf { negative } => pi_part(2, *negative, prec, round),
            Kind::Finite { .. } => {
                let (neg, m, lsb) = self.parts().expect("finite");
                let e = lsb + i128::from(limbs::bit_len(m));
                // `0 < x - atan x < x³/3 < 2^(3e - 1)` for `0 < x < 1`.
                if e <= -1
                    && let Some(r) = near(neg, m, lsb, false, 3 * e - 1, prec, round)
                {
                    return r;
                }
                let r = round_for(round, neg);
                let (mut y, o) = ziv(prec, r, |w| G::Atan.of(false, m, lsb, w));
                if neg {
                    y.set_sign(true);
                    (y, o.reverse())
                } else {
                    (y, o)
                }
            }
        }
    }

    /// The arctangent, to nearest at the precision of `self`.
    pub fn atan(&self) -> BigFloat {
        self.atan_round(self.prec, Round::Nearest).0
    }

    /// The arcsine rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// NaN for NaN and beyond `[-1, 1]`, `±0` for `±0`, `±π/2` rounded for `±1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let a = BigFloat::from(0.5).asin_round(53, Round::Nearest).0;
    /// assert_eq!(a, BigFloat::from(core::f64::consts::FRAC_PI_6));
    /// assert!(BigFloat::from(2).asin().is_nan());
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn asin_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { .. } => exact(BigFloat::nan(prec)),
            Kind::Zero { negative } => exact(BigFloat::special(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Finite { .. } => {
                let (neg, m, lsb) = self.parts().expect("finite");
                let e = lsb + i128::from(limbs::bit_len(m));
                let power = limbs::trailing_zeros(m) == limbs::bit_len(m) - 1;
                if e > 1 || e == 1 && !power {
                    return exact(BigFloat::nan(prec));
                }
                if e == 1 {
                    return pi_part(2, neg, prec, round);
                }
                // `0 < asin x - x < 0.18x³ < 2^(3e - 2)` for `0 < x < 1/2`.
                if e <= -1
                    && let Some(r) = near(neg, m, lsb, true, 3 * e - 2, prec, round)
                {
                    return r;
                }
                let r = round_for(round, neg);
                let (mut y, o) = ziv(prec, r, |w| G::Asin.of(false, m, lsb, w));
                if neg {
                    y.set_sign(true);
                    (y, o.reverse())
                } else {
                    (y, o)
                }
            }
        }
    }

    /// The arcsine, to nearest at the precision of `self`.
    pub fn asin(&self) -> BigFloat {
        self.asin_round(self.prec, Round::Nearest).0
    }

    /// The arccosine rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: NaN for NaN and beyond `[-1, 1]`, `+0` for 1 (exact), π rounded for -1, `π/2`
    /// rounded for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (a, o) = BigFloat::from(-1).acos_round(53, Round::Up);
    /// assert_eq!(a, BigFloat::pi_round(53, Round::Up).0);
    /// assert_eq!(o, Ordering::Greater);
    /// assert!(BigFloat::from(1).acos().is_zero());
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn acos_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { .. } => exact(BigFloat::nan(prec)),
            Kind::Zero { .. } => pi_part(2, false, prec, round),
            Kind::Finite { .. } => {
                let (neg, m, lsb) = self.parts().expect("finite");
                let e = lsb + i128::from(limbs::bit_len(m));
                let power = limbs::trailing_zeros(m) == limbs::bit_len(m) - 1;
                if e > 1 || e == 1 && !power {
                    return exact(BigFloat::nan(prec));
                }
                if e == 1 {
                    return if neg {
                        pi_part(4, false, prec, round)
                    } else {
                        exact(BigFloat::zero(prec))
                    };
                }
                ziv(prec, round, |w| G::Acos.of(neg, m, lsb, w))
            }
        }
    }

    /// The arccosine, to nearest at the precision of `self`.
    pub fn acos(&self) -> BigFloat {
        self.acos_round(self.prec, Round::Nearest).0
    }

    /// The angle of the point `(x, self)`, the arctangent of `self / x` in `[-π, π]` by the
    /// quadrant of the point, rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding; with the values of C99's Annex F and MPFR at zeros and infinities, the sign of
    /// the result that of `self`: `atan2(±0, x)` is `±0` for `x = +0` or above and `±π` for `x =
    /// -0` or below; `atan2(y, ±0)` is `±π/2` for `y` not zero; `atan2(±∞, +∞) = ±π/4`,
    /// `atan2(±∞, -∞) = ±3π/4`, `atan2(±∞, x) = ±π/2` for a finite `x`; `atan2(y, +∞) = ±0` and
    /// `atan2(y, -∞) = ±π` for a finite `y` not zero; NaN when either is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let y = BigFloat::from(-1);
    /// let a = y.atan2_round(&BigFloat::from(-1), 53, Round::Nearest).0;
    /// assert_eq!(a, BigFloat::from(-3.0 * core::f64::consts::FRAC_PI_4));
    /// let z = BigFloat::neg_zero(53).atan2(&BigFloat::neg_zero(53));
    /// assert_eq!(z, -BigFloat::pi(53));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn atan2_round(&self, x: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if self.is_nan() || x.is_nan() {
            return exact(BigFloat::nan(prec));
        }
        let ny = self.is_sign_negative();
        let nx = x.is_sign_negative();
        match (&self.kind, &x.kind) {
            (Kind::Zero { .. }, _) => {
                if nx {
                    pi_part(4, ny, prec, round)
                } else {
                    exact(BigFloat::special(prec, Kind::Zero { negative: ny }))
                }
            }
            (Kind::Inf { .. }, Kind::Inf { .. }) => {
                pi_part(if nx { 3 } else { 1 }, ny, prec, round)
            }
            (Kind::Inf { .. }, _) | (_, Kind::Zero { .. }) => pi_part(2, ny, prec, round),
            (_, Kind::Inf { .. }) => {
                if nx {
                    pi_part(4, ny, prec, round)
                } else {
                    exact(BigFloat::special(prec, Kind::Zero { negative: ny }))
                }
            }
            _ => {
                let (_, my, ly) = self.parts().expect("finite");
                let (_, mx, lx) = x.parts().expect("finite");
                let (y, x) = (Fx::exact(my.to_vec(), ly), Fx::exact(mx.to_vec(), lx));
                let big = cmp(&y, &x) == Ordering::Greater;
                let r = round_for(round, ny);
                let (mut v, o) = ziv(prec, r, |w| {
                    let z = if big {
                        ratio(&x, &y, w)
                    } else {
                        ratio(&y, &x, w)
                    };
                    let Some(z) = z.filter(|z| !z.m.is_empty()) else {
                        return undecided();
                    };
                    let a = atan_z(&z, w);
                    let part = match (big, nx) {
                        (false, false) => 0,
                        (false, true) => 2,
                        (true, false) => 1,
                        (true, true) => 3,
                    };
                    let r = with_pi(a, part, w);
                    approx(r.negative, &r)
                });
                if ny {
                    v.set_sign(true);
                    (v, o.reverse())
                } else {
                    (v, o)
                }
            }
        }
    }

    /// The angle of the point `(x, self)`, to nearest at the precision of `self`.
    pub fn atan2(&self, x: &BigFloat) -> BigFloat {
        self.atan2_round(x, self.prec, Round::Nearest).0
    }
}
