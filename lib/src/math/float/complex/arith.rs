//! `+`, `-`, `*`, `/`, the square, the norm `|z|²` and the absolute value `|z|`, each part of
//! each result correctly rounded on its own.
//!
//! A product's parts `ac - bd` and `ad + bc` are each the exact sum of two exact products,
//! rounded once; a quotient's `(ac + bd)/(c² + d²)` and `(bc - ad)/(c² + d²)` are each an exact
//! quotient of exact sums, rounded once, or, when the exact sums would be long (parts of very
//! different exponents) or the result has 2000 bits or more, Ziv's approximation from sums cut
//! to a working precision, with the exact quotient after it when two attempts do not decide (an
//! exact or halfway result). The norm is the exact sum of the exact squares, rounded once, and
//! the absolute value its square root, rounded once from the exact sum or from one that only a
//! sticky bit stands for.
//!
//! An exact zero part follows IEEE 754's sums: two zeros of one sign give that zero, zeros of
//! both signs or terms that cancel `+0`, or `-0` when that part rounds down.

use super::raw::{
    Exact, Val, div_trunc, prod, round_exact, square, sum_exact, sum_len, sum_round, sum_trunc,
    zero,
};
use super::{BigComplex, Ternary};
use crate::math::float::sum::{Term, sqrt_summed, sum_terms};
use crate::math::float::ziv::{Approx, guard, round_approx};
use crate::math::float::{BigFloat, Kind, Round};
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The two parts and their directions as a result.
#[inline]
fn pair(re: (BigFloat, Ordering), im: (BigFloat, Ordering)) -> (BigComplex, Ternary) {
    (BigComplex::new(re.0, im.0), (re.1, im.1))
}

/// An exact result of the precisions: both directions [`Ordering::Equal`].
#[inline]
fn exact_pair(re: BigFloat, im: BigFloat) -> (BigComplex, Ternary) {
    (BigComplex::new(re, im), (Ordering::Equal, Ordering::Equal))
}

/// `±∞` of the precision.
#[inline]
fn inf(prec: u64, negative: bool) -> BigFloat {
    BigFloat {
        prec,
        kind: Kind::Inf { negative },
    }
}

/// `±∞` for the sign `s` of an exact value, NaN for `s == 0`: `∞ · v`.
#[inline]
fn inf_times(prec: u64, s: i8) -> BigFloat {
    match s {
        0 => BigFloat::nan(prec),
        _ => inf(prec, s < 0),
    }
}

/// The sign of an exact sum, `-1`, `0` or `1`.
#[inline]
fn sign_of(e: &Exact) -> i8 {
    match e {
        Exact::Zero(_) => 0,
        Exact::Val(v) => {
            if v.negative {
                -1
            } else {
                1
            }
        }
    }
}

/// Checks both precisions.
#[inline]
#[track_caller]
fn check(prec: (u64, u64)) -> (u64, u64) {
    (BigFloat::check_prec(prec.0), BigFloat::check_prec(prec.1))
}

/// `±1` or `±0` for an infinite or finite part, of its sign, as Annex G "boxes" an infinity;
/// NaN as `+0`.
fn boxed(x: &BigFloat) -> BigFloat {
    let negative = x.is_sign_negative();
    let mut b = if x.is_infinite() {
        BigFloat::from(1u8)
    } else {
        BigFloat::zero(1)
    };
    b.set_sign(negative);
    b
}

/// `x`, or `+0` for NaN.
fn nan_to_zero(x: &BigFloat) -> BigFloat {
    if x.is_nan() {
        BigFloat::zero(1)
    } else {
        x.clone()
    }
}

/// `z·w` when `z` or `w` has an infinite part, by the C standard's `_Cmultd` of Annex G.5.1: the
/// products and sums of IEEE 754, and when both parts are NaN, the infinities recovered from the
/// parts boxed to `±1` and `±0`.
fn mul_infinite(z: &BigComplex, w: &BigComplex, prec: (u64, u64)) -> (BigComplex, Ternary) {
    let (a, b, c, d) = (&z.re, &z.im, &w.re, &w.im);
    // At two bits, only the special values and signs of these sums count: each has an infinite
    // or NaN term.
    let p = 2;
    let n = Round::Nearest;
    let x = a
        .mul_round(c, p, n)
        .0
        .sub_round(&b.mul_round(d, p, n).0, p, n)
        .0;
    let y = a
        .mul_round(d, p, n)
        .0
        .add_round(&b.mul_round(c, p, n).0, p, n)
        .0;
    if !(x.is_nan() && y.is_nan()) {
        debug_assert!(!x.is_finite() && !y.is_finite());
        return exact_pair(x.prec_round(prec.0, n).0, y.prec_round(prec.1, n).0);
    }
    let (mut a, mut b, mut c, mut d) = (a.clone(), b.clone(), c.clone(), d.clone());
    if z.is_infinite() {
        a = boxed(&a);
        b = boxed(&b);
        c = nan_to_zero(&c);
        d = nan_to_zero(&d);
    }
    if w.is_infinite() {
        c = boxed(&c);
        d = boxed(&d);
        a = nan_to_zero(&a);
        b = nan_to_zero(&b);
    }
    // `∞·(ac - bd)` and `∞·(ad + bc)`, the sums exact: all four are finite now.
    let re = sign_of(&sum_exact(&prod(&a, &c), &prod(&b, &d).neg(), n));
    let im = sign_of(&sum_exact(&prod(&a, &d), &prod(&b, &c), n));
    exact_pair(inf_times(prec.0, re), inf_times(prec.1, im))
}

impl BigComplex {
    /// `self + other`, each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_add`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let w = BigComplex::new(BigFloat::from(0.25), BigFloat::from(-0.25));
    /// let (s, o) = z.add_round(&w, 2, Round::Down);
    /// assert_eq!(s, BigComplex::new(BigFloat::from(1), BigFloat::from(0.75)));
    /// assert_eq!(o, (Ordering::Less, Ordering::Equal));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn add_round(
        &self,
        other: &BigComplex,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.add_round_parts(other, (prec, prec), (round, round))
    }

    /// [`add_round`](BigComplex::add_round) with a precision and a mode for each part, the real
    /// part's first, as MPC's `mpc_rnd_t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, Round};
    ///
    /// let (s, _) = BigComplex::from(1).add_round_parts(&BigComplex::i(53), (8, 16), (Round::Up, Round::Down));
    /// assert_eq!(s.prec(), (8, 16));
    /// ```
    #[track_caller]
    pub fn add_round_parts(
        &self,
        other: &BigComplex,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.add_round(&other.re, prec.0, round.0),
            self.im.add_round(&other.im, prec.1, round.1),
        )
    }

    /// `self - other`, each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_sub`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (d, _) = BigComplex::from(1).sub_round(&BigComplex::i(53), 53, Round::Nearest);
    /// assert_eq!(*d.im(), BigFloat::from(-1));
    /// ```
    #[track_caller]
    pub fn sub_round(
        &self,
        other: &BigComplex,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.sub_round_parts(other, (prec, prec), (round, round))
    }

    /// [`sub_round`](BigComplex::sub_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, Round};
    ///
    /// let (d, _) = BigComplex::i(53).sub_round_parts(&BigComplex::i(53), (8, 8), (Round::Up, Round::Down));
    /// assert!(d.im().is_sign_negative()); // an exact zero rounded down is -0
    /// ```
    #[track_caller]
    pub fn sub_round_parts(
        &self,
        other: &BigComplex,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.sub_round(&other.re, prec.0, round.0),
            self.im.sub_round(&other.im, prec.1, round.1),
        )
    }

    /// `self + x` for a real `x`, each part rounded, as MPC's `mpc_add_fr`: `x`'s imaginary part
    /// an exact zero, so that the imaginary part is `self`'s, rounded, of its sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::neg_zero(53));
    /// let (s, _) = z.add_real_round(&BigFloat::zero(53), 53, Round::Nearest);
    /// assert!(s.im().is_sign_negative()); // (1 - 0i) + (+0) = 1 - 0i
    /// ```
    #[track_caller]
    pub fn add_real_round(
        &self,
        x: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.add_real_round_parts(x, (prec, prec), (round, round))
    }

    /// [`add_real_round`](BigComplex::add_real_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (s, _) = BigComplex::i(53).add_real_round_parts(&BigFloat::from(2), (4, 6), (Round::Up, Round::Up));
    /// assert_eq!(s, BigComplex::new(BigFloat::from(2), BigFloat::from(1)));
    /// ```
    #[track_caller]
    pub fn add_real_round_parts(
        &self,
        x: &BigFloat,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.add_round(x, prec.0, round.0),
            self.im.prec_round(prec.1, round.1),
        )
    }

    /// `self - x` for a real `x`, each part rounded, as MPC's `mpc_sub_fr`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (d, _) = BigComplex::from(3).sub_real_round(&BigFloat::from(1), 53, Round::Nearest);
    /// assert_eq!(d, BigComplex::from(2));
    /// ```
    #[track_caller]
    pub fn sub_real_round(
        &self,
        x: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.sub_real_round_parts(x, (prec, prec), (round, round))
    }

    /// [`sub_real_round`](BigComplex::sub_real_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (d, _) = BigComplex::from(3).sub_real_round_parts(&BigFloat::from(1), (2, 2), (Round::Up, Round::Up));
    /// assert_eq!(d, BigComplex::from(2));
    /// ```
    #[track_caller]
    pub fn sub_real_round_parts(
        &self,
        x: &BigFloat,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.sub_round(x, prec.0, round.0),
            self.im.prec_round(prec.1, round.1),
        )
    }

    /// `self · x` for a real `x`, each part rounded, as MPC's `mpc_mul_fr`: `re·x + im·x·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(-3));
    /// let (p, _) = z.mul_real_round(&BigFloat::from(0.5), 53, Round::Nearest);
    /// assert_eq!(p, BigComplex::new(BigFloat::from(0.5), BigFloat::from(-1.5)));
    /// ```
    #[track_caller]
    pub fn mul_real_round(
        &self,
        x: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_real_round_parts(x, (prec, prec), (round, round))
    }

    /// [`mul_real_round`](BigComplex::mul_real_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (p, _) = BigComplex::i(53).mul_real_round_parts(&BigFloat::from(3), (2, 2), (Round::Down, Round::Down));
    /// assert_eq!(p, BigComplex::new(BigFloat::from(0), BigFloat::from(3)));
    /// ```
    #[track_caller]
    pub fn mul_real_round_parts(
        &self,
        x: &BigFloat,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.mul_round(x, prec.0, round.0),
            self.im.mul_round(x, prec.1, round.1),
        )
    }

    /// `self / x` for a real `x`, each part rounded, as MPC's `mpc_div_fr`: `re/x + im/x·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(-3));
    /// let (q, _) = z.div_real_round(&BigFloat::from(2), 53, Round::Nearest);
    /// assert_eq!(q, BigComplex::new(BigFloat::from(0.5), BigFloat::from(-1.5)));
    /// ```
    #[track_caller]
    pub fn div_real_round(
        &self,
        x: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.div_real_round_parts(x, (prec, prec), (round, round))
    }

    /// [`div_real_round`](BigComplex::div_real_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (q, o) = BigComplex::from(1).div_real_round_parts(&BigFloat::from(3), (8, 8), (Round::Up, Round::Up));
    /// assert_eq!(o.0, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn div_real_round_parts(
        &self,
        x: &BigFloat,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(
            self.re.div_round(x, prec.0, round.0),
            self.im.div_round(x, prec.1, round.1),
        )
    }

    /// `self · n` for an integer `n`, each part rounded, as MPC's `mpc_mul_si`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// let (p, _) = z.mul_int_round(-3, 53, Round::Nearest);
    /// assert_eq!(p, BigComplex::new(BigFloat::from(-3), BigFloat::from(-6)));
    /// ```
    #[track_caller]
    pub fn mul_int_round(
        &self,
        n: i64,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_real_round(&BigFloat::from(n), prec, round)
    }

    /// [`mul_int_round`](BigComplex::mul_int_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let (p, _) = BigComplex::i(4).mul_int_round_parts(5, (2, 2), (Round::Up, Round::Down));
    /// assert_eq!(*p.im(), BigFloat::from(4)); // 5 rounded down to 2 bits
    /// ```
    #[track_caller]
    pub fn mul_int_round_parts(
        &self,
        n: i64,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_real_round_parts(&BigFloat::from(n), prec, round)
    }

    /// `self · other`, each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_mul`: the parts `ac - bd` and `ad + bc` each the exact sum
    /// of two exact products, rounded once. A number with an infinite part times any other is
    /// the C standard's (Annex G.5.1): IEEE 754's products and sums, and when both parts are NaN,
    /// the infinities its `_Cmultd` recovers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// // (1 + 2^-60 i)(1 - 2^-60 i) = 1 + 2^-120, which 53 bits cannot hold.
    /// let t = BigFloat::from(1u64 << 60);
    /// let e = &BigFloat::from(1) / &t;
    /// let z = BigComplex::new(BigFloat::from(1), e.clone());
    /// let (p, o) = z.mul_round(&z.conj(), 53, Round::Up);
    /// assert_eq!(o, (Ordering::Greater, Ordering::Equal));
    /// assert!(p.im().is_zero());
    ///
    /// // (∞ + 0i)·(0 + i): ∞·0 is NaN in the real part, ∞ in the imaginary part.
    /// let x = BigComplex::from(BigFloat::infinity(53));
    /// let (p, _) = x.mul_round(&BigComplex::i(53), 53, Round::Nearest);
    /// assert!(p.re().is_nan() && p.im().is_infinite());
    /// ```
    #[track_caller]
    pub fn mul_round(
        &self,
        other: &BigComplex,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_round_parts(other, (prec, prec), (round, round))
    }

    /// [`mul_round`](BigComplex::mul_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(1));
    /// let (p, _) = z.mul_round_parts(&z, (53, 2), (Round::Nearest, Round::Down));
    /// assert_eq!(p, BigComplex::new(BigFloat::from(8), BigFloat::from(6)));
    /// ```
    #[track_caller]
    pub fn mul_round_parts(
        &self,
        other: &BigComplex,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = check(prec);
        if self.is_infinite() || other.is_infinite() {
            return mul_infinite(self, other, prec);
        }
        if !self.is_finite() || !other.is_finite() {
            return exact_pair(BigFloat::nan(prec.0), BigFloat::nan(prec.1));
        }
        let (a, b, c, d) = (&self.re, &self.im, &other.re, &other.im);
        if let Some(r) = mul_short(a, b, c, d, prec, round) {
            return r;
        }
        if let Some(r) = mul3(a, b, c, d, prec, round) {
            return r;
        }
        pair(
            sum_round(&prod(a, c), &prod(b, d).neg(), prec.0, round.0),
            sum_round(&prod(a, d), &prod(b, c), prec.1, round.1),
        )
    }

    /// The square `self²`, each part rounded to `prec` bits in the mode `round`, and the
    /// directions of the roundings: `self · self` as [`mul_round`](BigComplex::mul_round) has
    /// it, `a² - b²` and `2ab` each rounded once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (s, _) = z.sqr_round(53, Round::Nearest);
    /// assert_eq!(s, BigComplex::new(BigFloat::from(0), BigFloat::from(2)));
    /// ```
    #[track_caller]
    pub fn sqr_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.sqr_round_parts((prec, prec), (round, round))
    }

    /// [`sqr_round`](BigComplex::sqr_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(2), BigFloat::from(1));
    /// let (s, _) = z.sqr_round_parts((53, 53), (Round::Up, Round::Down));
    /// assert_eq!(s, BigComplex::new(BigFloat::from(3), BigFloat::from(4)));
    /// ```
    #[track_caller]
    pub fn sqr_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = check(prec);
        if !self.is_finite() {
            return self.mul_round_parts(self, prec, round);
        }
        let (a, b) = (&self.re, &self.im);
        let ab = match prod(a, b) {
            Exact::Val(v) => Exact::Val(v.scale(1)),
            z => z,
        };
        pair(
            sum_round(&square(a), &square(b).neg(), prec.0, round.0),
            round_exact(&ab, prec.1, round.1),
        )
    }

    /// The square, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert_eq!(BigComplex::i(10).sqr(), BigComplex::from(-1));
    /// ```
    pub fn sqr(&self) -> BigComplex {
        self.sqr_round_parts(self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// `self / other`, each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_div`: the parts `(ac + bd)/(c² + d²)` and
    /// `(bc - ad)/(c² + d²)` each rounded once. The special values are MPC's, after the C
    /// standard's Annex G.5.1: by a zero `∞·a + ∞·b·i` with the infinity of the sign of the real
    /// part of the divisor; an infinity over a finite number an infinity (or NaN) in each part by
    /// the signs of `a·c + b·d` and `b·c - a·d` with `a` and `b` boxed to `±1` and `0`; a finite
    /// number over an infinity zeros of the signs of the same sums with `c` and `d` boxed; and
    /// NaN in both parts otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(4), BigFloat::from(2));
    /// let w = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (q, o) = z.div_round(&w, 53, Round::Nearest);
    /// assert_eq!(q, BigComplex::new(BigFloat::from(3), BigFloat::from(-1)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    ///
    /// let (q, _) = BigComplex::from(1).div_round(&BigComplex::zero(53), 53, Round::Nearest);
    /// assert!(q.re().is_infinite() && q.im().is_nan());
    /// ```
    #[track_caller]
    pub fn div_round(
        &self,
        other: &BigComplex,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.div_round_parts(other, (prec, prec), (round, round))
    }

    /// [`div_round`](BigComplex::div_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let w = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (q, o) = BigComplex::from(1).div_round_parts(&w, (4, 4), (Round::Up, Round::Down));
    /// assert_eq!(q, BigComplex::new(BigFloat::from(0.5), BigFloat::from(-0.5)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    /// ```
    #[track_caller]
    pub fn div_round_parts(
        &self,
        other: &BigComplex,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = check(prec);
        let (z, w) = (self, other);
        if w.is_zero() {
            // `∞·a + ∞·b·i`, the infinity of the sign of `Re(w)`.
            let i = inf(2, w.re.is_sign_negative());
            return exact_pair(
                i.mul_round(&z.re, prec.0, Round::Nearest).0,
                i.mul_round(&z.im, prec.1, Round::Nearest).0,
            );
        }
        if z.is_infinite() && w.is_finite() {
            return div_inf_fin(z, w, prec);
        }
        if z.is_finite() && w.is_infinite() {
            return div_fin_inf(z, w, prec);
        }
        if !z.is_finite() || !w.is_finite() {
            return exact_pair(BigFloat::nan(prec.0), BigFloat::nan(prec.1));
        }
        div_finite(z, w, prec, round)
    }

    /// The norm `|self|² = re² + im²` rounded once to `prec` bits in the mode `round`, and the
    /// direction of the rounding, as MPC's `mpc_norm`: `+∞` when a part is infinite, even if the
    /// other is NaN, NaN when a part is NaN and the other not infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// assert_eq!(z.norm_round(53, Round::Nearest), (BigFloat::from(25), Ordering::Equal));
    /// assert_eq!(z.norm_round(4, Round::Down), (BigFloat::from(24), Ordering::Less));
    /// ```
    #[track_caller]
    pub fn norm_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if self.is_infinite() {
            return (BigFloat::infinity(prec), Ordering::Equal);
        }
        if !self.is_finite() {
            return (BigFloat::nan(prec), Ordering::Equal);
        }
        // The squares are `+0` or above zero, and so is their sum.
        sum_round(&square(&self.re), &square(&self.im), prec, round)
    }

    /// The norm `|self|²`, to nearest at the larger precision of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// assert_eq!(BigComplex::new(BigFloat::from(1), BigFloat::from(-2)).norm(), BigFloat::from(5));
    /// ```
    pub fn norm(&self) -> BigFloat {
        self.norm_round(self.max_prec(), Round::Nearest).0
    }

    /// The absolute value `|self| = √(re² + im²)` rounded once to `prec` bits in the mode
    /// `round`, and the direction of the rounding, as MPC's `mpc_abs` (MPFR's `mpfr_hypot`):
    /// `+∞` when a part is infinite, even if the other is NaN, NaN when a part is NaN and the
    /// other not infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(-4));
    /// assert_eq!(z.abs_round(53, Round::Nearest), (BigFloat::from(5), Ordering::Equal));
    /// let one = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// assert_eq!(one.abs_round(53, Round::Nearest).0, BigFloat::from(core::f64::consts::SQRT_2));
    /// ```
    #[track_caller]
    pub fn abs_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if self.is_infinite() {
            return (BigFloat::infinity(prec), Ordering::Equal);
        }
        if !self.is_finite() {
            return (BigFloat::nan(prec), Ordering::Equal);
        }
        let terms = [
            Term::product(&self.re, &self.re, false),
            Term::product(&self.im, &self.im, false),
        ];
        sqrt_summed(sum_terms(&terms, 2 * prec + 4), prec, round)
    }

    /// The absolute value `|self|`, to nearest at the larger precision of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// assert_eq!(BigComplex::new(BigFloat::from(-5), BigFloat::from(12)).abs(), BigFloat::from(13));
    /// ```
    pub fn abs(&self) -> BigFloat {
        self.abs_round(self.max_prec(), Round::Nearest).0
    }
}

/// The exact product of two parts that are not zero and whose significant bits fit one limb
/// each, in two limbs on the stack: the limbs, their number, their lowest bit's exponent and the
/// sign.
#[inline]
fn short_product(x: &BigFloat, y: &BigFloat) -> Option<([u64; 2], usize, i128, bool)> {
    let (nx, mx, lx) = super::raw::significant(x);
    let (ny, my, ly) = super::raw::significant(y);
    if mx.len() != 1 || my.len() != 1 {
        return None;
    }
    let p = u128::from(mx[0]) * u128::from(my[0]);
    let limbs = [p as u64, (p >> 64) as u64];
    Some((limbs, if limbs[1] == 0 { 1 } else { 2 }, lx + ly, nx != ny))
}

/// `(a + bi)(c + di)` for parts that are all finite, not zero and of one limb of significant
/// bits each (precisions to 64 bits, the usual short case): the four products exact in
/// registers, each part one rounded sum of two of them, as in the general case; `None`
/// otherwise.
fn mul_short(
    a: &BigFloat,
    b: &BigFloat,
    c: &BigFloat,
    d: &BigFloat,
    prec: (u64, u64),
    round: (Round, Round),
) -> Option<(BigComplex, Ternary)> {
    use crate::math::float::arith::{Mag, add_mags};
    if !(a.is_normal() && b.is_normal() && c.is_normal() && d.is_normal()) {
        return None;
    }
    let (ac, nac, lac, sac) = short_product(a, c)?;
    let (bd, nbd, lbd, sbd) = short_product(b, d)?;
    let (ad, nad, lad, sad) = short_product(a, d)?;
    let (bc, nbc, lbc, sbc) = short_product(b, c)?;
    fn mag(m: &[u64], lsb: i128, negative: bool) -> Mag<'_> {
        Mag { negative, m, lsb }
    }
    let mag_ac = mag(&ac[..nac], lac, sac);
    let mag_bd = mag(&bd[..nbd], lbd, !sbd);
    let mag_ad = mag(&ad[..nad], lad, sad);
    let mag_bc = mag(&bc[..nbc], lbc, sbc);
    Some(pair(
        add_mags(mag_ac, mag_bd, prec.0, round.0),
        add_mags(mag_ad, mag_bc, prec.1, round.1),
    ))
}

/// The precision of the parts from which a product takes three multiplications rather than
/// four.
pub(crate) const MUL3: u64 = 1500;

/// `(a + bi)(c + di)` by three exact products rather than four, for parts that are all finite,
/// not zero, of at least [`MUL3`] bits and of exponents near each other, `None` otherwise:
/// `ad`, `bc` and `(a + b)(c - d) = ac - ad + bc - bd`, so that `ac - bd` is the exact sum
/// `(a + b)(c - d) + ad - bc`, rounded once, and `ad + bc` the exact sum of two of them.
/// The sums `a + b` and `c - d` are exact and must be little longer than the parts.
fn mul3(
    a: &BigFloat,
    b: &BigFloat,
    c: &BigFloat,
    d: &BigFloat,
    prec: (u64, u64),
    round: (Round, Round),
) -> Option<(BigComplex, Ternary)> {
    let longest = a.prec().max(b.prec()).max(c.prec()).max(d.prec());
    if a.prec().min(b.prec()).min(c.prec()).min(d.prec()) < MUL3
        || !(a.is_normal() && b.is_normal() && c.is_normal() && d.is_normal())
    {
        return None;
    }
    let near = |x: &BigFloat, y: &BigFloat| {
        let (ex, ey) = (x.exponent().unwrap_or(0), y.exponent().unwrap_or(0));
        ex.abs_diff(ey) <= 32
    };
    if !near(a, b) || !near(c, d) {
        return None;
    }
    let val = |x: &BigFloat| match super::raw::exact(x) {
        Exact::Val(v) => v,
        Exact::Zero(_) => unreachable!("not zero"),
    };
    let s = super::raw::add_vals(&val(a), &val(b))?;
    let mut nd = val(d);
    nd.negative = !nd.negative;
    let t = super::raw::add_vals(&val(c), &nd)?;
    if s.bits().max(t.bits()) > longest + 64 {
        return None;
    }
    let mut m = crate::math::float::limbs::mul(&s.m, &t.m);
    crate::math::float::limbs::trim(&mut m);
    let u = Exact::Val(Val {
        negative: s.negative != t.negative,
        m,
        lsb: s.lsb + t.lsb,
    });
    let (ad, bc) = (prod(a, d), prod(b, c));
    let re = sum_round(
        &sum_exact(&u, &ad, round.0),
        &bc.clone().neg(),
        prec.0,
        round.0,
    );
    Some(pair(re, sum_round(&ad, &bc, prec.1, round.1)))
}

/// `z/w` for `z` with an infinite part and `w` finite and not zero, as MPC's `mpc_div_inf_fin`:
/// `∞·(a·c + b·d)` and `∞·(b·c - a·d)` with `a` and `b` boxed to `±1` for an infinite part and
/// `0` for the other (NaN included), the sums exact.
fn div_inf_fin(z: &BigComplex, w: &BigComplex, prec: (u64, u64)) -> (BigComplex, Ternary) {
    let unit = |x: &BigFloat| -> BigFloat {
        if x.is_infinite() {
            boxed(x)
        } else {
            BigFloat::zero(1)
        }
    };
    let (a, b) = (unit(&z.re), unit(&z.im));
    let (c, d) = (&w.re, &w.im);
    let n = Round::Nearest;
    let x = sign_of(&sum_exact(&prod(&a, c), &prod(&b, d), n));
    let y = sign_of(&sum_exact(&prod(&b, c), &prod(&a, d).neg(), n));
    exact_pair(inf_times(prec.0, x), inf_times(prec.1, y))
}

/// `z/w` for `z` finite and `w` with an infinite part, as MPC's `mpc_div_fin_inf`: zeros of the
/// signs of `a·c + b·d` and `b·c - a·d` with `c` and `d` boxed to `±1` for an infinite part and
/// `±0` for a finite one, the sums IEEE 754's to nearest.
fn div_fin_inf(z: &BigComplex, w: &BigComplex, prec: (u64, u64)) -> (BigComplex, Ternary) {
    let (c, d) = (boxed(&w.re), boxed(&w.im));
    let (a, b) = (&z.re, &z.im);
    let n = Round::Nearest;
    let x = sum_exact(&prod(a, &c), &prod(b, &d), n);
    let y = sum_exact(&prod(b, &c), &prod(a, &d).neg(), n);
    exact_pair(zero(prec.0, x.is_negative()), zero(prec.1, y.is_negative()))
}

/// The longest exact sum the quotient takes exactly, in bits, beyond the operands' lengths.
const EXACT_SPAN: u64 = 1 << 12;

/// The precision of the result from which a quotient goes by Ziv's strategy first even when its
/// exact sums are short: measured, from about 2000 bits the two quotients of cut sums of `w`
/// bits cost less than the exact ones of sums of `2p` bits (8% at 10^4 bits, 12% at 10^5).
const DIV_ZIV: u64 = 2000;

/// `z/w` for finite `z` and finite `w` that is not zero.
fn div_finite(
    z: &BigComplex,
    w: &BigComplex,
    prec: (u64, u64),
    round: (Round, Round),
) -> (BigComplex, Ternary) {
    let (a, b, c, d) = (&z.re, &z.im, &w.re, &w.im);
    let (zr, zi) = (a.is_sign_negative(), b.is_sign_negative());
    let (wr, wi) = (c.is_sign_negative(), d.is_sign_negative());
    // The zeros of `ac + bd` and `bc - ad` when the products in them are all zeros.
    let zero_re = |round: Round| {
        (
            zero(prec.0, zero_of(zr != wr, zi != wi, round)),
            Ordering::Equal,
        )
    };
    let zero_im = |round: Round| {
        (
            zero(prec.1, zero_of(zi != wr, zr == wi, round)),
            Ordering::Equal,
        )
    };
    if d.is_zero() {
        // `(ac + b·(±0))/c²` and `(bc - a·(±0))/c²`: `a/c` and `b/c`.
        let re = if a.is_zero() {
            zero_re(round.0)
        } else {
            a.div_round(c, prec.0, round.0)
        };
        let im = if b.is_zero() {
            zero_im(round.1)
        } else {
            b.div_round(c, prec.1, round.1)
        };
        return pair(re, im);
    }
    if c.is_zero() {
        // `(a·(±0) + bd)/d²` and `(b·(±0) - ad)/d²`: `b/d` and `-a/d`.
        let re = if b.is_zero() {
            zero_re(round.0)
        } else {
            b.div_round(d, prec.0, round.0)
        };
        let im = if a.is_zero() {
            zero_im(round.1)
        } else {
            let (q, o) = a.div_round(d, prec.1, super::super::arith::round_for(round.1, true));
            (-q, o.reverse())
        };
        return pair(re, im);
    }
    let (ac, bd, bc, ad) = (prod(a, c), prod(b, d), prod(b, c), prod(a, d).neg());
    let (cc, dd) = (square(c), square(d));
    let longest = a.prec().max(b.prec()).max(c.prec()).max(d.prec());
    let limit = 4 * longest + EXACT_SPAN;
    if prec.0.max(prec.1) < DIV_ZIV
        && sum_len(&ac, &bd) <= limit
        && sum_len(&bc, &ad) <= limit
        && sum_len(&cc, &dd) <= limit
    {
        let den = match sum_exact(&cc, &dd, Round::Nearest) {
            Exact::Val(v) => v,
            Exact::Zero(_) => unreachable!("c² + d² > 0"),
        };
        return pair(
            quotient(&sum_exact(&ac, &bd, round.0), &den, prec.0, round.0),
            quotient(&sum_exact(&bc, &ad, round.1), &den, prec.1, round.1),
        );
    }
    // Ziv's approximation: each sum cut to `w` bits is within `2^(2 - w)` of it, relatively,
    // and the quotient of the cut sums cut to `w` bits within `2^(1 - w)` of theirs, so that
    // the result is within `(1 + 2^(2-w))²/(1 - 2^(2-w))·(1 + 2^(1-w)) - 1 < 5.1·2^(1-w)` of
    // the quotient, relatively, for `w ≥ 20`. With the quotient `q` below `2^e`, the error is
    // then below `5.2·2^(1-w)·2^e < 2^(e + 4 - w)`.
    let mut w = prec.0.max(prec.1) + guard(prec.0.max(prec.1));
    let mut re: Option<(BigFloat, Ordering)> = None;
    let mut im: Option<(BigFloat, Ordering)> = None;
    for _ in 0..2 {
        let den = match sum_trunc(&cc, &dd, w, Round::Nearest) {
            Exact::Val(v) => v,
            Exact::Zero(_) => unreachable!("c² + d² > 0"),
        };
        for (part, (p, q), (pr, rd)) in [
            (&mut re, (&ac, &bd), (prec.0, round.0)),
            (&mut im, (&bc, &ad), (prec.1, round.1)),
        ] {
            if part.is_some() {
                continue;
            }
            match sum_trunc(p, q, w, rd) {
                Exact::Zero(negative) => *part = Some((zero(pr, negative), Ordering::Equal)),
                Exact::Val(n) => {
                    let v = div_trunc(n.negative, &n, &den, w);
                    let a = Approx {
                        negative: v.negative,
                        err: v.top() + 4 - i128::from(w),
                        lsb: v.lsb,
                        m: v.m,
                    };
                    *part = round_approx(&a, pr, rd);
                }
            }
        }
        if re.is_some() && im.is_some() {
            break;
        }
        w += w / 2;
    }
    // An exact or halfway result, or one too near either: the exact quotients.
    let den = || match sum_exact(&cc, &dd, Round::Nearest) {
        Exact::Val(v) => v,
        Exact::Zero(_) => unreachable!("c² + d² > 0"),
    };
    let re = re.unwrap_or_else(|| quotient(&sum_exact(&ac, &bd, round.0), &den(), prec.0, round.0));
    let im = im.unwrap_or_else(|| quotient(&sum_exact(&bc, &ad, round.1), &den(), prec.1, round.1));
    pair(re, im)
}

/// The sign of an exact zero sum of zeros of the signs `a` and `b` (IEEE 754's).
#[inline]
fn zero_of(a: bool, b: bool, round: Round) -> bool {
    super::raw::zero_sum(a, b, round)
}

/// `n/d` for the exact `n` and `d > 0`, rounded once.
fn quotient(n: &Exact, d: &Val, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match n {
        Exact::Zero(negative) => (zero(prec, *negative), Ordering::Equal),
        Exact::Val(n) => {
            super::super::arith::div_mags(n.negative, &n.m, n.lsb, &d.m, d.lsb, prec, round)
        }
    }
}

/// The precisions of an operator's result: for each part, the larger of the operands'.
#[inline]
fn op_prec(a: &BigComplex, b: &BigComplex) -> (u64, u64) {
    (a.re.prec().max(b.re.prec()), a.im.prec().max(b.im.prec()))
}

macro_rules! binary {
    ($Op:ident, $op:ident, $Assign:ident, $assign:ident, $method:ident) => {
        impl $Op<&BigComplex> for &BigComplex {
            type Output = BigComplex;
            /// Each part rounded to nearest at the larger precision of that part of the operands.
            #[inline]
            fn $op(self, rhs: &BigComplex) -> BigComplex {
                self.$method(rhs, op_prec(self, rhs), (Round::Nearest, Round::Nearest))
                    .0
            }
        }
        impl $Op<BigComplex> for &BigComplex {
            type Output = BigComplex;
            #[inline]
            fn $op(self, rhs: BigComplex) -> BigComplex {
                self.$op(&rhs)
            }
        }
        impl $Op<&BigComplex> for BigComplex {
            type Output = BigComplex;
            #[inline]
            fn $op(self, rhs: &BigComplex) -> BigComplex {
                (&self).$op(rhs)
            }
        }
        impl $Op<BigComplex> for BigComplex {
            type Output = BigComplex;
            #[inline]
            fn $op(self, rhs: BigComplex) -> BigComplex {
                (&self).$op(&rhs)
            }
        }
        impl $Assign<&BigComplex> for BigComplex {
            /// Each part rounded to nearest at the larger precision of that part of the operands.
            #[inline]
            fn $assign(&mut self, rhs: &BigComplex) {
                *self = (&*self).$op(rhs);
            }
        }
        impl $Assign<BigComplex> for BigComplex {
            #[inline]
            fn $assign(&mut self, rhs: BigComplex) {
                *self = (&*self).$op(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, add_round_parts);
binary!(Sub, sub, SubAssign, sub_assign, sub_round_parts);
binary!(Mul, mul, MulAssign, mul_assign, mul_round_parts);
binary!(Div, div, DivAssign, div_assign, div_round_parts);

impl Neg for BigComplex {
    type Output = BigComplex;
    /// Both signs turned over, exactly, as MPC's `mpc_neg`; NaN stays NaN.
    #[inline]
    fn neg(self) -> BigComplex {
        BigComplex::new(-self.re, -self.im)
    }
}

impl Neg for &BigComplex {
    type Output = BigComplex;
    #[inline]
    fn neg(self) -> BigComplex {
        -self.clone()
    }
}
