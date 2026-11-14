//! Arbitrary-precision complex numbers, as GNU MPC: a [`BigComplex`] is a pair of [`BigFloat`]s,
//! the real and the imaginary part, each of its own precision, and every operation rounds each
//! part of its exact result once, correctly, to a precision and in a mode of its own.
//!
//! The methods whose names end in `_round` take one precision and one mode for both parts and
//! return the result with the directions of the two roundings, `(re, im)`, as MPC's ternary
//! value with its `MPC_INEX_RE` and `MPC_INEX_IM`; those ending in `_round_parts` take a pair of
//! precisions and a pair of modes, MPC's `mpc_rnd_t`, the real part's first. The methods without
//! `_round` round to nearest at the precisions of `self`'s parts, and the operators at the larger
//! precision of the operands' real parts for the real part and of their imaginary parts for the
//! imaginary part.
//!
//! The special values are those of MPC, after the C standard's Annex G: a number with an infinite
//! part is an infinity even if its other part is NaN, `(∞ + i·NaN)·(2 + 0i)` is `∞ + i·NaN`, a
//! finite number over an infinity a zero, and so on, function by function.

mod arith;
mod exp;
pub(crate) mod fast;
mod inv;
mod pow;
pub(crate) mod raw;
mod real;
mod set;
mod sqrt;
mod trig;

use super::{BigFloat, Round};
use crate::math::complex::Complex;
use core::cmp::Ordering;
use core::fmt;

/// An arbitrary-precision complex number `re + im·i`, as GNU MPC's `mpc_t`: its parts are
/// [`BigFloat`]s of their own precisions, and every operation rounds each part of its exact
/// result once, correctly, to a precision and in a mode of its own.
///
/// The methods whose names end in `_round` take one precision and one mode for both parts and
/// return the result with the directions of the two roundings, `(re, im)`, as MPC's ternary
/// value with its `MPC_INEX_RE` and `MPC_INEX_IM`; those ending in `_round_parts` take a pair of
/// precisions and a pair of modes, MPC's `mpc_rnd_t`, the real part's first. The methods without
/// `_round` round to nearest at the precisions of `self`'s parts, and the operators each part at
/// the larger precision of that part of the operands.
///
/// The special values are those of MPC, after the C standard's Annex G: a number with an
/// infinite part is an infinity even if its other part is NaN, `(∞ + NaN·i)·(2 + 0i)` is
/// `∞ + NaN·i`, a finite number over an infinity a zero, and so on, method by method. An exact
/// zero part is IEEE 754's sum of zeros: `+0` for zeros of both signs or terms that cancel, `-0`
/// when that part rounds down.
///
/// # Examples
///
/// ```
/// use a::math::float::{BigComplex, BigFloat, Round};
/// use core::cmp::Ordering;
///
/// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2)); // 1 + 2i
/// let w = BigComplex::new(BigFloat::from(3), BigFloat::from(-1)); // 3 - i
/// let (p, o) = z.mul_round(&w, 53, Round::Nearest);
/// assert_eq!(p, BigComplex::new(BigFloat::from(5), BigFloat::from(5)));
/// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
/// ```
#[derive(Clone)]
pub struct BigComplex {
    pub(crate) re: BigFloat,
    pub(crate) im: BigFloat,
}

/// The directions of the roundings of the real and the imaginary part.
pub(crate) type Ternary = (Ordering, Ordering);

impl BigComplex {
    /// The number `re + im·i`, each part as it is, of its own precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1.5), BigFloat::from(-2));
    /// assert_eq!(z.prec(), (53, 32));
    /// ```
    #[inline]
    pub fn new(re: BigFloat, im: BigFloat) -> BigComplex {
        BigComplex { re, im }
    }

    /// `+0 + 0i`, both parts of the precision `prec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert!(BigComplex::zero(64).is_zero());
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC),
    /// as every function that takes a precision.
    #[track_caller]
    pub fn zero(prec: u64) -> BigComplex {
        BigComplex::new(BigFloat::zero(prec), BigFloat::zero(prec))
    }

    /// `NaN + NaN·i`, both parts of the precision `prec`, as a new MPC variable holds.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert!(BigComplex::nan(64).is_nan());
    /// ```
    #[track_caller]
    pub fn nan(prec: u64) -> BigComplex {
        BigComplex::new(BigFloat::nan(prec), BigFloat::nan(prec))
    }

    /// The imaginary unit `i` (`+0 + 1i`), both parts of the precision `prec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// let i = BigComplex::i(10);
    /// assert_eq!(&i * &i, BigComplex::from(-1));
    /// ```
    #[track_caller]
    pub fn i(prec: u64) -> BigComplex {
        let one = BigFloat::from(1u8).with_prec(prec);
        BigComplex::new(BigFloat::zero(prec), one)
    }

    /// The real part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// assert_eq!(*z.re(), BigFloat::from(3));
    /// ```
    #[inline]
    pub fn re(&self) -> &BigFloat {
        &self.re
    }

    /// The imaginary part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// assert_eq!(*z.im(), BigFloat::from(4));
    /// ```
    #[inline]
    pub fn im(&self) -> &BigFloat {
        &self.im
    }

    /// The real part, to change, as MPC's `mpc_realref`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let mut z = BigComplex::zero(53);
    /// *z.re_mut() = BigFloat::from(2.5);
    /// assert_eq!(z, BigComplex::from(2.5));
    /// ```
    #[inline]
    pub fn re_mut(&mut self) -> &mut BigFloat {
        &mut self.re
    }

    /// The imaginary part, to change, as MPC's `mpc_imagref`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let mut z = BigComplex::zero(53);
    /// *z.im_mut() = BigFloat::from(1);
    /// assert_eq!(z, BigComplex::i(53));
    /// ```
    #[inline]
    pub fn im_mut(&mut self) -> &mut BigFloat {
        &mut self.im
    }

    /// The parts `(re, im)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let (re, im) = BigComplex::i(8).into_parts();
    /// assert!(re.is_zero() && im == BigFloat::from(1));
    /// ```
    #[inline]
    pub fn into_parts(self) -> (BigFloat, BigFloat) {
        (self.re, self.im)
    }

    /// The precisions of the parts, `(re, im)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert_eq!(BigComplex::zero(100).prec(), (100, 100));
    /// ```
    #[inline]
    pub fn prec(&self) -> (u64, u64) {
        (self.re.prec(), self.im.prec())
    }

    /// The larger of the precisions of the parts.
    #[inline]
    pub(crate) fn max_prec(&self) -> u64 {
        self.re.prec().max(self.im.prec())
    }

    /// Whether a part is NaN (and the other not an infinity: see
    /// [`is_infinite`](BigComplex::is_infinite)).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// assert!(BigComplex::new(BigFloat::from(1), BigFloat::nan(53)).is_nan());
    /// assert!(!BigComplex::new(BigFloat::infinity(53), BigFloat::nan(53)).is_nan());
    /// ```
    #[inline]
    pub fn is_nan(&self) -> bool {
        (self.re.is_nan() || self.im.is_nan()) && !self.is_infinite()
    }

    /// Whether a part is an infinity, whatever the other, as the C standard's Annex G has it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// assert!(BigComplex::new(BigFloat::nan(53), BigFloat::neg_infinity(53)).is_infinite());
    /// ```
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }

    /// Whether both parts are finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert!(BigComplex::from(1).is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Whether both parts are zeros, of either sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert!(BigComplex::from(-0.0).is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    /// The number with each part rounded to `prec` bits in the mode `round`, and the directions
    /// of the roundings, as MPC's `mpc_set`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(5), BigFloat::from(-5)); // 101 in binary
    /// let (r, o) = z.prec_round(2, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(4), BigFloat::from(-4)));
    /// assert_eq!(o, (Ordering::Less, Ordering::Greater));
    /// ```
    #[track_caller]
    pub fn prec_round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
        self.prec_round_parts((prec, prec), (round, round))
    }

    /// [`prec_round`](BigComplex::prec_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::from(1.0 / 3.0);
    /// let (r, _) = z.prec_round_parts((24, 8), (Round::Up, Round::Down));
    /// assert_eq!(r.prec(), (24, 8));
    /// ```
    #[track_caller]
    pub fn prec_round_parts(
        &self,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let (re, a) = self.re.prec_round(prec.0, round.0);
        let (im, b) = self.im.prec_round(prec.1, round.1);
        (BigComplex::new(re, im), (a, b))
    }

    /// The number rounded to nearest at `prec` bits in both parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigComplex;
    ///
    /// assert_eq!(BigComplex::from(7).with_prec(200).prec(), (200, 200));
    /// ```
    #[track_caller]
    pub fn with_prec(&self, prec: u64) -> BigComplex {
        self.prec_round(prec, Round::Nearest).0
    }

    /// The conjugate `re - im·i`, exactly, of the same precisions, as MPC's `mpc_conj`: the
    /// sign of the imaginary part turned over, NaN aside.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// assert_eq!(z.conj(), BigComplex::new(BigFloat::from(1), BigFloat::from(-2)));
    /// ```
    pub fn conj(&self) -> BigComplex {
        BigComplex::new(self.re.clone(), -&self.im)
    }

    /// The number times `i`, or times `-i` when `negative`, exactly, as MPC's `mpc_mul_i`:
    /// `-im + re·i` (or `im - re·i`), the parts moved with their precisions.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// assert_eq!(z.mul_i(false), BigComplex::new(BigFloat::from(-2), BigFloat::from(1)));
    /// assert_eq!(z.mul_i(true), BigComplex::new(BigFloat::from(2), BigFloat::from(-1)));
    /// ```
    pub fn mul_i(&self, negative: bool) -> BigComplex {
        if negative {
            BigComplex::new(self.im.clone(), -&self.re)
        } else {
            BigComplex::new(-&self.im, self.re.clone())
        }
    }

    /// [`mul_i`](BigComplex::mul_i) rounded to `prec` bits in the mode `round`, and the
    /// directions of the roundings.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// let (r, _) = z.mul_i_round(false, 10, Round::Nearest);
    /// assert_eq!(r.prec(), (10, 10));
    /// ```
    #[track_caller]
    pub fn mul_i_round(
        &self,
        negative: bool,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_i(negative).prec_round(prec, round)
    }

    /// [`mul_i_round`](BigComplex::mul_i_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// let (r, _) = z.mul_i_round_parts(true, (8, 9), (Round::Up, Round::Up));
    /// assert_eq!(r.prec(), (8, 9));
    /// ```
    #[track_caller]
    pub fn mul_i_round_parts(
        &self,
        negative: bool,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.mul_i(negative).prec_round_parts(prec, round)
    }
}

impl From<BigFloat> for BigComplex {
    /// The real number `x + 0i`, the imaginary part `+0` of `x`'s precision.
    fn from(x: BigFloat) -> BigComplex {
        let im = BigFloat::zero(x.prec());
        BigComplex::new(x, im)
    }
}

impl From<&BigFloat> for BigComplex {
    /// The real number `x + 0i`, the imaginary part `+0` of `x`'s precision.
    fn from(x: &BigFloat) -> BigComplex {
        BigComplex::from(x.clone())
    }
}

impl From<(BigFloat, BigFloat)> for BigComplex {
    /// The number `re + im·i`.
    fn from((re, im): (BigFloat, BigFloat)) -> BigComplex {
        BigComplex::new(re, im)
    }
}

impl From<BigComplex> for (BigFloat, BigFloat) {
    /// The parts `(re, im)`.
    fn from(z: BigComplex) -> (BigFloat, BigFloat) {
        z.into_parts()
    }
}

impl From<Complex<f64>> for BigComplex {
    /// The number exactly, both parts at 53 bits.
    fn from(z: Complex<f64>) -> BigComplex {
        BigComplex::new(BigFloat::from(z.re), BigFloat::from(z.im))
    }
}

impl From<Complex<f32>> for BigComplex {
    /// The number exactly, both parts at 24 bits.
    fn from(z: Complex<f32>) -> BigComplex {
        BigComplex::new(BigFloat::from(z.re), BigFloat::from(z.im))
    }
}

macro_rules! from_real {
    ($($t:ty),*) => {$(
        impl From<$t> for BigComplex {
            /// The real number exactly, as [`BigFloat::from`] makes it, the imaginary part `+0`
            /// of the same precision.
            fn from(x: $t) -> BigComplex {
                BigComplex::from(BigFloat::from(x))
            }
        }
    )*};
}

from_real!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl PartialEq for BigComplex {
    /// Whether both parts are equal in value: NaN equals nothing, `-0` equals `+0`, and the
    /// precisions do not count.
    fn eq(&self, other: &BigComplex) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl fmt::Debug for BigComplex {
    /// Writes the parts as [`BigFloat`]'s `Debug` does: `BigComplex { re: …, im: … }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BigComplex")
            .field("re", &self.re)
            .field("im", &self.im)
            .finish()
    }
}

impl fmt::Display for BigComplex {
    /// Writes `(re im)`, as MPC's `mpc_out_str` lays it out, each part as
    /// [`BigFloat`'s `Display`](BigFloat#impl-Display-for-BigFloat) writes it, with the
    /// formatter's options applied to each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1.5), BigFloat::from(-8));
    /// assert_eq!(z.to_string(), "(1.5 -8)");
    /// assert_eq!(format!("{z:.2}"), "(1.50 -8.00)");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(&self.re, f)?;
        f.write_str(" ")?;
        fmt::Display::fmt(&self.im, f)?;
        f.write_str(")")
    }
}
