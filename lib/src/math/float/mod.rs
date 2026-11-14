//! Arbitrary-precision binary floating point with correct rounding.
//!
//! A [`BigFloat`] is NaN, an infinity, a signed zero, or `±m·2^e` with an integer mantissa `m` of
//! the float's precision `p` in bits, chosen per value, `2^(p-1) ≤ m < 2^p`. Every operation is
//! correctly rounded: its result is the exact result rounded once to the precision asked for, in
//! one of the four modes of [`Round`], and the methods whose names end in `_round` also return the
//! direction of that rounding, as MPFR's ternary value: [`Ordering::Less`] when the result is below
//! the exact one, [`Ordering::Equal`] when it is exact, [`Ordering::Greater`] when it is above.
//!
//! The operators round to nearest, ties to even, at the larger precision of their operands, and so
//! do the methods without `_round`, at the precision of `self`. There is no global state: the
//! precision and the mode are given to each operation, and a `BigFloat` is `Send` and `Sync`.
//!
//! The exponent `e` of the form `0.1…·2^e` lies in [`BigFloat::EXP_MIN`]..=[`BigFloat::EXP_MAX`];
//! a result beyond is rounded to an infinity or the largest finite number, and one below to zero or
//! the smallest positive number, by the mode, as MPFR does with its exponent range. There are no
//! subnormal numbers.
//!
//! Floats add, subtract, multiply and divide with the `_round` methods, the operators and their
//! assigning forms, with primitive numbers and [`Number`]s on either side too;
//! [`BigFloat::set_add`] and the other `set_` methods write a result into a number that exists,
//! and [`BigFloat::sum_round`] and [`BigFloat::dot_round`] round an exact sum or dot product
//! once. [`BigFloat::sqrt_round`], [`BigFloat::rec_sqrt_round`],
//! [`BigFloat::nth_root_round`], [`BigFloat::powi_round`], [`BigFloat::pow_number_round`] and
//! [`BigFloat::pow_rational_round`] give roots and powers.
//!
//! They convert to and from `f64` and `f32` ([`BigFloat::to_f64_round`],
//! [`BigFloat::from_f64_round`]), to integers ([`BigFloat::floor`], [`BigFloat::rint_round`],
//! [`BigFloat::to_number_round`], [`BigFloat::to_i64_round`]) and exactly to and from
//! [`Rational`](crate::math::rational::Rational)s, and compare exactly with integers, floats,
//! numbers and rationals. `Display` writes the shortest decimal that reads back, as `f64`'s does;
//! [`BigFloat::to_digits`] and [`BigFloat::to_string_radix`] write the digits of MPFR's
//! `mpfr_get_str` in a radix from 2 to 36, and [`BigFloat::from_str_radix`] and `FromStr` read
//! any number of digits, rounded once.
//!
//! The exponentials and logarithms ([`BigFloat::exp_round`], [`BigFloat::ln_round`] and their
//! kind), [`BigFloat::pow_round`], the hyperbolic functions and their inverses, the circular
//! functions and their inverses ([`BigFloat::sin_round`], [`BigFloat::atan2_round`] and their
//! kind) and the constants π, `e`, `ln 2`, `ln 10`, `γ` and Catalan's `G`
//! ([`BigFloat::pi_round`] and the others, kept per process once computed) are correctly rounded
//! in every mode, with MPFR's special values.
//!
//! A [`BigComplex`] is a complex number of two floats, as GNU MPC's `mpc_t`: each part of each
//! result is rounded once in a mode of its own, for the arithmetic, the square root, the norm and
//! the absolute value, and for the elementary functions, with MPC's branch cuts and special
//! values. Both are elements of matrices, [`Matrix`](crate::math::matrix::Matrix), whose products
//! round each element once.

mod arith;
mod atan;
mod ball;
mod cmp;
mod compare;
pub(crate) mod complex;
pub(crate) mod consts;
mod convert;
mod exp;
mod fix;
mod general;
mod hyper;
mod integer;
pub(crate) mod limbs;
mod log;
mod pow;
mod power;
mod power_fixed;
mod prim;
mod rec_sqrt;
mod root_small;
mod round;
mod same;
pub(crate) mod scale;
mod short;
mod sincos;
mod small;
mod small_fix;
mod split;
mod sqrt_small;
pub(crate) mod sum;
mod text;
mod trig;
pub(crate) mod ziv;

use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use core::fmt;

pub use complex::BigComplex;
pub use convert::TryFromBigFloatError;
pub use text::{BigFloatErrorKind, ParseBigFloatError};

/// A rounding mode: the direction in which a result that is not exact is rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Round {
    /// To the nearest number of the precision; of two equally near, the one whose last bit is
    /// zero (ties to even), as IEEE 754's `roundTiesToEven`.
    #[default]
    Nearest,
    /// Toward zero: the number of the precision nearest to zero that is not beyond the result.
    TowardZero,
    /// Toward positive infinity: the least number of the precision not below the result.
    Up,
    /// Toward negative infinity: the greatest number of the precision not above the result.
    Down,
}

/// An arbitrary-precision binary floating-point number: NaN, `±∞`, `±0`, or `±m·2^e` with an
/// integer mantissa `m` of the precision `p`, `2^(p-1) ≤ m < 2^p`. See the [module](self).
#[derive(Clone)]
pub struct BigFloat {
    /// The precision in bits, from [`BigFloat::MIN_PREC`] to [`BigFloat::MAX_PREC`].
    pub(crate) prec: u64,
    /// The value.
    pub(crate) kind: Kind,
}

/// The value of a [`BigFloat`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Not a number.
    Nan,
    /// An infinity of the sign.
    Inf { negative: bool },
    /// A zero of the sign.
    Zero { negative: bool },
    /// `±0.mant·2^exp`: `mant` of `ceil(prec / 64)` limbs, the least significant first, the top
    /// bit of the top limb set and the bits below the precision zero, so that its value as an
    /// integer `M` gives the number `±M·2^(exp - 64·len)`.
    Finite {
        negative: bool,
        exp: i64,
        mant: Vec<u64>,
    },
}

impl BigFloat {
    /// The least precision, 1 bit.
    pub const MIN_PREC: u64 = 1;
    /// The greatest precision, `2^40` bits (128 GiB of mantissa).
    pub const MAX_PREC: u64 = 1 << 40;
    /// The greatest exponent `e` of a finite number `0.1…·2^e`.
    pub const EXP_MAX: i64 = (1 << 62) - 1;
    /// The least exponent `e` of a finite number `0.1…·2^e` that is not zero.
    pub const EXP_MIN: i64 = -Self::EXP_MAX;

    /// Checks a precision, panicking outside [`MIN_PREC`](Self::MIN_PREC)..=[`MAX_PREC`](Self::MAX_PREC).
    #[inline]
    #[track_caller]
    pub(crate) fn check_prec(prec: u64) -> u64 {
        assert!(
            (Self::MIN_PREC..=Self::MAX_PREC).contains(&prec),
            "precision {prec} is not in 1..=2^40"
        );
        prec
    }

    /// NaN of the precision `prec`.
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](Self::MIN_PREC)..=[`MAX_PREC`](Self::MAX_PREC), as every
    /// function that takes a precision.
    #[track_caller]
    pub fn nan(prec: u64) -> BigFloat {
        BigFloat {
            prec: Self::check_prec(prec),
            kind: Kind::Nan,
        }
    }

    /// Positive infinity of the precision `prec`.
    #[track_caller]
    pub fn infinity(prec: u64) -> BigFloat {
        Self::special(prec, Kind::Inf { negative: false })
    }

    /// Negative infinity of the precision `prec`.
    #[track_caller]
    pub fn neg_infinity(prec: u64) -> BigFloat {
        Self::special(prec, Kind::Inf { negative: true })
    }

    /// Positive zero of the precision `prec`.
    #[track_caller]
    pub fn zero(prec: u64) -> BigFloat {
        Self::special(prec, Kind::Zero { negative: false })
    }

    /// Negative zero of the precision `prec`.
    #[track_caller]
    pub fn neg_zero(prec: u64) -> BigFloat {
        Self::special(prec, Kind::Zero { negative: true })
    }

    #[inline]
    #[track_caller]
    fn special(prec: u64, kind: Kind) -> BigFloat {
        BigFloat {
            prec: Self::check_prec(prec),
            kind,
        }
    }

    /// The number `m·2^e` rounded to the precision `prec` in the mode `round`, and the direction
    /// of the rounding; `m` may be zero, which gives `+0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from_parts(&Number::from(7), -1, 2, Round::Nearest);
    /// assert_eq!(x, BigFloat::from(4)); // 3.5 to 2 bits: 3 and 4 are as near, 4 is even
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn from_parts(m: &Number, e: i64, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = Self::check_prec(prec);
        let limbs = limbs::of_number(m);
        if limbs.is_empty() {
            return (BigFloat::zero(prec), Ordering::Equal);
        }
        round::round_mag(m.is_negative(), &limbs, i128::from(e), false, prec, round)
    }

    /// The precision in bits.
    #[inline]
    pub fn prec(&self) -> u64 {
        self.prec
    }

    /// Whether the number is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::Nan)
    }

    /// Whether the number is an infinity.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        matches!(self.kind, Kind::Inf { .. })
    }

    /// Whether the number is neither NaN nor an infinity.
    #[inline]
    pub fn is_finite(&self) -> bool {
        matches!(self.kind, Kind::Zero { .. } | Kind::Finite { .. })
    }

    /// Whether the number is a zero of either sign.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self.kind, Kind::Zero { .. })
    }

    /// Whether the number is finite and not zero.
    #[inline]
    pub fn is_normal(&self) -> bool {
        matches!(self.kind, Kind::Finite { .. })
    }

    /// Whether the sign bit is set: for `-0`, `-∞` and the negative numbers; NaN has none.
    #[inline]
    pub fn is_sign_negative(&self) -> bool {
        match self.kind {
            Kind::Nan => false,
            Kind::Inf { negative } | Kind::Zero { negative } | Kind::Finite { negative, .. } => {
                negative
            }
        }
    }

    /// Whether the sign bit is clear: for `+0`, `+∞` and the positive numbers; NaN has none.
    #[inline]
    pub fn is_sign_positive(&self) -> bool {
        !self.is_nan() && !self.is_sign_negative()
    }

    /// The exponent `e` of the number in the form `±0.1…·2^e`, `None` for NaN, an infinity or a
    /// zero: `2^(e-1) ≤ |x| < 2^e`.
    #[inline]
    pub fn exponent(&self) -> Option<i64> {
        match self.kind {
            Kind::Finite { exp, .. } => Some(exp),
            _ => None,
        }
    }

    /// The integer mantissa and the exponent `(m, e)` of a finite number that is not zero, with
    /// `x = m·2^e`, `m` of the sign of `x` and `2^(p-1) ≤ |m| < 2^p` for the precision `p`, in
    /// little-endian bytes; `None` for NaN, an infinity or a zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::numbers::Number;
    ///
    /// let x = BigFloat::from(-0.375);
    /// let (m, e) = x.mantissa_exponent().unwrap();
    /// assert_eq!(x.prec(), 53);
    /// assert_eq!(m, Number::from(-3) << 51);
    /// assert_eq!(e, -54);
    /// ```
    pub fn mantissa_exponent(&self) -> Option<(Number, i64)> {
        match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => {
                let unused = 64 * mant.len() as u64 - self.prec;
                let m = limbs::shr(mant, unused);
                Some((
                    limbs::to_number(&m, *negative, ByteOrder::LittleEndian),
                    exp - self.prec as i64,
                ))
            }
            _ => None,
        }
    }

    /// The number rounded to the precision `prec` in the mode `round`, and the direction of the
    /// rounding; NaN, the infinities and the zeros take the precision and stay as they are.
    #[track_caller]
    pub fn prec_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = Self::check_prec(prec);
        match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => round::round_mag(
                *negative,
                mant,
                i128::from(*exp) - 64 * mant.len() as i128,
                false,
                prec,
                round,
            ),
            kind => (
                BigFloat {
                    prec,
                    kind: kind.clone(),
                },
                Ordering::Equal,
            ),
        }
    }

    /// The number rounded to nearest at the precision `prec`.
    #[track_caller]
    pub fn with_prec(&self, prec: u64) -> BigFloat {
        self.prec_round(prec, Round::Nearest).0
    }

    /// Sets the precision to `prec`, rounding the value in the mode `round`, and returns the
    /// direction of the rounding.
    #[track_caller]
    pub fn set_prec(&mut self, prec: u64, round: Round) -> Ordering {
        let (x, o) = self.prec_round(prec, round);
        *self = x;
        o
    }

    /// The absolute value, of the same precision.
    pub fn abs(&self) -> BigFloat {
        let mut x = self.clone();
        x.set_sign(false);
        x
    }

    /// The number with the sign bit of `sign`, as `f64::copysign`; NaN stays NaN.
    pub fn copysign(&self, sign: &BigFloat) -> BigFloat {
        let mut x = self.clone();
        x.set_sign(sign.is_sign_negative());
        x
    }

    /// Sets the sign bit, NaN aside.
    #[inline]
    pub(crate) fn set_sign(&mut self, neg: bool) {
        match &mut self.kind {
            Kind::Nan => {}
            Kind::Inf { negative } | Kind::Zero { negative } | Kind::Finite { negative, .. } => {
                *negative = neg
            }
        }
    }

    /// The parts of a finite number that is not zero: its sign, its mantissa's limbs as an integer
    /// `M` and the exponent of their lowest bit, `|x| = M·2^lsb`.
    #[inline]
    pub(crate) fn parts(&self) -> Option<(bool, &[u64], i128)> {
        match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => Some((*negative, mant, i128::from(*exp) - 64 * mant.len() as i128)),
            _ => None,
        }
    }
}

impl fmt::Debug for BigFloat {
    /// Writes the precision and the value in hexadecimal, `BigFloat { prec: 53, value: 0x3p-1 }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigFloat {{ prec: {}, value: ", self.prec)?;
        self.write_hex(f)?;
        f.write_str(" }")
    }
}

impl BigFloat {
    /// Writes `nan`, `inf`, `-inf`, `0`, `-0` or `±0x<m>p<e>` for the number `m·2^e`, `m` odd,
    /// in hexadecimal and `e` in decimal.
    pub(crate) fn write_hex(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::Nan => f.write_str("nan"),
            Kind::Inf { negative } => f.write_str(if *negative { "-inf" } else { "inf" }),
            Kind::Zero { negative } => f.write_str(if *negative { "-0" } else { "0" }),
            Kind::Finite {
                negative,
                exp,
                mant,
            } => {
                let z = limbs::trailing_zeros(mant);
                let m = limbs::shr(mant, z);
                let e = exp - 64 * mant.len() as i64 + z as i64;
                let hex = limbs::to_number(&m, false, ByteOrder::LittleEndian).to_str_radix(16);
                write!(f, "{}0x{hex}p{e}", if *negative { "-" } else { "" })
            }
        }
    }
}
