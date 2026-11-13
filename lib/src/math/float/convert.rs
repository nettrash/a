//! Exact conversions into a [`BigFloat`] from the primitive floats and integers and from
//! [`Number`]s, at the precision that holds them; a [`Rational`] rounded to a precision; and a
//! finite `BigFloat` into the `Rational` it is, exactly.

use super::arith::div_mags;
use super::limbs;
use super::round::round_mag;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::{ByteOrder, Number};
use crate::math::rational::Rational;
use core::cmp::Ordering;
use core::fmt;

/// The error of a conversion of NaN or an infinity into a type that has neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TryFromBigFloatError(());

impl fmt::Display for TryFromBigFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NaN or an infinity has no value of the type")
    }
}

impl core::error::Error for TryFromBigFloatError {}

/// `±m·2^e` at the precision `prec`, exactly: `m` must have at most `prec` bits.
fn exact(negative: bool, m: u128, e: i64, prec: u64) -> BigFloat {
    if m == 0 {
        return BigFloat {
            prec,
            kind: Kind::Zero { negative },
        };
    }
    let mut v = vec![m as u64, (m >> 64) as u64];
    limbs::trim(&mut v);
    round_mag(negative, &v, i128::from(e), false, prec, Round::Nearest).0
}

impl From<f64> for BigFloat {
    /// The float exactly, at 53 bits, NaN, the infinities and the zeros as they are.
    fn from(x: f64) -> BigFloat {
        let bits = x.to_bits();
        let negative = bits >> 63 == 1;
        let (field, frac) = ((bits >> 52 & 0x7ff) as i64, bits & ((1 << 52) - 1));
        match field {
            0x7ff if frac != 0 => BigFloat::nan(53),
            0x7ff => BigFloat {
                prec: 53,
                kind: Kind::Inf { negative },
            },
            0 => exact(negative, u128::from(frac), -1074, 53),
            _ => exact(negative, u128::from(frac | 1 << 52), field - 1075, 53),
        }
    }
}

impl From<f32> for BigFloat {
    /// The float exactly, at 24 bits, NaN, the infinities and the zeros as they are.
    fn from(x: f32) -> BigFloat {
        let bits = x.to_bits();
        let negative = bits >> 31 == 1;
        let (field, frac) = ((bits >> 23 & 0xff) as i64, bits & ((1 << 23) - 1));
        match field {
            0xff if frac != 0 => BigFloat::nan(24),
            0xff => BigFloat {
                prec: 24,
                kind: Kind::Inf { negative },
            },
            0 => exact(negative, u128::from(frac), -149, 24),
            _ => exact(negative, u128::from(frac | 1 << 23), field - 150, 24),
        }
    }
}

macro_rules! from_integer {
    ($($t:ty => $bits:expr),*) => {$(
        impl From<$t> for BigFloat {
            /// The integer exactly, at the precision of the type's bits; zero is `+0`.
            #[allow(unused_comparisons)]
            fn from(x: $t) -> BigFloat {
                let negative = x < 0;
                exact(negative, (x as i128).unsigned_abs(), 0, $bits)
            }
        }
    )*};
}

from_integer!(i8 => 8, i16 => 16, i32 => 32, i64 => 64, isize => 64, u8 => 8, u16 => 16,
    u32 => 32, u64 => 64, usize => 64, i128 => 128);

impl From<u128> for BigFloat {
    /// The integer exactly, at 128 bits; zero is `+0`.
    fn from(x: u128) -> BigFloat {
        exact(false, x, 0, 128)
    }
}

impl From<&Number> for BigFloat {
    /// The integer exactly, at the precision of its bits (1 for zero, which is `+0`).
    fn from(n: &Number) -> BigFloat {
        let m = limbs::of_number(n);
        let prec = limbs::bit_len(&m).max(1);
        if m.is_empty() {
            return BigFloat::zero(1);
        }
        round_mag(n.is_negative(), &m, 0, false, prec, Round::Nearest).0
    }
}

impl From<Number> for BigFloat {
    /// The integer exactly, at the precision of its bits (1 for zero, which is `+0`).
    fn from(n: Number) -> BigFloat {
        BigFloat::from(&n)
    }
}

impl BigFloat {
    /// The integer `n` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    #[track_caller]
    pub fn from_number(n: &Number, prec: u64, round: Round) -> (BigFloat, Ordering) {
        BigFloat::from_parts(n, 0, prec, round)
    }

    /// The rational `r` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::rational::Rational;
    /// use core::cmp::Ordering;
    ///
    /// let third = Rational::from(1) / 3u32;
    /// let (x, o) = BigFloat::from_rational(&third, 53, Round::Nearest);
    /// assert_eq!(x, BigFloat::from(1.0 / 3.0));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn from_rational(r: &Rational, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let a = limbs::of_number(r.numer());
        if a.is_empty() {
            return (BigFloat::zero(prec), Ordering::Equal);
        }
        let b = limbs::of_number(r.denom());
        div_mags(r.is_negative(), &a, 0, &b, 0, prec, round)
    }
}

impl TryFrom<&BigFloat> for Rational {
    type Error = TryFromBigFloatError;

    /// The rational that a finite number is, exactly, in little-endian bytes; a zero of either
    /// sign is zero. NaN and the infinities are an error.
    fn try_from(x: &BigFloat) -> Result<Rational, TryFromBigFloatError> {
        match x.mantissa_exponent() {
            Some((m, e)) => Ok(if e >= 0 {
                Rational::from_integer(m << e as usize)
            } else {
                let z = m.trailing_zeros().unwrap_or(0).min(e.unsigned_abs());
                let den = Number::new_from_u8(1, ByteOrder::LittleEndian)
                    << (e.unsigned_abs() - z) as usize;
                Rational::new(m >> z as usize, den)
            }),
            None if x.is_zero() => Ok(Rational::from_integer(Number::new_from_u8(
                0,
                ByteOrder::LittleEndian,
            ))),
            None => Err(TryFromBigFloatError(())),
        }
    }
}

impl TryFrom<BigFloat> for Rational {
    type Error = TryFromBigFloatError;

    /// As [`TryFrom<&BigFloat>`](#impl-TryFrom%3C%26BigFloat%3E-for-Rational).
    fn try_from(x: BigFloat) -> Result<Rational, TryFromBigFloatError> {
        Rational::try_from(&x)
    }
}
