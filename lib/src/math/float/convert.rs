//! Exact conversions into a [`BigFloat`] from the primitive floats and integers and from
//! [`Number`]s, at the precision that holds them, and rounded ones to any precision; a
//! [`Rational`] rounded to a precision; a finite `BigFloat` into the `Rational` it is, exactly;
//! and a `BigFloat` into the primitive floats and integers and into [`Number`]s, rounded in a
//! mode or exactly.

use super::arith::div_mags;
use super::limbs;
use super::round::{away, round_mag, signed};
use super::scale::{Dir, decide, int_parts};
use super::small::{self, Fit};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::{ByteOrder, Number};
use crate::math::rational::Rational;
use core::cmp::Ordering;
use core::fmt;

/// The error of a conversion of a [`BigFloat`] into a type that cannot hold its value: NaN or an
/// infinity into a [`Rational`], or a number that is not an integer, or not one in the type's
/// range, into a [`Number`] or a primitive integer.
///
/// # Examples
///
/// ```
/// use a::math::float::BigFloat;
/// use a::math::numbers::Number;
///
/// assert!(Number::try_from(&BigFloat::from(2.5)).is_err());
/// assert!(u8::try_from(&BigFloat::from(256)).is_err());
/// assert_eq!(i8::try_from(&BigFloat::from(-128)), Ok(-128));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TryFromBigFloatError(());

impl fmt::Display for TryFromBigFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "the float has no value of the type: NaN, infinite, not an integer or out of range",
        )
    }
}

impl core::error::Error for TryFromBigFloatError {}

/// `±m·2^lsb` at the precision `prec` when it holds the `m` (not zero) exactly, without the
/// general rounding: the mantissa shifted to the top of its limbs.
#[inline]
fn word(negative: bool, m: u64, lsb: i64, prec: u64) -> BigFloat {
    debug_assert!(m != 0 && u64::from(64 - m.leading_zeros()) <= prec);
    let n = prec.div_ceil(64) as usize;
    let lz = m.leading_zeros();
    let mut mant = vec![0u64; n];
    mant[n - 1] = m << lz;
    BigFloat {
        prec,
        kind: Kind::Finite {
            negative,
            exp: lsb + i64::from(64 - lz),
            mant,
        },
    }
}

/// `±m·2^e` at the precision `prec`, exactly: `m` must have at most `prec` bits.
fn exact(negative: bool, m: u128, e: i64, prec: u64) -> BigFloat {
    if m == 0 {
        return BigFloat {
            prec,
            kind: Kind::Zero { negative },
        };
    }
    if m >> 64 == 0 {
        return word(negative, m as u64, e, prec);
    }
    let mut v = vec![m as u64, (m >> 64) as u64];
    limbs::trim(&mut v);
    round_mag(negative, &v, i128::from(e), false, prec, Round::Nearest).0
}

/// The parts of an `f64`: its sign and `m·2^e` for a finite one, or the special value.
#[inline]
fn f64_parts(x: f64) -> Result<(bool, u64, i64), Kind> {
    let bits = x.to_bits();
    let negative = bits >> 63 == 1;
    let (field, frac) = ((bits >> 52 & 0x7ff) as i64, bits & ((1 << 52) - 1));
    match field {
        0x7ff if frac != 0 => Err(Kind::Nan),
        0x7ff => Err(Kind::Inf { negative }),
        0 if frac == 0 => Err(Kind::Zero { negative }),
        0 => Ok((negative, frac, -1074)),
        _ => Ok((negative, frac | 1 << 52, field - 1075)),
    }
}

/// The parts of an `f32`, as [`f64_parts`].
#[inline]
fn f32_parts(x: f32) -> Result<(bool, u64, i64), Kind> {
    let bits = x.to_bits();
    let negative = bits >> 31 == 1;
    let (field, frac) = (
        (bits >> 23 & 0xff) as i64,
        u64::from(bits & ((1 << 23) - 1)),
    );
    match field {
        0xff if frac != 0 => Err(Kind::Nan),
        0xff => Err(Kind::Inf { negative }),
        0 if frac == 0 => Err(Kind::Zero { negative }),
        0 => Ok((negative, frac, -149)),
        _ => Ok((negative, frac | 1 << 23, field - 150)),
    }
}

/// The value of float parts at the precision `prec` in the mode `round`.
#[inline]
fn of_parts(
    parts: Result<(bool, u64, i64), Kind>,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    match parts {
        Ok((negative, m, e)) => {
            if u64::from(64 - m.leading_zeros()) <= prec {
                return (word(negative, m, e, prec), Ordering::Equal);
            }
            if let Some(r) = round_word(negative, m, e, prec, round) {
                let kind = Kind::Finite {
                    negative,
                    exp: r.exp,
                    mant: vec![r.m],
                };
                return (BigFloat { prec, kind }, r.o);
            }
            round_mag(negative, &[m], i128::from(e), false, prec, round)
        }
        Err(kind) => (BigFloat { prec, kind }, Ordering::Equal),
    }
}

/// `±m·2^e` rounded in a register to `prec` bits, fewer than `m` has (at most 64): `None` out of
/// the exponent range.
#[inline(always)]
fn round_word(negative: bool, m: u64, e: i64, prec: u64, round: Round) -> Option<Fit<u64>> {
    debug_assert!(prec < 64 && m != 0);
    let lz = m.leading_zeros();
    small::fit(
        negative,
        m << lz,
        0,
        e + i64::from(64 - lz),
        false,
        (64 - prec) as u32,
        round,
    )
}

impl BigFloat {
    /// Sets `self` to the float parts `parts` rounded to its precision in the mode `round`,
    /// into its limbs when it has them, and returns the direction of the rounding.
    #[inline]
    fn set_parts(&mut self, parts: Result<(bool, u64, i64), Kind>, round: Round) -> Ordering {
        let prec = self.prec;
        let (negative, m, e) = match parts {
            Ok(p) => p,
            Err(kind) => {
                self.kind = kind;
                return Ordering::Equal;
            }
        };
        let lz = m.leading_zeros();
        if let Kind::Finite {
            negative: sign,
            exp,
            mant,
        } = &mut self.kind
        {
            if u64::from(64 - lz) <= prec {
                let n = mant.len();
                mant[..n - 1].fill(0);
                mant[n - 1] = m << lz;
                *sign = negative;
                *exp = e + i64::from(64 - lz);
                return Ordering::Equal;
            }
            if let Some(r) = round_word(negative, m, e, prec, round) {
                mant[0] = r.m;
                *sign = negative;
                *exp = r.exp;
                return r.o;
            }
        }
        let (x, o) = of_parts(Ok((negative, m, e)), prec, round);
        self.kind = x.kind;
        o
    }

    /// Sets `self` to the float `x` rounded to the precision of `self` in the mode `round`,
    /// reusing its vector, and returns the direction of the rounding, as MPFR's `mpfr_set_d`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut r = BigFloat::zero(4);
    /// assert_eq!(r.set_f64(0.1, Round::Nearest), Ordering::Greater);
    /// assert_eq!(r, BigFloat::from(0.1015625));
    /// ```
    pub fn set_f64(&mut self, x: f64, round: Round) -> Ordering {
        self.set_parts(f64_parts(x), round)
    }

    /// Sets `self` to the float `x` rounded to the precision of `self` in the mode `round`,
    /// reusing its vector, and returns the direction of the rounding, as MPFR's `mpfr_set_flt`.
    pub fn set_f32(&mut self, x: f32, round: Round) -> Ordering {
        self.set_parts(f32_parts(x), round)
    }
}

impl From<f64> for BigFloat {
    /// The float exactly, at 53 bits, NaN, the infinities and the zeros as they are.
    #[inline]
    fn from(x: f64) -> BigFloat {
        of_parts(f64_parts(x), 53, Round::Nearest).0
    }
}

impl From<f32> for BigFloat {
    /// The float exactly, at 24 bits, NaN, the infinities and the zeros as they are.
    #[inline]
    fn from(x: f32) -> BigFloat {
        of_parts(f32_parts(x), 24, Round::Nearest).0
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

// Rounded conversions from the primitive floats

impl BigFloat {
    /// The float `x` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as `mpfr_set_d`: exact at 53 bits or more; NaN, the infinities and the zeros as
    /// they are.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from_f64_round(0.1, 4, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(0.1015625), Ordering::Greater));
    /// assert_eq!(BigFloat::from_f64_round(-0.0, 7, Round::Up).0.is_sign_negative(), true);
    /// ```
    #[track_caller]
    pub fn from_f64_round(x: f64, prec: u64, round: Round) -> (BigFloat, Ordering) {
        of_parts(f64_parts(x), BigFloat::check_prec(prec), round)
    }

    /// The float `x` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as `mpfr_set_flt`: exact at 24 bits or more.
    #[track_caller]
    pub fn from_f32_round(x: f32, prec: u64, round: Round) -> (BigFloat, Ordering) {
        of_parts(f32_parts(x), BigFloat::check_prec(prec), round)
    }
}

// To the primitive floats

/// The binary format of a primitive float: the bits of its significand, the exponent `e` of its
/// least normal number `2^(e-1)` (`0.1·2^e`), that of its largest numbers (`2^(e-1) ≤ |x| <
/// 2^e`), and the exponent of its least subnormal number.
struct Format {
    bits: i64,
    min_normal: i64,
    max: i64,
    tiny: i64,
}

/// `f64`: 53 bits, `2^-1022` to below `2^1024`, subnormals down to `2^-1074`.
const F64: Format = Format {
    bits: 53,
    min_normal: -1021,
    max: 1024,
    tiny: -1074,
};

/// `f32`: 24 bits, `2^-126` to below `2^128`, subnormals down to `2^-149`.
const F32: Format = Format {
    bits: 24,
    min_normal: -125,
    max: 128,
    tiny: -149,
};

impl BigFloat {
    /// The bits of the float of the format `fmt` nearest to the number in the mode `round`, and
    /// the direction of the rounding; NaN aside, its sign apart.
    ///
    /// A number `0.1…·2^e` has `min(bits, e - tiny)` bits in the format, fewer among the
    /// subnormals; those of the mantissa's top limb with the bit below them decide the rounding,
    /// with the lower limbs as a sticky bit. The float's encoding then follows from the rounded
    /// mantissa `t` alone: a normal number is `(e - 2 + bias)·2^(bits-1) + t`, which a carry of
    /// `t` to `2^bits` takes to the next binade, and the infinity past the largest; a subnormal is
    /// `t`, and a carry to `2^(bits-1)` makes it the least normal number.
    #[inline]
    fn to_format(&self, fmt: &Format, round: Round) -> (u64, Ordering) {
        let (negative, exp, mant) = match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => (*negative, *exp, mant.as_slice()),
            _ => unreachable!("a finite number that is not zero"),
        };
        if exp > fmt.max {
            let to_infinity = round == Round::Nearest || away(negative, round);
            let mag = if to_infinity {
                ((2 * fmt.max - 1) as u64) << (fmt.bits - 1)
            } else {
                (((2 * fmt.max - 1) as u64) << (fmt.bits - 1)) - 1
            };
            let o = if to_infinity {
                Ordering::Greater
            } else {
                Ordering::Less
            };
            return (mag, signed(negative, o));
        }
        let top = mant[mant.len() - 1];
        let low = mant[..mant.len() - 1].iter().any(|&x| x != 0);
        let p = fmt.bits.min(exp - fmt.tiny);
        let (t, half, rest) = if p < 0 {
            (0, false, true)
        } else if p == 0 {
            (0, true, top << 1 != 0 || low)
        } else {
            (
                top >> (64 - p),
                top >> (63 - p) & 1 == 1,
                top << (p + 1) != 0 || low,
            )
        };
        let up = match round {
            Round::Nearest => half && (rest || t & 1 == 1),
            _ => away(negative, round) && (half || rest),
        };
        let t = t + u64::from(up);
        let field = (exp - fmt.min_normal).max(0) as u64;
        let mag = (field << (fmt.bits - 1)) + t;
        let o = if !(half || rest) {
            Ordering::Equal
        } else if up {
            Ordering::Greater
        } else {
            Ordering::Less
        };
        (mag, signed(negative, o))
    }

    /// The number rounded to an `f64` in the mode `round`, and the direction of the rounding, as
    /// `mpfr_get_d`: correctly rounded among the subnormals too; past the largest finite `f64`, an
    /// infinity, or the largest when the mode rounds toward zero; NaN as NaN (with
    /// [`Ordering::Equal`]), and the infinities and the zeros with their signs.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let third = BigFloat::from(1).div_round(&BigFloat::from(3), 200, Round::Nearest).0;
    /// assert_eq!(third.to_f64_round(Round::Nearest), (1.0 / 3.0, Ordering::Less));
    /// assert_eq!(third.to_f64_round(Round::Up).0, (1.0f64 / 3.0).next_up());
    /// let huge = BigFloat::from(2).with_prec(10).mul_round(&BigFloat::from(f64::MAX), 10, Round::Up).0;
    /// assert_eq!(huge.to_f64_round(Round::TowardZero), (f64::MAX, Ordering::Less));
    /// assert_eq!(huge.to_f64(), f64::INFINITY);
    /// ```
    pub fn to_f64_round(&self, round: Round) -> (f64, Ordering) {
        match self.kind {
            Kind::Nan => (f64::NAN, Ordering::Equal),
            Kind::Inf { negative } => (
                if negative {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
                Ordering::Equal,
            ),
            Kind::Zero { negative } => (if negative { -0.0 } else { 0.0 }, Ordering::Equal),
            Kind::Finite { negative, .. } => {
                let (mag, o) = self.to_format(&F64, round);
                (f64::from_bits(mag | u64::from(negative) << 63), o)
            }
        }
    }

    /// The number rounded to an `f64` to nearest, as [`to_f64_round`](Self::to_f64_round).
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.to_f64_round(Round::Nearest).0
    }

    /// The number rounded to an `f32` in the mode `round`, and the direction of the rounding, as
    /// `mpfr_get_flt`, as [`to_f64_round`](Self::to_f64_round) does for an `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let x = BigFloat::from(0.1);
    /// assert_eq!(x.to_f32_round(Round::Nearest), (0.1f32, Ordering::Greater));
    /// assert_eq!(BigFloat::from(1e-50).to_f32_round(Round::Up), (f32::from_bits(1), Ordering::Greater));
    /// ```
    pub fn to_f32_round(&self, round: Round) -> (f32, Ordering) {
        match self.kind {
            Kind::Nan => (f32::NAN, Ordering::Equal),
            Kind::Inf { negative } => (
                if negative {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                },
                Ordering::Equal,
            ),
            Kind::Zero { negative } => (if negative { -0.0 } else { 0.0 }, Ordering::Equal),
            Kind::Finite { negative, .. } => {
                let (mag, o) = self.to_format(&F32, round);
                (f32::from_bits(mag as u32 | u32::from(negative) << 31), o)
            }
        }
    }

    /// The number rounded to an `f32` to nearest, as [`to_f32_round`](Self::to_f32_round).
    #[inline]
    pub fn to_f32(&self) -> f32 {
        self.to_f32_round(Round::Nearest).0
    }
}

// To integers

impl BigFloat {
    /// The integer of the sign and magnitude that the finite number, not zero, rounds to in the
    /// direction `dir`, and the direction of the rounding of the magnitude.
    pub(crate) fn integer_mag(&self, dir: Dir) -> (bool, Vec<u64>, Ordering) {
        let (negative, m, lsb) = self.parts().expect("a finite number that is not zero");
        let (t, h, z) = int_parts(m, lsb);
        let (d, o) = decide(t, h, z, dir);
        (negative, d, o)
    }

    /// The integer that the number rounds to in the mode `round`, and the direction of the
    /// rounding, as `mpfr_get_z`; `None` for NaN and the infinities. The integer has as many bits
    /// as the number's exponent: one of `2^60` would not fit in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// let x = BigFloat::from(-2.5);
    /// assert_eq!(x.to_number_round(Round::Nearest), Some((Number::from(-2), Ordering::Greater)));
    /// assert_eq!(x.to_number_round(Round::Down), Some((Number::from(-3), Ordering::Less)));
    /// assert_eq!(BigFloat::nan(5).to_number_round(Round::Up), None);
    /// ```
    pub fn to_number_round(&self, round: Round) -> Option<(Number, Ordering)> {
        match self.kind {
            Kind::Nan | Kind::Inf { .. } => None,
            Kind::Zero { .. } => Some((
                Number::new_from_u8(0, ByteOrder::LittleEndian),
                Ordering::Equal,
            )),
            Kind::Finite { negative, .. } => {
                let (_, d, o) = self.integer_mag(Dir::of(negative, round));
                Some((
                    limbs::to_number(&d, negative, ByteOrder::LittleEndian),
                    signed(negative, o),
                ))
            }
        }
    }

    /// The sign and the magnitude of the integer that the number rounds to in the mode `round`,
    /// `None` for one of more than 128 bits, and the direction of the rounding; for a finite
    /// number.
    fn rounded_u128(&self, round: Round) -> (bool, Option<u128>, Ordering) {
        match &self.kind {
            Kind::Zero { negative } => (*negative, Some(0), Ordering::Equal),
            Kind::Finite { negative, exp, .. } => {
                if *exp > 129 {
                    // At least `2^128`, past any of the types.
                    let o = signed(*negative, Ordering::Less);
                    return (*negative, None, o);
                }
                let (_, d, o) = self.integer_mag(Dir::of(*negative, round));
                let mag = match d.len() {
                    0 => Some(0),
                    1 => Some(u128::from(d[0])),
                    2 => Some(u128::from(d[0]) | u128::from(d[1]) << 64),
                    _ => None,
                };
                (*negative, mag, signed(*negative, o))
            }
            _ => unreachable!("a finite number"),
        }
    }
}

macro_rules! to_int_round {
    ($($name:ident => $t:ty),*) => {$(
        impl BigFloat {
            #[doc = concat!("The `", stringify!($t), "` that the number rounds to in the mode `round`, and the direction of")]
            /// the rounding, as `mpfr_get_si`: saturating like the `as` casts of the primitive
            /// floats, so that a number past the type's range, an infinity included, gives its
            /// largest or least value (and a direction against the number); `None` for NaN.
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::float::{BigFloat, Round};
            /// use core::cmp::Ordering;
            ///
            /// let x = BigFloat::from(-7.5);
            #[doc = concat!("assert_eq!(BigFloat::from(7.5).", stringify!($name), "(Round::Nearest), Some((8, Ordering::Greater)));")]
            #[doc = concat!("assert_eq!(BigFloat::infinity(10).", stringify!($name), "(Round::Up), Some((", stringify!($t), "::MAX, Ordering::Less)));")]
            #[doc = concat!("assert_eq!(BigFloat::nan(10).", stringify!($name), "(Round::Up), None);")]
            /// # let _ = x;
            /// ```
            pub fn $name(&self, round: Round) -> Option<($t, Ordering)> {
                let (negative, mag, o) = match self.kind {
                    Kind::Nan => return None,
                    Kind::Inf { negative } => (negative, None, signed(negative, Ordering::Less)),
                    _ => self.rounded_u128(round),
                };
                let value = mag.and_then(|m| {
                    if negative {
                        if m == 0 {
                            Some(0)
                        } else if m <= 1u128 << 127 {
                            // `-m` as an `i128`, then into the type.
                            <$t>::try_from(-((m - 1) as i128) - 1).ok()
                        } else {
                            None
                        }
                    } else {
                        <$t>::try_from(m).ok()
                    }
                });
                Some(match value {
                    Some(v) => (v, o),
                    None if negative => (<$t>::MIN, Ordering::Greater),
                    None => (<$t>::MAX, Ordering::Less),
                })
            }
        }
    )*};
}

to_int_round!(to_i64_round => i64, to_u64_round => u64, to_i128_round => i128, to_u128_round => u128);

impl TryFrom<&BigFloat> for Number {
    type Error = TryFromBigFloatError;

    /// The integer that a number is, exactly, in little-endian bytes; a zero of either sign is
    /// zero.
    ///
    /// # Errors
    ///
    /// [`TryFromBigFloatError`] for NaN, an infinity, or a number that is not an integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::numbers::Number;
    ///
    /// assert_eq!(Number::try_from(&BigFloat::from(-6.0)), Ok(Number::from(-6)));
    /// assert!(Number::try_from(&BigFloat::from(0.5)).is_err());
    /// ```
    fn try_from(x: &BigFloat) -> Result<Number, TryFromBigFloatError> {
        match x.kind {
            Kind::Zero { .. } => Ok(Number::new_from_u8(0, ByteOrder::LittleEndian)),
            Kind::Finite { negative, .. } => {
                let (_, d, o) = x.integer_mag(Dir::Floor);
                if o == Ordering::Equal {
                    Ok(limbs::to_number(&d, negative, ByteOrder::LittleEndian))
                } else {
                    Err(TryFromBigFloatError(()))
                }
            }
            _ => Err(TryFromBigFloatError(())),
        }
    }
}

impl TryFrom<BigFloat> for Number {
    type Error = TryFromBigFloatError;

    /// As [`TryFrom<&BigFloat>`](#impl-TryFrom%3C%26BigFloat%3E-for-Number).
    fn try_from(x: BigFloat) -> Result<Number, TryFromBigFloatError> {
        Number::try_from(&x)
    }
}

macro_rules! to_primitive {
    ($($t:ty),*) => {$(
        impl TryFrom<&BigFloat> for $t {
            type Error = TryFromBigFloatError;

            #[doc = concat!("The `", stringify!($t), "` that a number is, exactly; a zero of either sign is zero.")]
            ///
            /// # Errors
            ///
            /// [`TryFromBigFloatError`] for NaN, an infinity, a number that is not an integer, or
            /// one out of the type's range.
            fn try_from(x: &BigFloat) -> Result<$t, TryFromBigFloatError> {
                if !x.is_finite() {
                    return Err(TryFromBigFloatError(()));
                }
                let (negative, mag, o) = x.rounded_u128(Round::TowardZero);
                let err = TryFromBigFloatError(());
                if o != Ordering::Equal {
                    return Err(err);
                }
                let m = mag.ok_or(err)?;
                #[allow(unused_comparisons)]
                if negative && m != 0 {
                    if m > 1u128 << 127 {
                        return Err(err);
                    }
                    let v = (-((m - 1) as i128)) - 1;
                    <$t>::try_from(v).map_err(|_| err)
                } else {
                    <$t>::try_from(m).map_err(|_| err)
                }
            }
        }

        impl TryFrom<BigFloat> for $t {
            type Error = TryFromBigFloatError;

            #[doc = concat!("The `", stringify!($t), "` that a number is, as from a borrowed one.")]
            fn try_from(x: BigFloat) -> Result<$t, TryFromBigFloatError> {
                <$t>::try_from(&x)
            }
        }
    )*};
}

to_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
