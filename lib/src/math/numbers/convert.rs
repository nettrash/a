//! Conversions between numbers and the primitive types: [`From`] every primitive integer,
//! [`TryFrom`] a number to each of them, and [`Number::to_f64`] and [`Number::to_f32`], correctly
//! rounded.

use super::cmp::Magnitude;
use super::{ByteOrder, Number};
use core::fmt;

/// The error of a conversion of a [`Number`] to a primitive integer type that cannot hold it.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = Number::new_from_i32(-1, ByteOrder::BigEndian);
/// assert!(u8::try_from(&n).is_err());
/// assert_eq!(i8::try_from(&n), Ok(-1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromNumberError(());

impl fmt::Display for TryFromNumberError {
    /// Writes what was wrong, in the words of the primitive integers' `TryFromIntError`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "out of range integral type conversion attempted".fmt(f)
    }
}

impl std::error::Error for TryFromNumberError {}

/// The sign and the magnitude of `n` when the magnitude fits in 16 bytes: the sign is never set
/// for zero.
#[inline(always)]
fn small_parts(n: &Number) -> Option<(bool, u128)> {
    let m = Magnitude::of(n);
    (m.bytes.len() <= 16).then(|| (m.negative, m.small()))
}

/// `From` and `TryFrom` between numbers and the unsigned primitive integer types.
macro_rules! unsigned {
    ($($t:ident $new:ident),* $(,)?) => {$(
        impl From<$t> for Number {
            #[doc = concat!("Converts a `", stringify!($t), "` to a number, little-endian, as [`Number::", stringify!($new), "`] builds it.")]
            #[inline]
            fn from(value: $t) -> Number {
                Number::$new(value, ByteOrder::LittleEndian)
            }
        }

        impl TryFrom<&Number> for $t {
            type Error = TryFromNumberError;

            #[doc = concat!("Converts a number to a `", stringify!($t), "`, or fails when the number is negative or too large for it.")]
            #[inline]
            fn try_from(n: &Number) -> Result<$t, TryFromNumberError> {
                match small_parts(n) {
                    Some((false, m)) => <$t>::try_from(m).map_err(|_| TryFromNumberError(())),
                    _ => Err(TryFromNumberError(())),
                }
            }
        }

        impl TryFrom<Number> for $t {
            type Error = TryFromNumberError;

            #[doc = concat!("Converts a number to a `", stringify!($t), "`, as from a borrowed number.")]
            #[inline]
            fn try_from(n: Number) -> Result<$t, TryFromNumberError> {
                <$t>::try_from(&n)
            }
        }
    )*};
}

/// `From` and `TryFrom` between numbers and the signed primitive integer types, with the
/// unsigned type of the same width.
macro_rules! signed {
    ($($t:ident $u:ident $new:ident),* $(,)?) => {$(
        impl From<$t> for Number {
            #[doc = concat!("Converts an `", stringify!($t), "` to a number, little-endian, as [`Number::", stringify!($new), "`] builds it.")]
            #[inline]
            fn from(value: $t) -> Number {
                Number::$new(value, ByteOrder::LittleEndian)
            }
        }

        impl TryFrom<&Number> for $t {
            type Error = TryFromNumberError;

            #[doc = concat!("Converts a number to an `", stringify!($t), "`, or fails when it is out of the type's range.")]
            #[inline]
            fn try_from(n: &Number) -> Result<$t, TryFromNumberError> {
                let Some((negative, m)) = small_parts(n) else {
                    return Err(TryFromNumberError(()));
                };
                let m = <$u>::try_from(m).map_err(|_| TryFromNumberError(()))?;
                if negative {
                    // The magnitude of `MIN` is one more than `MAX`: it is the most a negative
                    // number may have, and its two's complement is `MIN` itself.
                    if m > <$t>::MIN.unsigned_abs() {
                        return Err(TryFromNumberError(()));
                    }
                    Ok((m as $t).wrapping_neg())
                } else {
                    <$t>::try_from(m).map_err(|_| TryFromNumberError(()))
                }
            }
        }

        impl TryFrom<Number> for $t {
            type Error = TryFromNumberError;

            #[doc = concat!("Converts a number to an `", stringify!($t), "`, as from a borrowed number.")]
            #[inline]
            fn try_from(n: Number) -> Result<$t, TryFromNumberError> {
                <$t>::try_from(&n)
            }
        }
    )*};
}

unsigned!(
    u8 new_from_u8,
    u16 new_from_u16,
    u32 new_from_u32,
    u64 new_from_u64,
    u128 new_from_u128,
    usize new_from_usize,
);
signed!(
    i8 u8 new_from_i8,
    i16 u16 new_from_i16,
    i32 u32 new_from_i32,
    i64 u64 new_from_i64,
    i128 u128 new_from_i128,
    isize usize new_from_isize,
);

/// The number of bits and the top 64 bits of the magnitude `m`, of more than 16 bytes: the bits
/// from the most significant set one down, shifted up so that it is bit 63.
#[inline(always)]
fn top_bits(m: Magnitude) -> (u64, u64) {
    let v = m.bytes;
    let n = v.len();
    debug_assert!(n > 16);
    // The eight most significant bytes and the ninth.
    let (high, ninth) = if m.big_endian {
        (
            u64::from_be_bytes(v[..8].try_into().unwrap_or_default()),
            v[8],
        )
    } else {
        (
            u64::from_le_bytes(v[n - 8..].try_into().unwrap_or_default()),
            v[n - 9],
        )
    };
    let lz = high.leading_zeros();
    debug_assert!(lz < 8);
    let bits = 8 * n as u64 - u64::from(lz);
    // The ninth byte supplies the `lz` bits that the shift brings in at the bottom.
    (bits, (high << lz) | (u64::from(ninth) >> (8 - lz)))
}

/// Whether any bit of the magnitude below its top `64` bits is set: of its `bits` bits, the
/// `bits - 64` lowest.
#[cold]
fn sticky(m: Magnitude, bits: u64) -> bool {
    let low = bits - 64;
    let (whole, part) = ((low / 8) as usize, (low % 8) as u32);
    let v = m.bytes;
    let n = v.len();
    let (bytes, partial) = if m.big_endian {
        (&v[n - whole..], v[n - whole - 1])
    } else {
        (&v[..whole], v[whole])
    };
    bytes.iter().any(|&b| b != 0) || partial & ((1u8 << part) - 1) != 0
}

impl Number {
    /// Returns the number as the nearest `f64`, ties to even, as the conversions of the primitive
    /// integers with `as` do; infinity, with the number's sign, when it is too large.
    ///
    /// A magnitude of up to 16 bytes is converted as a `u128` is. Of a longer one, the top 64 bits
    /// are converted by the hardware and scaled by a power of two, which is exact: the bits below
    /// them can only matter when the top ones are exactly halfway between two `f64`s, and are read
    /// only then.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i64(-12345, ByteOrder::BigEndian).to_f64(), -12345.0);
    /// // 2^53 + 1 is halfway between two f64s: to the even one.
    /// assert_eq!(Number::new_from_u64((1 << 53) + 1, ByteOrder::BigEndian).to_f64(), 9007199254740992.0);
    /// let n = Number::from_str_radix(&"9".repeat(400), 10, ByteOrder::LittleEndian).unwrap();
    /// assert_eq!(n.to_f64(), f64::INFINITY);
    /// ```
    pub fn to_f64(&self) -> f64 {
        let m = Magnitude::of(self);
        if m.bytes.len() <= 16 {
            // Rounded once, as the conversions of `u128` round.
            let magnitude = m.small() as f64;
            return if m.negative { -magnitude } else { magnitude };
        }
        let (bits, top) = top_bits(m);
        let magnitude = if bits > 1024 {
            f64::INFINITY
        } else {
            // The 53 bits of an `f64` are the top ones, and bits 0 to 10 decide the rounding:
            // below `0x400` down, above up, and at `0x400` exactly the bits below the top 64 do.
            let halfway = top & 0x7FF == 0x400;
            let top = top | u64::from(halfway && sticky(m, bits));
            // `2^(bits - 64)`, from `2^65` to `2^960`.
            let scale = f64::from_bits((bits - 64 + 1023) << 52);
            top as f64 * scale
        };
        if m.negative { -magnitude } else { magnitude }
    }

    /// Returns the number as the nearest `f32`, ties to even, as the conversions of the primitive
    /// integers with `as` do; infinity, with the number's sign, when it is too large. It is
    /// rounded once, from the number itself: rounding the [`to_f64`](Self::to_f64) to an `f32`
    /// would round twice, and could be one off.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::new_from_i64(-12345, ByteOrder::BigEndian).to_f32(), -12345.0);
    /// assert_eq!(Number::new_from_u128(u128::MAX, ByteOrder::BigEndian).to_f32(), f32::INFINITY);
    /// ```
    pub fn to_f32(&self) -> f32 {
        let m = Magnitude::of(self);
        // Above 16 bytes a magnitude is at least `2^128`, beyond the largest `f32` and halfway to
        // the next power of two: infinite.
        let magnitude = if m.bytes.len() <= 16 {
            // Rounded once, as the conversions of `u128` round.
            m.small() as f32
        } else {
            f32::INFINITY
        };
        if m.negative { -magnitude } else { magnitude }
    }
}
