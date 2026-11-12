//! Conversions between rationals and the primitive types: [`TryFrom`] a float, which is exact,
//! [`Rational::to_f64`] and [`Rational::to_f32`], correctly rounded, and [`TryFrom`] a rational to
//! a [`Number`] or a primitive integer, for the integers.
//!
//! Every finite float is a dyadic rational `m·2^e`, so a float converts exactly: the odd part of
//! its significand over a power of two, or an integer, in words when both fit in 64 bits, with no
//! allocation. The other way, `a/b` is rounded once, to the nearest float, ties to even. When both
//! parts are below `2^53` (`2^24` for `f32`), they are floats exactly, and one division of floats
//! rounds their quotient correctly. Other parts in words are divided as their nearest `f64`s: the
//! quotient, rounded to the format, is within two units in its last place of `a/b`, and the exact
//! remainder of `a/b` by it, in 128 bits, says whether it or a neighbour is the nearest float.
//! Longer parts have the top 128 bits of `a` divided by the top 64 bits of `b`, in registers: the
//! quotient `q`, of 64 or 65 bits, is within 4 of `a/b` scaled by the same power of two, and above
//! it by less than 1, so when no halfway point between two floats lies from `q - 4` to `q + 1`,
//! `a/b` rounds as `q` with a sticky bit does. When `b` has at most 64 bits the quotient and its
//! remainder are exact, and the bits of `a` below its top 128 are read only when the remainder is
//! zero. Only when a halfway point between two floats lies within the bounds, for one fraction in
//! 700 of random parts (measured), is `a·2^s` divided by `b`, for the `s` that gives a quotient of
//! 65 or 66 bits, its remainder deciding the sticky bit: one division whose cost is linear in the
//! length of `b`.

use super::{Rational, one};
use crate::math::numbers::{ByteOrder, Magnitude, Number, divide_2by1, reciprocal};
use core::fmt;

/// The error of a conversion of a float that is not a number or infinite to a [`Rational`].
///
/// # Examples
///
/// ```
/// use a::math::rational::Rational;
///
/// assert!(Rational::try_from(f64::NAN).is_err());
/// assert!(Rational::try_from(f32::NEG_INFINITY).is_err());
/// assert_eq!(Rational::try_from(0.75).unwrap().to_string(), "3/4");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromFloatError(());

impl fmt::Display for TryFromFloatError {
    /// Writes what was wrong.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "cannot convert a NaN or an infinite float to a rational".fmt(f)
    }
}

impl std::error::Error for TryFromFloatError {}

/// The error of a conversion of a [`Rational`] that is not an integer, or not one in the range of
/// the type, to a [`Number`] or a primitive integer.
///
/// # Examples
///
/// ```
/// use a::math::numbers::Number;
/// use a::math::rational::Rational;
///
/// let half = Rational::from(1) / 2u32;
/// assert!(Number::try_from(&half).is_err());
/// assert!(u8::try_from(&Rational::from(256)).is_err());
/// assert_eq!(i8::try_from(&Rational::from(-128)), Ok(-128));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromRationalError(());

impl fmt::Display for TryFromRationalError {
    /// Writes what was wrong.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "the rational is not an integer in the range of the type".fmt(f)
    }
}

impl std::error::Error for TryFromRationalError {}

// Floats to rationals

/// `±m·2^e` as a rational, little-endian, for a non-zero `m` of at most 53 bits and `e` from -1074
/// to 971.
#[inline(always)]
fn dyadic(negative: bool, m: u64, e: i32) -> Rational {
    let tz = m.trailing_zeros();
    let (m, e) = (m >> tz, e + tz as i32);
    // In words when both parts fit in 64 bits: `m·2^e` below `2^64`, or `m/2^-e` up to `2^63`.
    let order = ByteOrder::LittleEndian;
    if (0..64).contains(&e) && m.leading_zeros() >= e as u32 {
        return Rational::from_word(negative, m << e, 1, order);
    }
    if (-63..0).contains(&e) {
        return Rational::from_word(negative, m, 1 << -e, order);
    }
    long_dyadic(negative, m, e)
}

/// [`dyadic`] of an odd `m` and an `e` that make a part of more than 64 bits: the numerator
/// `m·2^e` or the denominator `2^-e` written straight into its vector.
#[inline(never)]
fn long_dyadic(negative: bool, m: u64, e: i32) -> Rational {
    if e >= 0 {
        // `m·2^e`: `e / 8` zero bytes, then `m` shifted by `e % 8` bits.
        let e = e as usize;
        let top = u128::from(m) << (e % 8);
        let len = 16 - top.leading_zeros() as usize / 8;
        let mut value = Vec::with_capacity(e / 8 + len);
        value.resize(e / 8, 0);
        value.extend_from_slice(&top.to_le_bytes()[..len]);
        let num = Number {
            value,
            negative,
            byte_order: ByteOrder::LittleEndian,
        };
        return Rational::big(num, one(ByteOrder::LittleEndian));
    }
    // `m/2^-e`, with `m` odd: in lowest terms as it is.
    let len = 8 - m.leading_zeros() as usize / 8;
    let num = Number {
        value: m.to_le_bytes()[..len].to_vec(),
        negative,
        byte_order: ByteOrder::LittleEndian,
    };
    let k = (-e) as usize;
    let mut value = vec![0u8; k / 8 + 1];
    value[k / 8] = 1 << (k % 8);
    let den = Number {
        value,
        negative: false,
        byte_order: ByteOrder::LittleEndian,
    };
    Rational::big(num, den)
}

impl TryFrom<f64> for Rational {
    type Error = TryFromFloatError;

    /// The float `x` exactly, as the fraction `m/2^k` in lowest terms or an integer, little-endian;
    /// `-0.0` is zero.
    ///
    /// # Errors
    ///
    /// [`TryFromFloatError`] when `x` is not a number or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!(Rational::try_from(-2.5).unwrap().to_string(), "-5/2");
    /// // 0.1 is not one tenth, but the float nearest to it.
    /// assert_eq!(Rational::try_from(0.1).unwrap().to_string(), "3602879701896397/36028797018963968");
    /// assert_eq!(Rational::try_from(1e20).unwrap().to_string(), "100000000000000000000");
    /// ```
    #[inline]
    fn try_from(x: f64) -> Result<Rational, TryFromFloatError> {
        let bits = x.to_bits();
        let field = ((bits >> 52) & 0x7FF) as i32;
        let fraction = bits & ((1 << 52) - 1);
        if field == 0x7FF {
            return Err(TryFromFloatError(()));
        }
        let (m, e) = if field == 0 {
            (fraction, -1074)
        } else {
            (fraction | 1 << 52, field - 1075)
        };
        if m == 0 {
            return Ok(Rational::default());
        }
        Ok(dyadic(bits >> 63 == 1, m, e))
    }
}

impl TryFrom<f32> for Rational {
    type Error = TryFromFloatError;

    /// The float `x` exactly, as [`TryFrom<f64>`](#impl-TryFrom<f64>-for-Rational) has it; `-0.0`
    /// is zero.
    ///
    /// # Errors
    ///
    /// [`TryFromFloatError`] when `x` is not a number or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!(Rational::try_from(0.1f32).unwrap().to_string(), "13421773/134217728");
    /// assert!(Rational::try_from(f32::NAN).is_err());
    /// ```
    #[inline]
    fn try_from(x: f32) -> Result<Rational, TryFromFloatError> {
        // Every `f32` is an `f64` exactly.
        Rational::try_from(f64::from(x))
    }
}

// Rationals to floats

/// The parameters of a binary floating-point format.
trait Format {
    /// The bits of the significand, the implicit one with them.
    const P: i64;
    /// The exponent of the smallest normal number.
    const EMIN: i64;
    /// The exponent of the largest finite number.
    const EMAX: i64;
    /// The bits of the nearest float to `x / y`, for `x` and `y` below `2^P`, `y` not zero: one
    /// division of floats, correctly rounded.
    fn quotient(x: u64, y: u64) -> u64;
    /// The bits of a float within a few units in the last place of `x / y`, for any `x` and `y`
    /// not zero: their nearest `f64`s divided, rounded to the format.
    fn estimate(x: u64, y: u64) -> u64;
}

/// `f64`.
struct Double;

/// `f32`.
struct Single;

impl Format for Double {
    const P: i64 = 53;
    const EMIN: i64 = -1022;
    const EMAX: i64 = 1023;
    #[inline(always)]
    fn quotient(x: u64, y: u64) -> u64 {
        (x as f64 / y as f64).to_bits()
    }
    #[inline(always)]
    fn estimate(x: u64, y: u64) -> u64 {
        (x as f64 / y as f64).to_bits()
    }
}

impl Format for Single {
    const P: i64 = 24;
    const EMIN: i64 = -126;
    const EMAX: i64 = 127;
    #[inline(always)]
    fn quotient(x: u64, y: u64) -> u64 {
        u64::from((x as f32 / y as f32).to_bits())
    }
    #[inline(always)]
    fn estimate(x: u64, y: u64) -> u64 {
        u64::from(((x as f64 / y as f64) as f32).to_bits())
    }
}

/// The bits of the magnitude of the float of the format `F` nearest to `m·2^exp`, ties to even,
/// or to a number a little above `m·2^exp`, strictly between it and `(m + 1)·2^exp`, when `sticky`
/// is set; infinity when it is too large, zero when it is too small. `m` has more than `F::P`
/// bits, so that at least one is rounded off.
#[inline(always)]
fn round<F: Format>(m: u128, sticky: bool, exp: i64) -> u64 {
    round_near::<F>(m, sticky, exp).0
}

/// [`round`], and how far `m` is above the halfway point below it, between the float its bits
/// above those rounded off make and the next: `m` less that point, wrapped, so that it is below 4
/// when a halfway point lies in `(m - 4, m]`. The point is the only one within `2^(k - 1)` of `m`,
/// for the `k ≥ 10` bits rounded off; when `m` and `m - 4` are in two binades, or round to
/// infinity or to zero, they round alike but for the halfway point between zero and the smallest
/// subnormal, which this counts.
#[inline(always)]
fn round_near<F: Format>(m: u128, sticky: bool, exp: i64) -> (u64, u128) {
    let len = 128 - i64::from(m.leading_zeros());
    debug_assert!(len > F::P);
    // `m·2^exp` is in `[2^e, 2^(e+1))`.
    let e = len - 1 + exp;
    if e > F::EMAX {
        return (((2 * F::EMAX + 1) as u64) << (F::P - 1), u128::MAX);
    }
    // The bits kept: `P` for a normal number, fewer for a subnormal one, none at all for a number
    // in `[2^(EMIN-P), 2^(EMIN-P+1))`, which rounds to zero or to the smallest subnormal.
    let keep = if e >= F::EMIN {
        F::P
    } else {
        F::P - (F::EMIN - e)
    };
    if keep < 0 {
        return (0, u128::MAX);
    }
    let k = (len - keep) as u32;
    let kept = m.checked_shr(k).unwrap_or(0) as u64;
    // Without branches, which the bits of random numbers would mislead.
    let half = (m >> (k - 1)) & 1 == 1;
    let rest = sticky | (m & ((1u128 << (k - 1)) - 1) != 0);
    let near = (m & ((1u128 << (k - 1) << 1).wrapping_sub(1))).wrapping_sub(1u128 << (k - 1));
    let kept = kept + u64::from(half & (rest | (kept & 1 == 1)));
    let bits = if e >= F::EMIN {
        // The significand with its implicit bit adds one to the exponent field, and a carry out of
        // it, to `2^P`, one more: up to infinity's bits from the largest exponent.
        (((e + F::EMAX - 1) as u64) << (F::P - 1)) + kept
    } else {
        // A subnormal number, or the smallest normal one when the rounding carries into it.
        kept
    };
    (bits, near)
}

/// The number of bits of the non-empty magnitude `m` and its top 128 bits, the most significant
/// set one at bit 127: of a short magnitude all its bits, shifted up.
#[inline(always)]
fn top_128(m: Magnitude) -> (i64, u128) {
    let v = m.bytes;
    let n = v.len();
    if n <= 16 {
        let x = m.small();
        let lz = x.leading_zeros();
        return (128 - i64::from(lz), x << lz);
    }
    let (high, next) = if m.big_endian {
        (
            u128::from_be_bytes(v[..16].try_into().unwrap_or_default()),
            v[16],
        )
    } else {
        (
            u128::from_le_bytes(v[n - 16..].try_into().unwrap_or_default()),
            v[n - 17],
        )
    };
    let lz = high.leading_zeros();
    let bits = 8 * n as i64 - i64::from(lz);
    // The seventeenth byte supplies the `lz` bits that the shift brings in at the bottom.
    (bits, (high << lz) | (u128::from(next) >> (8 - lz)))
}

/// The number of bits of the non-empty magnitude `m` and its top 64 bits, the most significant set
/// one at bit 63: of a short magnitude all its bits, shifted up.
#[inline(always)]
fn top_64(m: Magnitude) -> (i64, u64) {
    let v = m.bytes;
    let n = v.len();
    if n <= 8 {
        let x = m.small() as u64;
        let lz = x.leading_zeros();
        return (64 - i64::from(lz), x << lz);
    }
    let (high, next) = if m.big_endian {
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
    let bits = 8 * n as i64 - i64::from(lz);
    (bits, (high << lz) | (u64::from(next) >> (8 - lz)))
}

/// Whether any of the `count` lowest bits of the magnitude `m`, fewer than it has, is set.
#[cold]
fn low_bits(m: Magnitude, count: u64) -> bool {
    let (whole, part) = ((count / 8) as usize, (count % 8) as u32);
    let v = m.bytes;
    let n = v.len();
    let (bytes, partial) = if m.big_endian {
        (&v[n - whole..], v[n - whole - 1])
    } else {
        (&v[..whole], v[whole])
    };
    bytes.iter().any(|&b| b != 0) || partial & ((1u8 << part) - 1) != 0
}

/// The bits of the magnitude of the float of the format `F` nearest to `|a|/b`, for the
/// magnitudes of a non-zero numerator and of a denominator, one of more than 64 bits (parts in
/// words take [`word_quotient`]).
#[inline(always)]
fn quotient<F: Format>(a: Magnitude, b: Magnitude) -> u64 {
    let (na, ta) = top_128(a);
    let (nb, tb) = top_64(b);
    // `|a|/b = (ta + fa)/(tb + fb) · 2^exp` for fractions `fa` and `fb` of a unit, zero when the
    // magnitudes have no more bits than are read.
    // `|a|/b` is in `[2^(na - nb - 1), 2^(na - nb + 1))`: past the largest finite number, or
    // below half the smallest subnormal one.
    if na - nb > F::EMAX + 1 {
        return ((2 * F::EMAX + 1) as u64) << (F::P - 1);
    }
    if na - nb < F::EMIN - F::P {
        return 0;
    }
    let exp = (na - 128) - (nb - 64);
    // `ta/tb`, of 64 or 65 bits, by the reciprocal of `tb`, as a division of limbs takes it.
    let (high, low) = ((ta >> 64) as u64, ta as u64);
    let top = high >= tb;
    let high = high - (tb & u64::from(top).wrapping_neg());
    let (q, r) = divide_2by1(high, low, tb, reciprocal(tb));
    let q = u128::from(top) << 64 | u128::from(q);
    if nb <= 64 {
        // `fb = 0`: `(ta + fa)/tb` is `q + (r + fa)/tb`, below `q + 1`, and `q` exactly when
        // `r` and `fa` are zero.
        let sticky = r != 0 || na > 128 && low_bits(a, (na - 128) as u64);
        return round::<F>(q, sticky, exp);
    }
    // Above `q - ta/(tb (tb + 1))`, more than `q - 4`, and below `(ta + 1)/tb`, at most `q + 1`.
    let (bits, near) = round_near::<F>(q, true, exp);
    if near >= 4 {
        return bits;
    }
    exact::<F>(a, b, na, nb)
}

/// [`quotient`] by a division of `|a|·2^s` by `b`, whose quotient has 65 or 66 bits.
#[cold]
#[inline(never)]
fn exact<F: Format>(a: Magnitude, b: Magnitude, na: i64, nb: i64) -> u64 {
    let number = |m: Magnitude| {
        let mut value = m.bytes.to_vec();
        if m.big_endian {
            value.reverse();
        }
        Number {
            value,
            negative: false,
            byte_order: ByteOrder::LittleEndian,
        }
    };
    // `a·2^s/b` is in `(2^(na - nb - 1 + s), 2^(na - nb + 1 + s))`, from `2^64` to `2^66`.
    let s = 65 - (na - nb);
    let (x, dropped) = if s >= 0 {
        (number(a) << s as usize, false)
    } else {
        // `floor(floor(a/2^k)/b) = floor(a/(2^k b))`: the bits below `2^k` only set the sticky bit.
        let k = (-s) as u64;
        (number(a) >> k as usize, low_bits(a, k))
    };
    let (q, r) = x.div_rem(&number(b));
    let q = Magnitude::of(&q).small();
    round::<F>(q, dropped || !r.is_zero(), -s)
}

/// The bits of the float of the format `F` nearest to `a/b`, ties to even, for parts of a rational
/// in words: `a` and `b` not zero.
///
/// When both are below `2^P` they are floats exactly, and one division rounds their quotient
/// correctly. Otherwise the quotient of their nearest `f64`s, rounded to the format, is a float
/// `c = m·2^e` within two units in its last place of `a/b`, and the exact remainder `r = a·2^-e -
/// m·b`, in 128 bits, says where `a/b = (m + r/b)·2^e` lies against the halfway points `m ± 1/2`:
/// beyond one, the neighbour is the nearer, and is checked in turn; on one, the even of the two.
/// `a/b` is from `2^-64` to `2^64`, a normal number in both formats.
#[inline(always)]
fn word_quotient<F: Format>(a: u64, b: u64) -> u64 {
    if (a | b) >> F::P == 0 {
        return F::quotient(a, b);
    }
    // The implicit bit of a significand, and the exponent of the unit in the last place of a float
    // whose exponent field is zero.
    let hidden = 1u64 << (F::P - 1);
    let offset = F::EMAX + F::P - 1;
    let mut c = F::estimate(a, b);
    loop {
        let m = (c & (hidden - 1)) | hidden;
        let e = (c >> (F::P - 1)) as i64 - offset;
        // `a·2^-e` and `m·b` scaled alike, with `b` scaled as they are: `(m + r/unit)·2^e = a/b`.
        // Both sides are below `2^118`, as `m·2^e` is within a few units of `a/b`.
        let p = u128::from(m) * u128::from(b);
        let (x, y, unit) = if e <= 0 {
            (u128::from(a) << -e, p, u128::from(b))
        } else {
            (u128::from(a), p << e, u128::from(b) << e)
        };
        let (r, unit) = (x as i128 - y as i128, unit as i128);
        let d = 2 * r;
        if d > unit {
            c += 1;
            continue;
        }
        if d == unit {
            // Halfway up: to the even significand, which a carry into the next binade is too. A
            // fraction in lowest terms is halfway only over a power of two, which the estimate
            // rounds itself, but the checks keep the quotient exact for any parts.
            return c + (m & 1);
        }
        if m == hidden && r < 0 {
            // At the bottom of a binade the float below is half a unit away, and the halfway point
            // to it a quarter.
            if d < -unit {
                c -= 1;
                continue;
            }
            return c - u64::from(4 * r < -unit);
        }
        if d < -unit {
            c -= 1;
            continue;
        }
        return c - (m & 1) * u64::from(d == -unit);
    }
}

impl Rational {
    /// Returns the number as the nearest `f64`, ties to even: rounded once, correctly, subnormal
    /// numbers included; infinity, of the number's sign, when it is too large, and zero when it is
    /// too small, `-0.0` for a negative number as IEEE 754 rounds it, `0.0` for zero.
    ///
    /// When both parts are below `2^53`, it is their quotient as floats; other parts of up to 64
    /// bits take the quotient of their nearest floats, checked by the exact remainder in 128 bits.
    /// Otherwise the top 128 bits of the numerator are divided by the top 64 of the denominator, in
    /// registers, which decides the rounding but for about one fraction in 700, and those are
    /// divided at the length of the denominator, the remainder setting the sticky bit: the time is
    /// constant but for them.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(1) / 3u32).to_f64(), 1.0 / 3.0);
    /// assert_eq!((Rational::from(-7) / 2u32).to_f64(), -3.5);
    /// // 2^53 + 1 is halfway between two f64s: to the even one.
    /// assert_eq!(Rational::from((1u64 << 53) + 1).to_f64(), 9007199254740992.0);
    /// let tiny = Rational::new(Number::from(-1), Number::from(1) << 1100);
    /// assert_eq!(tiny.to_f64().to_bits(), (-0.0f64).to_bits());
    /// ```
    pub fn to_f64(&self) -> f64 {
        let Some(x) = self.word() else {
            // Not zero, which is in words.
            let (a, d) = self.magnitudes();
            let bits = quotient::<Double>(a, d);
            return f64::from_bits(bits | u64::from(a.negative) << 63);
        };
        if x.num == 0 {
            return 0.0;
        }
        let bits = word_quotient::<Double>(x.num, x.den);
        f64::from_bits(bits | u64::from(x.negative) << 63)
    }

    /// Returns the number as the nearest `f32`, ties to even, as [`to_f64`](Self::to_f64) rounds
    /// to an `f64`. It is rounded once, from the number itself: rounding the `to_f64` to an `f32`
    /// would round twice, and could be one off.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(1) / 3u32).to_f32(), 1.0f32 / 3.0);
    /// // Halfway between two f32s, and a little above: up. Through an f64 it would be down.
    /// let r = Rational::from((1u64 << 24) + 1) + Rational::from(1) / (1u64 << 40);
    /// assert_eq!(r.to_f32(), 16777218.0);
    /// assert_eq!(r.to_f64() as f32, 16777216.0);
    /// ```
    pub fn to_f32(&self) -> f32 {
        if let Some(x) = self.word() {
            if x.num == 0 {
                return 0.0;
            }
            let bits = word_quotient::<Single>(x.num, x.den) as u32;
            return f32::from_bits(bits | u32::from(x.negative) << 31);
        }
        let (a, d) = self.magnitudes();
        let bits = quotient::<Single>(a, d) as u32;
        f32::from_bits(bits | u32::from(a.negative) << 31)
    }
}

// Rationals to integers

impl TryFrom<&Rational> for Number {
    type Error = TryFromRationalError;

    /// The numerator of an integer, in new vectors, in the byte order of the rational.
    ///
    /// # Errors
    ///
    /// [`TryFromRationalError`] when the rational is not an integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(6) / 3u32;
    /// assert_eq!(Number::try_from(&r), Ok(Number::from(2)));
    /// assert!(Number::try_from(&(Rational::from(5) / 3u32)).is_err());
    /// ```
    #[inline]
    fn try_from(r: &Rational) -> Result<Number, TryFromRationalError> {
        if r.is_integer() {
            Ok(r.numer_number())
        } else {
            Err(TryFromRationalError(()))
        }
    }
}

impl TryFrom<Rational> for Number {
    type Error = TryFromRationalError;

    /// The numerator of an integer, in its vector.
    ///
    /// # Errors
    ///
    /// [`TryFromRationalError`] when the rational is not an integer.
    #[inline]
    fn try_from(r: Rational) -> Result<Number, TryFromRationalError> {
        if !r.is_integer() {
            return Err(TryFromRationalError(()));
        }
        if r.word().is_some() {
            // One new vector, not the two of `into_parts`.
            return Ok(r.numer_number());
        }
        Ok(r.into_parts().0)
    }
}

/// `TryFrom` a rational to each primitive integer type.
macro_rules! to_primitive {
    ($($t:ty),*) => {$(
        impl TryFrom<&Rational> for $t {
            type Error = TryFromRationalError;

            #[doc = concat!("Converts an integer to a `", stringify!($t), "`, or fails when the rational is not an integer or is out of the type's range.")]
            #[inline]
            fn try_from(r: &Rational) -> Result<$t, TryFromRationalError> {
                if let Some(x) = r.word() {
                    if x.den != 1 {
                        return Err(TryFromRationalError(()));
                    }
                    let v = if x.negative { -i128::from(x.num) } else { i128::from(x.num) };
                    return <$t>::try_from(v).map_err(|_| TryFromRationalError(()));
                }
                // A long integer: its numerator is there, a number.
                if r.is_integer() {
                    <$t>::try_from(r.numer()).map_err(|_| TryFromRationalError(()))
                } else {
                    Err(TryFromRationalError(()))
                }
            }
        }

        impl TryFrom<Rational> for $t {
            type Error = TryFromRationalError;

            #[doc = concat!("Converts an integer to a `", stringify!($t), "`, as from a borrowed rational.")]
            #[inline]
            fn try_from(r: Rational) -> Result<$t, TryFromRationalError> {
                <$t>::try_from(&r)
            }
        }
    )*};
}

to_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
