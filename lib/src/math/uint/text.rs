//! The text of the fixed-width integers: `Display` in decimal; `LowerHex`, `UpperHex`, `Octal` and
//! `Binary`, all through `Formatter::pad_integral`, so that the flags `+`, `#`, `0`, the width,
//! the fill and the alignment do what they do for the primitive integers; `to_str_radix` in any
//! radix from 2 to 36; and `from_str_radix` and `FromStr`, with the primitives' rules and errors.
//!
//! An [`Int`] is written in decimal and by `to_str_radix` as its sign and the digits of its
//! absolute value, and in hexadecimal, octal and binary as the bits of its two's complement, as
//! `i64` is. Digits in a radix that is a power of two are read off the bits; in another radix the
//! value is divided by the largest power of the radix that fits in a limb, one limb at a time,
//! `O(N²)` in all, and read back the same way, by products with that power.

use super::limbs;
use super::{Int, Uint};
use core::fmt;
use core::num::IntErrorKind;
use core::str::FromStr;

/// An error from `from_str_radix` or `str::parse` of a [`Uint`] or an [`Int`]: what was wrong with
/// the text, as the standard library's [`IntErrorKind`] names it, from [`kind`](Self::kind).
///
/// # Examples
///
/// ```
/// use a::math::uint::{Int, Uint};
/// use core::num::IntErrorKind;
///
/// let e = Uint::<1>::from_str_radix("18446744073709551616", 10).unwrap_err();
/// assert_eq!(*e.kind(), IntErrorKind::PosOverflow);
/// assert_eq!(e.to_string(), "number too large to fit in target type");
/// assert_eq!(*"-".parse::<Int<2>>().unwrap_err().kind(), IntErrorKind::InvalidDigit);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseIntError {
    kind: IntErrorKind,
}

impl ParseIntError {
    /// What was wrong with the text: [`IntErrorKind::Empty`] for no text,
    /// [`IntErrorKind::InvalidDigit`] for a character that is not a digit of the radix (or a sign
    /// alone, or `-` before an unsigned number), and [`IntErrorKind::PosOverflow`] or
    /// [`IntErrorKind::NegOverflow`] for a value out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    /// use core::num::IntErrorKind;
    ///
    /// let e = Int::<1>::from_str_radix("-9223372036854775809", 10).unwrap_err();
    /// assert_eq!(*e.kind(), IntErrorKind::NegOverflow);
    /// ```
    #[inline]
    pub const fn kind(&self) -> &IntErrorKind {
        &self.kind
    }
}

impl fmt::Display for ParseIntError {
    /// Writes what was wrong, in the words of the standard library's `ParseIntError`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            IntErrorKind::Empty => "cannot parse integer from empty string",
            IntErrorKind::InvalidDigit => "invalid digit found in string",
            IntErrorKind::PosOverflow => "number too large to fit in target type",
            IntErrorKind::NegOverflow => "number too small to fit in target type",
            _ => "number would be zero for non-zero type",
        }
        .fmt(f)
    }
}

impl std::error::Error for ParseIntError {}

/// The error of the kind `kind`.
const fn error(kind: IntErrorKind) -> ParseIntError {
    ParseIntError { kind }
}

/// Panics unless `radix` is from 2 to 36.
#[track_caller]
const fn check_radix(radix: u32) {
    assert!(
        2 <= radix && radix <= 36,
        "from_str_radix: the radix must be from 2 to 36"
    );
}

/// The number of digits in `radix` that a limb holds whatever they are, `k`, and `radix^k`: the
/// largest power of the radix that fits in a limb.
const fn chunk(radix: u32) -> (usize, u64) {
    let r = radix as u64;
    let mut power = r;
    let mut k = 1;
    while let Some(next) = power.checked_mul(r) {
        power = next;
        k += 1;
    }
    (k, power)
}

/// Splits the sign off `src`, the primitives' way: `+` always, `-` only when `signed`; a sign
/// alone is an invalid digit, and empty text is empty.
const fn split_sign(src: &str, signed: bool) -> Result<(bool, &[u8]), ParseIntError> {
    let s = src.as_bytes();
    if s.is_empty() {
        return Err(error(IntErrorKind::Empty));
    }
    match s[0] {
        b'+' | b'-' if s.len() == 1 => Err(error(IntErrorKind::InvalidDigit)),
        b'+' => Ok((false, s.split_at(1).1)),
        b'-' if signed => Ok((true, s.split_at(1).1)),
        _ => Ok((false, s)),
    }
}

/// Reads the digits `digits` in `radix` as the magnitude of an integer of `N` limbs: the value of
/// the digits before the first character that is not a digit of the radix, whether that value
/// overflowed the width, and whether there is such a character.
///
/// The primitives read from the left and stop at the first error, an overflow or a character that
/// is not a digit, whichever comes first; as the value of a prefix only grows with its length, the
/// prefix before the bad character overflows exactly when they would report the overflow. The
/// digits are taken a limb's worth at a time.
const fn read_magnitude<const N: usize>(digits: &[u8], radix: u32) -> (Uint<N>, bool, bool) {
    let (k, _) = chunk(radix);
    let mut acc = Uint::<N>::ZERO;
    let mut i = 0;
    while i < digits.len() {
        let (mut value, mut power, mut count) = (0u64, 1u64, 0);
        let mut invalid = false;
        while count < k && i < digits.len() {
            match (digits[i] as char).to_digit(radix) {
                Some(d) => {
                    value = value * radix as u64 + d as u64;
                    power *= radix as u64;
                    count += 1;
                    i += 1;
                }
                None => {
                    invalid = true;
                    break;
                }
            }
        }
        if count > 0 {
            let (l, carry) = limbs::mul_add_1(acc.as_limbs(), power, value);
            if carry != 0 {
                return (acc, true, false);
            }
            acc = Uint::from_limbs(l);
        }
        if invalid {
            return (acc, false, true);
        }
    }
    (acc, false, false)
}

impl<const N: usize> Uint<N> {
    /// Reads an integer from text in `radix`, as `u64::from_str_radix` does: an optional `+`, then
    /// at least one digit, from `0` to the radix's last, letters in either case, and nothing else;
    /// zeros in front count for nothing.
    ///
    /// # Errors
    ///
    /// [`ParseIntError`] of the kind [`IntErrorKind::Empty`] for empty text,
    /// [`IntErrorKind::InvalidDigit`] for a character that is not a digit (`-` included) or a sign
    /// alone, and [`IntErrorKind::PosOverflow`] for a value beyond [`MAX`](Self::MAX); of an
    /// overflow and a bad character, the one met first from the left.
    ///
    /// # Panics
    ///
    /// If `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    /// use core::num::IntErrorKind;
    ///
    /// const P: Uint<4> = match Uint::from_str_radix(
    ///     "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
    ///     16,
    /// ) {
    ///     Ok(p) => p,
    ///     Err(_) => panic!(),
    /// };
    /// assert_eq!(P.as_limbs()[3], 0xffff_ffff_0000_0001);
    /// assert_eq!(Uint::<2>::from_str_radix("+Zz", 36), Ok(Uint::from_u64(36 * 35 + 35)));
    /// let e = Uint::<2>::from_str_radix("12a", 10).unwrap_err();
    /// assert_eq!(*e.kind(), IntErrorKind::InvalidDigit);
    /// ```
    #[track_caller]
    pub const fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
        check_radix(radix);
        let digits = match split_sign(src, false) {
            Ok((_, d)) => d,
            Err(e) => return Err(e),
        };
        match read_magnitude::<N>(digits, radix) {
            (_, true, _) => Err(error(IntErrorKind::PosOverflow)),
            (_, _, true) => Err(error(IntErrorKind::InvalidDigit)),
            (m, false, false) => Ok(m),
        }
    }

    /// The value as text in `radix`: the digits, the most significant first, without zeros in
    /// front, letters lower-case; `0` for zero.
    ///
    /// # Panics
    ///
    /// If `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(255).to_str_radix(16), "ff");
    /// assert_eq!(Uint::<2>::MAX.to_str_radix(36), "f5lxx1zz5pnorynqglhzmsp33");
    /// ```
    #[track_caller]
    pub fn to_str_radix(self, radix: u32) -> String {
        check_radix(radix);
        // The digits are ASCII, so the conversion cannot fail.
        String::from_utf8(self.digits(radix)).unwrap_or_default()
    }

    /// The digits in `radix`, ASCII, lower-case, the most significant first; `0` for zero.
    fn digits(self, radix: u32) -> Vec<u8> {
        const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        if self.is_zero() {
            return vec![b'0'];
        }
        if radix.is_power_of_two() {
            // Read off the bits, `k` to a digit, from the top.
            let k = radix.trailing_zeros();
            let count = self.bits().div_ceil(k);
            let mask = (1u64 << k) - 1;
            let l = self.as_limbs();
            return (0..count)
                .rev()
                .map(|j| {
                    let at = j * k;
                    let (i, s) = ((at / 64) as usize, at % 64);
                    let mut d = l[i] >> s;
                    if s + k > 64 && i + 1 < N {
                        d |= l[i + 1] << (64 - s);
                    }
                    DIGITS[(d & mask) as usize]
                })
                .collect();
        }
        // The chunks of k digits, from the least significant up, by divisions by radix^k.
        let (k, power) = chunk(radix);
        let mut chunks = Vec::new();
        let mut x = self;
        while !x.is_zero() {
            let (q, r) = x.div_rem_u64(power);
            chunks.push(r);
            x = q;
        }
        let mut out = Vec::with_capacity(chunks.len() * k);
        let mut buf = [0u8; 64];
        for (n, &c) in chunks.iter().rev().enumerate() {
            let mut c = c;
            let mut len = 0;
            while c > 0 {
                buf[len] = DIGITS[(c % radix as u64) as usize];
                c /= radix as u64;
                len += 1;
            }
            // Every chunk but the first is padded with zeros to k digits.
            if n > 0 {
                out.resize(out.len() + (k - len), b'0');
            }
            out.extend(buf[..len].iter().rev());
        }
        out
    }
}

impl<const N: usize> Int<N> {
    /// Reads an integer from text in `radix`, as `i64::from_str_radix` does: an optional `+` or
    /// `-`, then at least one digit, from `0` to the radix's last, letters in either case, and
    /// nothing else; zeros in front count for nothing, and `-0` is zero.
    ///
    /// # Errors
    ///
    /// [`ParseIntError`] of the kind [`IntErrorKind::Empty`] for empty text,
    /// [`IntErrorKind::InvalidDigit`] for a character that is not a digit or a sign alone, and
    /// [`IntErrorKind::PosOverflow`] or [`IntErrorKind::NegOverflow`] for a value beyond
    /// [`MAX`](Self::MAX) or below [`MIN`](Self::MIN); of an overflow and a bad character, the
    /// one met first from the left.
    ///
    /// # Panics
    ///
    /// If `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    /// use core::num::IntErrorKind;
    ///
    /// assert_eq!(Int::<2>::from_str_radix("-ff", 16), Ok(Int::from_i64(-255)));
    /// assert_eq!(
    ///     Int::<2>::from_str_radix("-170141183460469231731687303715884105728", 10),
    ///     Ok(Int::MIN)
    /// );
    /// let e = Int::<2>::from_str_radix("170141183460469231731687303715884105728", 10);
    /// assert_eq!(*e.unwrap_err().kind(), IntErrorKind::PosOverflow);
    /// ```
    #[track_caller]
    pub const fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
        check_radix(radix);
        let (negative, digits) = match split_sign(src, true) {
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        let (m, overflow, invalid) = read_magnitude::<N>(digits, radix);
        // At most 2^(64N − 1) for a negative number, one less for a positive one; the wrapping
        // keeps a width of zero limbs, whose only value is 0, from underflowing.
        let top = m.bit(Self::BITS.wrapping_sub(1));
        let fits = !top || (negative && m.trailing_zeros() == Self::BITS.wrapping_sub(1));
        if overflow || !fits {
            Err(error(if negative {
                IntErrorKind::NegOverflow
            } else {
                IntErrorKind::PosOverflow
            }))
        } else if invalid {
            Err(error(IntErrorKind::InvalidDigit))
        } else {
            Ok(Int::from_sign_magnitude(negative, m))
        }
    }

    /// The value as text in `radix`: `-` for a negative number, then the digits of its absolute
    /// value, the most significant first, without zeros in front, letters lower-case; `0` for
    /// zero.
    ///
    /// # Panics
    ///
    /// If `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-255).to_str_radix(16), "-ff");
    /// assert_eq!(Int::<1>::MIN.to_str_radix(2), format!("-1{}", "0".repeat(63)));
    /// ```
    #[track_caller]
    pub fn to_str_radix(self, radix: u32) -> String {
        let digits = self.unsigned_abs().to_str_radix(radix);
        if self.is_negative() {
            format!("-{digits}")
        } else {
            digits
        }
    }
}

/// Writes `-` (for `negative`) or `+` (with the `+` flag), `0x` and the hexadecimal digits of `m`
/// with the formatter's width, fill and alignment, or with zeros after the prefix for the `0`
/// flag: what `pad_integral` writes for `{:#x}`, which `Debug` cannot ask of it, as the `#` flag
/// cannot be set on a formatter.
pub(super) fn debug_hex<const N: usize>(
    f: &mut fmt::Formatter<'_>,
    negative: bool,
    m: Uint<N>,
) -> fmt::Result {
    use core::fmt::{Alignment, Write};
    let digits = m.digits(16);
    // The digits are ASCII.
    let digits = core::str::from_utf8(&digits).unwrap_or_default();
    let sign = if negative {
        "-"
    } else if f.sign_plus() {
        "+"
    } else {
        ""
    };
    let len = sign.len() + 2 + digits.len();
    let padding = match f.width() {
        Some(w) if w > len => w - len,
        _ => 0,
    };
    if f.sign_aware_zero_pad() {
        // The sign and the prefix before the zeros, whatever the fill and alignment.
        f.write_str(sign)?;
        f.write_str("0x")?;
        for _ in 0..padding {
            f.write_char('0')?;
        }
        return f.write_str(digits);
    }
    // Integers align right unless asked otherwise; the centre gives the odd fill to the right.
    let (before, after) = match f.align() {
        Some(Alignment::Left) => (0, padding),
        Some(Alignment::Center) => (padding / 2, padding.div_ceil(2)),
        _ => (padding, 0),
    };
    let fill = f.fill();
    for _ in 0..before {
        f.write_char(fill)?;
    }
    f.write_str(sign)?;
    f.write_str("0x")?;
    f.write_str(digits)?;
    for _ in 0..after {
        f.write_char(fill)?;
    }
    Ok(())
}

/// Writes `digits` through `pad_integral`, upper-cased when asked.
fn pad(
    f: &mut fmt::Formatter<'_>,
    non_negative: bool,
    prefix: &str,
    mut digits: Vec<u8>,
    upper: bool,
) -> fmt::Result {
    if upper {
        digits.make_ascii_uppercase();
    }
    // The digits are ASCII.
    let text = core::str::from_utf8(&digits).unwrap_or_default();
    f.pad_integral(non_negative, prefix, text)
}

impl<const N: usize> fmt::Display for Uint<N> {
    /// Writes the value in decimal; the formatter's flags, width and fill apply as they do to the
    /// primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.to_string(), u128::MAX.to_string());
    /// assert_eq!(format!("{:>6}", Uint::<2>::from_u64(42)), "    42");
    /// assert_eq!(format!("{:+06}", Uint::<2>::from_u64(42)), "+00042");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(f, true, "", self.digits(10), false)
    }
}

impl<const N: usize> fmt::LowerHex for Uint<N> {
    /// Writes the value in hexadecimal, lower-case; `0x` in front with the `#` flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<4>::from_u64(0xbeef);
    /// assert_eq!(format!("{x:x}"), "beef");
    /// assert_eq!(format!("{x:#066x}"), format!("0x{}beef", "0".repeat(60)));
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(f, true, "0x", self.digits(16), false)
    }
}

impl<const N: usize> fmt::UpperHex for Uint<N> {
    /// Writes the value in hexadecimal, upper-case; `0x` in front with the `#` flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(format!("{:#X}", Uint::<2>::from_u64(0xbeef)), "0xBEEF");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(f, true, "0x", self.digits(16), true)
    }
}

impl<const N: usize> fmt::Octal for Uint<N> {
    /// Writes the value in octal; `0o` in front with the `#` flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(format!("{:#o}", Uint::<2>::from_u64(8)), "0o10");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(f, true, "0o", self.digits(8), false)
    }
}

impl<const N: usize> fmt::Binary for Uint<N> {
    /// Writes the value in binary; `0b` in front with the `#` flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(format!("{:#b}", Uint::<2>::from_u64(5)), "0b101");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(f, true, "0b", self.digits(2), false)
    }
}

impl<const N: usize> fmt::Display for Int<N> {
    /// Writes the value in decimal: `-` for a negative number, then the digits of its absolute
    /// value; the formatter's flags, width and fill apply as they do to the primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.to_string(), i128::MIN.to_string());
    /// assert_eq!(format!("{:+}", Int::<2>::ONE), "+1");
    /// assert_eq!(format!("{:06}", Int::<2>::from_i64(-42)), "-00042");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pad(
            f,
            !self.is_negative(),
            "",
            self.unsigned_abs().digits(10),
            false,
        )
    }
}

/// The formatting traits that write the bits of an `Int`'s two's complement, as the primitives do.
macro_rules! twos_complement {
    ($($trait:ident $radix:literal $prefix:literal $upper:literal),*) => {$(
        impl<const N: usize> fmt::$trait for Int<N> {
            #[doc = concat!("Writes the bits of the two's complement in radix ", stringify!($radix), ", as `i64` does: `-1` is all ones; `", $prefix, "` in front with the `#` flag.")]
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::uint::Int;
            ///
            /// assert_eq!(format!("{:x}", Int::<2>::MINUS_ONE), "f".repeat(32));
            /// assert_eq!(format!("{:#b}", Int::<1>::MIN), format!("{:#b}", i64::MIN));
            /// ```
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                pad(f, true, $prefix, self.cast_unsigned().digits($radix), $upper)
            }
        }
    )*};
}

twos_complement!(LowerHex 16 "0x" false, UpperHex 16 "0x" true, Octal 8 "0o" false, Binary 2 "0b" false);

impl<const N: usize> FromStr for Uint<N> {
    type Err = ParseIntError;

    /// Reads an integer in decimal, as [`Uint::from_str_radix`] does.
    #[inline]
    fn from_str(src: &str) -> Result<Self, ParseIntError> {
        Self::from_str_radix(src, 10)
    }
}

impl<const N: usize> FromStr for Int<N> {
    type Err = ParseIntError;

    /// Reads an integer in decimal, as [`Int::from_str_radix`] does.
    #[inline]
    fn from_str(src: &str) -> Result<Self, ParseIntError> {
        Self::from_str_radix(src, 10)
    }
}
