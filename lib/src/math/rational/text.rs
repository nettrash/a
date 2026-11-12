//! Rationals as text: [`Display`](fmt::Display) writes `-3/4`, or `5` for an integer, and
//! [`Debug`](fmt::Debug) writes the same; [`FromStr`] reads integers, fractions and decimals with
//! an exponent, exactly; [`Rational::from_str_radix`] and [`Rational::to_str_radix`] read and write
//! integers and fractions in the radices from 2 to 36; and the conversions from integers.
//!
//! Parts of up to 64 bits are read and written in registers, in every radix, with no allocation
//! but the text written: decimal digits are read eight at a time, checked and combined by three
//! products (Lemire's method), binary ones eight at a time packed by one product, those of the
//! other powers of two by shifts, and those of other radices by a product and a sum each, the last
//! digit of the largest words checked for overflow; a fraction is then reduced by the gcd of two
//! words. Words are written in decimal two digits at a time from a table, in binary eight at a time
//! spread by one product, in the other powers of two by shifts, and in other radices in blocks
//! below `2^32`, each digit by a product with the radix's reciprocal.
//!
//! Longer parts are read and written as [`Number`]s are, by divide and conquer from a few hundred
//! digits on, in time `O(M(n) log n)` for `n` digits; a fraction read is then reduced by the gcd of
//! its parts. A decimal `d₁…dₙ.f₁…fₘ e x` is the integer `d₁…dₙf₁…fₘ` (its trailing zeros dropped,
//! each adding one to the exponent) times `10^(x − m)`. A negative power of ten leaves the
//! fraction `N/10^k` in lowest terms but for the factors 2 or 5 of `N`, which its last digit tells
//! apart: one of `1, 3, 7, 9` shares neither with `10^k`, an even one only twos, which a shift
//! removes, and `5` only fives, which a remainder by `5^27` counts, a gcd with `5^k` past that.
//! Digits that make a word, with an exponent of up to 19 in magnitude, are read in registers and
//! reduced so, into words. `Display` writes fractions of longer parts of up to 320 characters on
//! the stack, so that the only allocation is the text the formatter keeps.

use super::small::{Small, gcd};
use super::{Rational, is_one, one};
use crate::math::numbers::{ByteOrder, Magnitude, Number};
use core::fmt;
use core::fmt::Write;
use core::str::FromStr;

impl Rational {
    /// The largest exponent, in magnitude, that decimal text may have when it is parsed:
    /// `1e1000000` is read, `1e1000001` is an error of kind
    /// [`RationalErrorKind::ExponentOutOfRange`].
    ///
    /// The bound keeps the work of parsing in proportion to the length of the text, as untrusted
    /// text needs: `1e1000000` is a numerator of 415 KB, read in about a millisecond, where
    /// `1e1000000000`, twelve characters, would ask for 415 MB. A larger power of ten is built on
    /// purpose with [`Number::pow`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::{Rational, RationalErrorKind};
    ///
    /// assert_eq!(Rational::MAX_PARSE_EXPONENT, 1_000_000);
    /// assert!("1e-1000000".parse::<Rational>().is_ok());
    /// let e = "1e1000001".parse::<Rational>().unwrap_err();
    /// assert_eq!(*e.kind(), RationalErrorKind::ExponentOutOfRange);
    ///
    /// // A larger power, on purpose.
    /// let big = Rational::from(Number::from(10).pow(1_000_001));
    /// assert!(big.is_integer());
    /// ```
    pub const MAX_PARSE_EXPONENT: u64 = 1_000_000;
}

impl fmt::Display for Rational {
    /// Writes the number in decimal as `numerator/denominator`, `-` in front of a negative one, and
    /// as the numerator alone for an integer; the formatter's flags, width and fill apply to the
    /// whole, as they do to the primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(-3) / 4u32;
    /// assert_eq!(r.to_string(), "-3/4");
    /// assert_eq!(format!("{:>7}", r), "   -3/4");
    /// assert_eq!(format!("{:+}", Rational::from(5)), "+5");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(x) = Small::of(self) {
            let mut buf = [0u8; WORD_TEXT];
            let i = small_text(x.negative, x.num, x.den, &mut buf);
            if f.width().is_none() && !f.sign_plus() {
                // The sign and the digits in one piece: one write, and for `to_string` one
                // allocation of the text, where `pad_integral` writes the sign apart.
                return f.write_str(ascii(&buf[i..]));
            }
            let digits = i + usize::from(x.negative);
            return f.pad_integral(!x.negative, "", ascii(&buf[digits..]));
        }
        let (num, den) = (self.numer(), self.denom());
        if is_one(den) {
            return fmt::Display::fmt(num, f);
        }
        let negative = num.is_negative();
        // At most `log10(256) < 2.41` digits a byte, and one more, for each part, and the sign.
        let bound = (num.value.len() + den.value.len()) * 5 / 2 + 4;
        if bound <= STACK_TEXT {
            // Written on the stack, and from there once to the formatter.
            let mut text = StackText {
                bytes: [0; STACK_TEXT],
                len: 0,
            };
            write!(text, "{num}/{den}")?;
            return f.pad_integral(!negative, "", text.as_str(usize::from(negative)));
        }
        let mut s = String::with_capacity(bound);
        write!(s, "{num}/{den}")?;
        f.pad_integral(!negative, "", &s[usize::from(negative)..])
    }
}

/// The pairs of decimal digits from `00` to `99`.
const PAIRS: [[u8; 2]; 100] = {
    let mut t = [[0; 2]; 100];
    let mut i = 0;
    while i < 100 {
        t[i] = [b'0' + (i / 10) as u8, b'0' + (i % 10) as u8];
        i += 1;
    }
    t
};

/// Writes the decimal digits of `x` in `buf` to end before `end`, two at a time; returns where
/// they start.
#[inline(always)]
fn digits_u64(mut x: u64, buf: &mut [u8; WORD_TEXT], end: usize) -> usize {
    let mut i = end;
    // Blocks of eight digits from the bottom, each written as four pairs at once.
    while x >= 100_000_000 {
        let block = x % 100_000_000;
        x /= 100_000_000;
        let (high, low) = ((block / 10_000) as usize, (block % 10_000) as usize);
        i -= 8;
        buf[i..i + 2].copy_from_slice(&PAIRS[high / 100]);
        buf[i + 2..i + 4].copy_from_slice(&PAIRS[high % 100]);
        buf[i + 4..i + 6].copy_from_slice(&PAIRS[low / 100]);
        buf[i + 6..i + 8].copy_from_slice(&PAIRS[low % 100]);
    }
    while x >= 100 {
        let pair = (x % 100) as usize;
        x /= 100;
        i -= 2;
        buf[i..i + 2].copy_from_slice(&PAIRS[pair]);
    }
    if x >= 10 {
        i -= 2;
        buf[i..i + 2].copy_from_slice(&PAIRS[x as usize]);
    } else {
        i -= 1;
        buf[i] = b'0' + x as u8;
    }
    i
}

/// The most characters of a rational in words in decimal: a sign, 20 digits, `/` and 20 more.
const WORD_TEXT: usize = 42;

/// Writes `-num/den` in decimal at the end of `buf`, from the last character back, the `-` only
/// when `negative` is set and `/den` only when `den` is not 1; returns where the text starts.
#[inline]
fn small_text(negative: bool, num: u64, den: u64, buf: &mut [u8; WORD_TEXT]) -> usize {
    let mut i = WORD_TEXT;
    if den != 1 {
        i = digits_u64(den, buf, i) - 1;
        buf[i] = b'/';
    }
    i = digits_u64(num, buf, i) - 1;
    buf[i] = b'-';
    i + usize::from(!negative)
}

/// ASCII bytes as text.
#[inline(always)]
fn ascii(bytes: &[u8]) -> &str {
    core::str::from_utf8(bytes).unwrap_or_default()
}

/// Text of fractions of up to this many characters is written on the stack by `Display`: parts
/// of up to about 64 bytes each.
const STACK_TEXT: usize = 320;

/// A buffer on the stack that text is written to.
struct StackText {
    bytes: [u8; STACK_TEXT],
    len: usize,
}

impl StackText {
    /// The text written, from byte `from` on.
    fn as_str(&self, from: usize) -> &str {
        // The bytes are ASCII digits, `-` and `/`, written as whole strings.
        ascii(&self.bytes[from..self.len])
    }
}

impl fmt::Write for StackText {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        self.bytes
            .get_mut(self.len..end)
            .ok_or(fmt::Error)?
            .copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// The digits of the radices from 2 to 36, letters lower-case.
const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// For each radix from 3 on, the largest power `b = radix^k` below `2^32` and `k`: a word is cut into
/// blocks of `k` digits by divisions by `b`, and the digits of a block found by multiplications.
const BLOCKS: [(u64, usize); 37] = {
    let mut t = [(0u64, 0usize); 37];
    let mut r = 3;
    while r <= 36 {
        let (mut b, mut k) = (1u64, 0);
        while b * r < 1 << 32 {
            b *= r;
            k += 1;
        }
        t[r as usize] = (b, k);
        r += 1;
    }
    t
};

/// `floor(2^64 / radix) + 1` for each radix from 3 on that is not a power of two: for `x < 2^32`,
/// `floor(x / radix)` is the high word of `x` times it, as the error of the product, below
/// `x / 2^64 < 2^-32`, is less than `1 / radix`, the distance from `x / radix` to the next integer
/// above it.
const INVERSES: [u64; 37] = {
    let mut t = [0u64; 37];
    let mut r = 3;
    while r <= 36 {
        t[r as usize] = u64::MAX / r + 1;
        r += 1;
    }
    t
};

/// Writes the digits of `x` in `radix`, not 10, at the end of `buf`, without zeros in front, `0`
/// for zero; returns where they start.
#[inline]
fn digits_radix(x: u64, radix: u32, buf: &mut [u8; 64]) -> usize {
    let mut i = 64;
    if radix == 2 {
        if x == 0 {
            buf[63] = b'0';
            return 63;
        }
        // Each byte of `x` spread into eight digits by one product, the highest bit first.
        let bits = 64 - x.leading_zeros() as usize;
        for g in 0..bits.div_ceil(8) {
            let b = (x >> (8 * g)) & 0xFF;
            let t = (b.wrapping_mul(0x8040_2010_0804_0201) >> 7) & 0x0101_0101_0101_0101;
            buf[56 - 8 * g..64 - 8 * g].copy_from_slice(&(t | 0x3030_3030_3030_3030).to_le_bytes());
        }
        return 64 - bits;
    }
    if radix.is_power_of_two() {
        let (bits, mask) = (radix.trailing_zeros(), u64::from(radix) - 1);
        let mut x = x;
        loop {
            i -= 1;
            buf[i] = DIGITS[(x & mask) as usize];
            x >>= bits;
            if x == 0 {
                return i;
            }
        }
    }
    let (b, k) = BLOCKS[radix as usize];
    let v = INVERSES[radix as usize];
    let r = u64::from(radix);
    let mut x = x;
    loop {
        // A block below `2^32` and what is above it.
        let (high, mut block) = if x >= b { (x / b, x % b) } else { (0, x) };
        let mut n = 0;
        loop {
            let q = ((u128::from(block) * u128::from(v)) >> 64) as u64;
            i -= 1;
            buf[i] = DIGITS[(block - q * r) as usize];
            block = q;
            n += 1;
            // The whole block, zeros in front, below the top one.
            if block == 0 && (high == 0 || n == k) {
                break;
            }
        }
        if high == 0 {
            return i;
        }
        x = high;
    }
}

/// The text of a rational in words in `radix`: the sign, the digits of the numerator and, but for
/// an integer, `/` and those of the denominator, written on the stack and copied once into the
/// string.
fn word_text(x: Small, radix: u32) -> String {
    if radix == 10 {
        let mut buf = [0u8; WORD_TEXT];
        let i = small_text(x.negative, x.num, x.den, &mut buf);
        return String::from(ascii(&buf[i..]));
    }
    let mut out = [0u8; 130];
    let mut len = usize::from(x.negative);
    out[0] = b'-';
    {
        let mut digits = [0u8; 64];
        let i = digits_radix(x.num, radix, &mut digits);
        out[len..len + 64 - i].copy_from_slice(&digits[i..]);
        len += 64 - i;
        if x.den != 1 {
            out[len] = b'/';
            len += 1;
            let j = digits_radix(x.den, radix, &mut digits);
            out[len..len + 64 - j].copy_from_slice(&digits[j..]);
            len += 64 - j;
        }
    }
    String::from(ascii(&out[..len]))
}

impl fmt::Debug for Rational {
    /// Writes the number as [`Display`](fmt::Display) does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl From<Number> for Rational {
    /// The integer `n` as `n/1`, as [`Rational::from_integer`] makes it.
    #[inline]
    fn from(n: Number) -> Rational {
        Rational::from_integer(n)
    }
}

impl From<&Number> for Rational {
    /// The integer `n` as `n/1`: in words when it fits in 64 bits, and otherwise in new vectors.
    #[inline]
    fn from(n: &Number) -> Rational {
        let m = Magnitude::of(n);
        if m.bytes.len() <= 8 {
            return Rational::from_word(m.negative, m.small() as u64, 1, n.byte_order());
        }
        Rational::from_integer(n.clone())
    }
}

macro_rules! from_unsigned {
    ($($t:ty),*) => {$(
        impl From<$t> for Rational {
            #[doc = concat!("The `", stringify!($t), "` `v` as `v/1`, little-endian.")]
            #[inline]
            fn from(v: $t) -> Rational {
                // In words when it fits in 64 bits.
                Rational::from_u128(false, v as u128, 1, ByteOrder::LittleEndian)
            }
        }
    )*};
}

macro_rules! from_signed {
    ($($t:ty),*) => {$(
        impl From<$t> for Rational {
            #[doc = concat!("The `", stringify!($t), "` `v` as `v/1`, little-endian.")]
            #[inline]
            fn from(v: $t) -> Rational {
                // In words when it fits in 64 bits.
                Rational::from_u128(v < 0, (v as i128).unsigned_abs(), 1, ByteOrder::LittleEndian)
            }
        }
    )*};
}

from_unsigned!(u8, u16, u32, u64, u128, usize);
from_signed!(i8, i16, i32, i64, i128, isize);

// Writing

impl Rational {
    /// Returns the number as text in `radix`: `-` for a negative number, then the digits of the
    /// numerator's magnitude, `/` and the digits of the denominator, letters lower-case, as
    /// [`Number::to_str_radix`] writes each part; the numerator alone for an integer.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(-255) / 16u32;
    /// assert_eq!(r.to_str_radix(16), "-ff/10");
    /// assert_eq!(r.to_str_radix(2), "-11111111/10000");
    /// assert_eq!(Rational::from(35).to_str_radix(36), "z");
    /// ```
    #[track_caller]
    pub fn to_str_radix(&self, radix: u32) -> String {
        assert!(
            (2..=36).contains(&radix),
            "from_str_radix_int: must lie in the range `[2, 36]` - found {radix}"
        );
        if let Some(x) = Small::of(self) {
            return word_text(x, radix);
        }
        let (num, den) = (self.numer(), self.denom());
        if is_one(den) {
            return num.to_str_radix(radix);
        }
        if radix == 10 {
            // In one vector, from the buffers of the formatting of the parts.
            let bound = (num.value.len() + den.value.len()) * 5 / 2 + 4;
            let mut s = String::with_capacity(bound);
            let _ = write!(s, "{num}/{den}");
            return s;
        }
        let den = den.to_str_radix(radix);
        let mut s = num.to_str_radix(radix);
        s.reserve_exact(1 + den.len());
        s.push('/');
        s.push_str(&den);
        s
    }

    /// Reads a rational from text in `radix`: an optional sign, `+` or `-`, then the digits of an
    /// integer, or of a numerator, `/` and the digits of a denominator, which is not zero and has
    /// no sign, letters in either case, with no other characters; the fraction is reduced to
    /// lowest terms, and built little-endian. There are no decimal points or exponents, in any
    /// radix: `e` is a digit from radix 15 on; [`FromStr`] reads decimals.
    ///
    /// # Errors
    ///
    /// [`ParseRationalError`] of kind [`RationalErrorKind::Empty`] for empty text,
    /// [`RationalErrorKind::InvalidDigit`] for a character that is not a digit of the radix where
    /// one should be, a part with no digits, or a second sign, and
    /// [`RationalErrorKind::ZeroDenominator`] for a zero denominator.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::{Rational, RationalErrorKind};
    ///
    /// let r = Rational::from_str_radix("-FF/1e", 16).unwrap();
    /// assert_eq!(r.to_string(), "-17/2");
    /// assert_eq!(Rational::from_str_radix("z", 36).unwrap(), Rational::from(35));
    /// let error = Rational::from_str_radix("1/0", 2).unwrap_err();
    /// assert_eq!(*error.kind(), RationalErrorKind::ZeroDenominator);
    /// ```
    #[track_caller]
    pub fn from_str_radix(src: &str, radix: u32) -> Result<Rational, ParseRationalError> {
        assert!(
            (2..=36).contains(&radix),
            "from_str_radix_int: must lie in the range `[2, 36]` - found {radix}"
        );
        let (negative, body) = signed(src)?;
        match body.bytes().position(|c| c == b'/') {
            None => integer(body, radix, negative),
            Some(i) => fraction(&body[..i], &body[i + 1..], radix, negative),
        }
    }
}

// Reading

impl FromStr for Rational {
    type Err = ParseRationalError;

    /// Reads a rational from decimal text, exactly, and builds it little-endian. The text is an
    /// optional sign, `+` or `-`, then one of
    ///
    /// - an integer, `digits`: `-12`;
    /// - a fraction, `digits/digits`, the denominator not zero and without a sign, reduced to
    ///   lowest terms: `6/-4` is an error, `-6/4` is `-3/2`;
    /// - a decimal, `digits.digits`, `digits.` or `.digits`, or an integer, with an optional
    ///   exponent, `e` or `E`, an optional sign and digits: `0.1` is `1/10`, `-1.5e-3` is
    ///   `-3/2000`, `2e10` is `20000000000`.
    ///
    /// There is no whitespace, no `_` and no other character, as in the text of a
    /// [`Number`]; `-0` and `-0.0` are zero. The exponent may be at most
    /// [`Rational::MAX_PARSE_EXPONENT`] (`1000000`) in magnitude, which keeps the work bounded by
    /// the length of the text: `1e1000000000` would be a numerator of 415 MB, built from 12
    /// characters.
    ///
    /// # Errors
    ///
    /// [`ParseRationalError`] of kind [`RationalErrorKind::Empty`] for empty text,
    /// [`RationalErrorKind::InvalidDigit`] for a character out of place, or a part with no digits
    /// where it needs some, [`RationalErrorKind::ZeroDenominator`] for a zero denominator, and
    /// [`RationalErrorKind::ExponentOutOfRange`] for an exponent beyond the bound, whatever the
    /// digits before it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::{Rational, RationalErrorKind};
    ///
    /// let r: Rational = "-1.5e-3".parse().unwrap();
    /// assert_eq!(r.to_string(), "-3/2000");
    /// assert_eq!("12/-4".parse::<Rational>().unwrap_err().kind(), &RationalErrorKind::InvalidDigit);
    /// assert_eq!("0.1".parse::<Rational>().unwrap(), Rational::from(1) / 10u32);
    /// assert_eq!(".5".parse::<Rational>().unwrap(), "1/2".parse::<Rational>().unwrap());
    /// assert_eq!("2e10".parse::<Rational>().unwrap(), Rational::from(20_000_000_000u64));
    /// let error = "1e1000001".parse::<Rational>().unwrap_err();
    /// assert_eq!(*error.kind(), RationalErrorKind::ExponentOutOfRange);
    /// ```
    fn from_str(src: &str) -> Result<Rational, ParseRationalError> {
        let (negative, body) = signed(src)?;
        let bytes = body.as_bytes();
        match bytes
            .iter()
            .position(|&c| matches!(c, b'/' | b'.' | b'e' | b'E'))
        {
            None => integer(body, 10, negative),
            Some(i) if bytes[i] == b'/' => fraction(&body[..i], &body[i + 1..], 10, negative),
            Some(i) => decimal(&body[..i], &body[i..], negative),
        }
    }
}

/// The sign of the text and the text after it.
#[inline]
fn signed(src: &str) -> Result<(bool, &str), ParseRationalError> {
    match src.as_bytes().first() {
        None => Err(error(RationalErrorKind::Empty)),
        Some(b'-') => Ok((true, &src[1..])),
        Some(b'+') => Ok((false, &src[1..])),
        Some(_) => Ok((false, src)),
    }
}

/// The value of each byte as a digit, letters in either case, and 36 for a byte that is a digit in
/// no radix.
const VALUES: [u8; 256] = {
    let mut t = [36u8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = i as u8;
        i += 1;
    }
    let mut i = 0;
    while i < 26 {
        t[b'a' as usize + i] = 10 + i as u8;
        t[b'A' as usize + i] = 10 + i as u8;
        i += 1;
    }
    t
};

/// For each radix, how many digits any word holds, `radix^n ≤ 2^64`, and how many the largest
/// word has: one more, unless the radix is a power of two whose bits divide 64.
const WORD_DIGITS: [(usize, usize); 37] = {
    let mut t = [(0usize, 0usize); 37];
    let mut r = 2;
    while r <= 36 {
        let (mut p, mut n) = (1u128, 0);
        while p * r <= 1 << 64 {
            p *= r;
            n += 1;
        }
        t[r as usize] = (n, if p == 1 << 64 { n } else { n + 1 });
        r += 1;
    }
    t
};

/// The value of the digit `c` in `radix`, `radix` or more when it is not one.
#[inline(always)]
fn digit(c: u8, radix: u32) -> u64 {
    if radix <= 10 {
        u64::from(c.wrapping_sub(b'0'))
    } else {
        u64::from(VALUES[c as usize])
    }
}

/// The value of the digits `run` in `radix` when it is a word: `None` for no digits, a character
/// that is not one, or more digits than the largest word has (zeros in front too); a value past
/// `2^64` with as many digits as the largest word is `None` as well.
#[inline(always)]
fn word(run: &[u8], radix: u32) -> Option<u64> {
    match radix {
        10 => decimal_word(run),
        2 => binary_word(run),
        4 => bits_word::<2>(run),
        8 => bits_word::<3>(run),
        16 => bits_word::<4>(run),
        32 => bits_word::<5>(run),
        _ => radix_word(run, radix),
    }
}

/// [`word`] in a radix that is not a power of two: a product and a sum a digit, the last digit of
/// the largest words checked.
#[inline(always)]
fn radix_word(run: &[u8], radix: u32) -> Option<u64> {
    let (safe, most) = WORD_DIGITS[radix as usize];
    if run.is_empty() || run.len() > most {
        return None;
    }
    let r = u64::from(radix);
    let (head, tail) = run.split_at(run.len().min(safe));
    let mut v = 0u64;
    for &c in head {
        let d = digit(c, radix);
        if d >= r {
            return None;
        }
        v = v * r + d;
    }
    for &c in tail {
        let d = digit(c, radix);
        if d >= r {
            return None;
        }
        v = v.checked_mul(r)?.checked_add(d)?;
    }
    Some(v)
}

/// [`word`] in the radix `2^B`, from 4 to 32: a shift a digit.
#[inline(always)]
fn bits_word<const B: u32>(run: &[u8]) -> Option<u64> {
    let most = 64usize.div_ceil(B as usize);
    if run.is_empty() || run.len() > most {
        return None;
    }
    let mut v = 0u64;
    for &c in run {
        let d = digit(c, 1 << B);
        if d >> B != 0 {
            return None;
        }
        v = v << B | d;
    }
    // With as many digits as the largest word, the first one must leave room for the others.
    if run.len() == most && digit(run[0], 1 << B) >> (64 - B * (most as u32 - 1)) != 0 {
        return None;
    }
    Some(v)
}

/// [`word`] in binary: eight digits at a time, checked and packed into a byte by one product.
#[inline(always)]
fn binary_word(run: &[u8]) -> Option<u64> {
    if run.is_empty() || run.len() > 64 {
        return None;
    }
    let mut v = 0u64;
    let (chunks, rest) = run.as_chunks::<8>();
    for &chunk in chunks {
        let x = u64::from_le_bytes(chunk);
        // Each byte `0` or `1`, which the product gathers into one, the first the highest bit.
        if x & 0xFEFE_FEFE_FEFE_FEFE != 0x3030_3030_3030_3030 {
            return None;
        }
        let bits = (x & 0x0101_0101_0101_0101).wrapping_mul(0x8040_2010_0804_0201) >> 56;
        v = v << 8 | bits;
    }
    for &c in rest {
        let d = c.wrapping_sub(b'0');
        if d > 1 {
            return None;
        }
        v = v << 1 | u64::from(d);
    }
    Some(v)
}

/// [`word`] in decimal: eight digits at a time, checked and combined in three products (Lemire's
/// method), up to 19 digits, and a twentieth checked.
#[inline(always)]
fn decimal_word(run: &[u8]) -> Option<u64> {
    if run.is_empty() || run.len() > 20 {
        return None;
    }
    let (head, tail) = run.split_at(run.len().min(19));
    let mut v = 0u64;
    let (chunks, rest) = head.as_chunks::<8>();
    for &chunk in chunks {
        let x = u64::from_le_bytes(chunk);
        // Each byte from `0x30` to `0x39`: its high half 3, and still 3 with 6 added.
        let high = 0xF0F0_F0F0_F0F0_F0F0;
        if x & high | (x.wrapping_add(0x0606_0606_0606_0606) & high) >> 4 != 0x3333_3333_3333_3333 {
            return None;
        }
        // The first digit is the low byte: pairs, then quadruples, then the eight.
        let x = x & 0x0F0F_0F0F_0F0F_0F0F;
        let x = (x.wrapping_mul(10) + (x >> 8)) & 0x00FF_00FF_00FF_00FF;
        let x = (x.wrapping_mul(100) + (x >> 16)) & 0x0000_FFFF_0000_FFFF;
        let x = (x.wrapping_mul(10_000) + (x >> 32)) & 0xFFFF_FFFF;
        v = v * 100_000_000 + x;
    }
    for &c in rest {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + u64::from(d);
    }
    for &c in tail {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(u64::from(d))?;
    }
    Some(v)
}

/// The value of the decimal digits `high` followed by `low` when it is a word, as [`word`] has it.
#[inline(always)]
fn word_pair(high: &[u8], low: &[u8]) -> Option<u64> {
    if low.is_empty() {
        return word(high, 10);
    }
    if high.len() + low.len() > 20 {
        return None;
    }
    let v = if high.is_empty() { 0 } else { word(high, 10)? };
    let mut v = Some(v);
    for &c in low {
        let d = digit(c, 10);
        if d > 9 {
            return None;
        }
        v = v?.checked_mul(10)?.checked_add(d);
    }
    v
}

/// The magnitude that the digits `run` write in `radix`, little-endian, in the shortest form.
fn magnitude(run: &str, radix: u32) -> Result<Number, ParseRationalError> {
    match run.as_bytes().first() {
        None | Some(b'+' | b'-') => Err(error(RationalErrorKind::InvalidDigit)),
        Some(_) => Number::from_str_radix(run, radix, ByteOrder::LittleEndian)
            .map_err(|_| error(RationalErrorKind::InvalidDigit)),
    }
}

/// `n` with the sign `negative`, which zero never has.
#[inline(always)]
fn with_sign(mut n: Number, negative: bool) -> Number {
    n.negative = negative && !n.is_zero();
    n
}

/// The integer of the sign `negative` and the magnitude `m`, little-endian; zero is not negative.
#[inline(always)]
fn small_integer(negative: bool, m: u128) -> Rational {
    Rational::from_u128(negative && m != 0, m, 1, ByteOrder::LittleEndian)
}

/// The integer of the sign `negative` and the digits `run` in `radix`: in registers when it is a
/// word.
#[inline(always)]
fn integer(run: &str, radix: u32, negative: bool) -> Result<Rational, ParseRationalError> {
    match word(run.as_bytes(), radix) {
        Some(v) => Ok(small_integer(negative, u128::from(v))),
        None => long_integer(run, radix, negative),
    }
}

/// [`integer`] of digits that are not a word, or not digits.
#[inline(never)]
fn long_integer(run: &str, radix: u32, negative: bool) -> Result<Rational, ParseRationalError> {
    let n = with_sign(magnitude(run, radix)?, negative);
    Ok(Rational::from_reduced(n, one(ByteOrder::LittleEndian)))
}

/// The fraction `±p/q` of the digits `p` and `q` in `radix`, in lowest terms: in registers when
/// both are words.
#[inline(always)]
fn fraction(p: &str, q: &str, radix: u32, negative: bool) -> Result<Rational, ParseRationalError> {
    if let (Some(x), Some(y)) = (word(p.as_bytes(), radix), word(q.as_bytes(), radix)) {
        if y == 0 {
            return Err(error(RationalErrorKind::ZeroDenominator));
        }
        if x == 0 {
            return Ok(small_integer(false, 0));
        }
        let g = gcd(x, y);
        return Ok(Rational::from_word(
            negative,
            x / g,
            y / g,
            ByteOrder::LittleEndian,
        ));
    }
    long_fraction(p, q, radix, negative)
}

/// [`fraction`] of parts that are not both words, or not digits.
#[inline(never)]
fn long_fraction(
    p: &str,
    q: &str,
    radix: u32,
    negative: bool,
) -> Result<Rational, ParseRationalError> {
    let num = magnitude(p, radix)?;
    let den = magnitude(q, radix)?;
    if den.is_zero() {
        return Err(error(RationalErrorKind::ZeroDenominator));
    }
    Ok(Rational::new(with_sign(num, negative), den))
}

/// `10^k` for `k` up to 19.
const TENS: [u64; 20] = {
    let mut t = [1u64; 20];
    let mut i = 1;
    while i < 20 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// `5^k` for `k` up to 27.
const FIVES: [u64; 28] = {
    let mut t = [1u64; 28];
    let mut i = 1;
    while i < 28 {
        t[i] = t[i - 1] * 5;
        i += 1;
    }
    t
};

/// The decimal of the sign `negative` whose integer digits are `int`, and `rest` the point, the
/// digits after it and the exponent, or the exponent alone, starting with `.`, `e` or `E`.
fn decimal(int: &str, rest: &str, negative: bool) -> Result<Rational, ParseRationalError> {
    let invalid = || error(RationalErrorKind::InvalidDigit);
    let (frac, exponent) = match rest.strip_prefix('.') {
        Some(after) => match after.bytes().position(|c| c == b'e' || c == b'E') {
            Some(j) => (&after[..j], Some(&after[j + 1..])),
            None => (after, None),
        },
        None => ("", Some(&rest[1..])),
    };
    if int.is_empty() && frac.is_empty() {
        return Err(invalid());
    }
    // The exponent: a sign and at least one digit, its value held at `u64::MAX`.
    let (exponent_negative, x) = match exponent {
        None => (false, 0u64),
        Some(e) => {
            let (en, digits) = match e.as_bytes().first() {
                Some(b'-') => (true, &e[1..]),
                Some(b'+') => (false, &e[1..]),
                _ => (false, e),
            };
            if digits.is_empty() {
                return Err(invalid());
            }
            let mut x = 0u64;
            for c in digits.bytes() {
                let d = c.wrapping_sub(b'0');
                if d > 9 {
                    return Err(invalid());
                }
                x = x.saturating_mul(10).saturating_add(u64::from(d));
            }
            (en, x)
        }
    };
    if x > Rational::MAX_PARSE_EXPONENT {
        // The digits are checked first, so that text with a wrong character is always
        // `InvalidDigit`.
        let digits = |s: &str| s.bytes().all(|c| c.is_ascii_digit());
        if !digits(int) || !digits(frac) {
            return Err(invalid());
        }
        return Err(error(RationalErrorKind::ExponentOutOfRange));
    }
    // The value is `digits · 10^k`: the trailing zeros of the digits dropped, each raising `k`.
    let mut k = if exponent_negative {
        -(x as i64)
    } else {
        x as i64
    };
    let frac_digits = frac.trim_end_matches('0');
    // The digits, in one piece or two: the integer's and the fraction's.
    let (high, low) = if frac_digits.is_empty() {
        let int_digits = int.trim_end_matches('0');
        k += (int.len() - int_digits.len()) as i64;
        if int_digits.is_empty() {
            // Zeros alone: zero.
            return Ok(small_integer(false, 0));
        }
        (int_digits, "")
    } else {
        k -= frac_digits.len() as i64;
        (int, frac_digits)
    };
    // Digits that make a word and a power of ten of up to 19, in registers.
    if k.unsigned_abs() < 20
        && let Some(v) = word_pair(high.as_bytes(), low.as_bytes())
    {
        let f = k.unsigned_abs() as usize;
        if k >= 0 {
            return Ok(small_integer(negative, u128::from(v) * u128::from(TENS[f])));
        }
        // `v/10^f`, `v` not divisible by 10: its twos or its fives, up to `f`, are the divisor.
        let (num, den) = if v % 2 == 0 {
            let t = v.trailing_zeros().min(f as u32);
            (v >> t, TENS[f] >> t)
        } else if v % 5 == 0 {
            let mut c = 1;
            while c < f && v % FIVES[c + 1] == 0 {
                c += 1;
            }
            (v / FIVES[c], TENS[f] / FIVES[c])
        } else {
            (v, TENS[f])
        };
        return Ok(Rational::from_word(
            negative,
            num,
            den,
            ByteOrder::LittleEndian,
        ));
    }
    let joined;
    let digits: &str = if high.is_empty() {
        low
    } else if low.is_empty() {
        high
    } else {
        let mut s = String::with_capacity(high.len() + low.len());
        s.push_str(high);
        s.push_str(low);
        joined = s;
        &joined
    };
    let n = magnitude(digits, 10)?;
    let f = u32::try_from(k.unsigned_abs())
        .map_err(|_| error(RationalErrorKind::ExponentOutOfRange))?;
    let five = |e: u32| Number::from(5u8).pow(e);
    if k >= 0 {
        let n = (&n * &five(f)) << f as usize;
        return Ok(Rational::from_reduced(
            with_sign(n, negative),
            one(ByteOrder::LittleEndian),
        ));
    }
    // `n/10^f`, `n` not divisible by 10: its last digit says which of 2 and 5 it may share.
    let last = digits.as_bytes()[digits.len() - 1] - b'0';
    let (num, den) = if last.is_multiple_of(2) {
        let t = n.trailing_zeros().unwrap_or(0).min(u64::from(f)) as usize;
        (n >> t, five(f) << (f as usize - t))
    } else if last == 5 {
        // The fives of `n` up to 27 from one remainder by `5^27 < 2^63`, as `n ≡ r` modulo
        // `5^27`; past them, by a gcd with `5^f`.
        let r = u64::try_from(&n % FIVES[27]).unwrap_or(0);
        let v = if r == 0 {
            27
        } else {
            (0..27)
                .take_while(|&i| r.is_multiple_of(FIVES[i + 1]))
                .count() as u32
        }
        .min(f);
        if v == 27 && f > 27 {
            let p = five(f);
            let g = n.gcd(&p);
            (n.div_exact(&g), p.div_exact(&g) << f as usize)
        } else {
            (n / FIVES[v as usize], five(f - v) << f as usize)
        }
    } else {
        (n, five(f) << f as usize)
    };
    Ok(Rational::from_reduced(with_sign(num, negative), den))
}

/// The error of the kind `kind`.
#[cold]
fn error(kind: RationalErrorKind) -> ParseRationalError {
    ParseRationalError { kind }
}

/// An error from [`Rational`]'s [`FromStr`] or [`Rational::from_str_radix`]: what was wrong with
/// the text, from [`kind`](Self::kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRationalError {
    kind: RationalErrorKind,
}

/// What [`ParseRationalError`] found wrong with the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RationalErrorKind {
    /// The text is empty.
    Empty,
    /// A character is not a digit where one should be or out of place, or a part has no digits.
    InvalidDigit,
    /// The denominator is zero.
    ZeroDenominator,
    /// The exponent of a decimal is more than [`Rational::MAX_PARSE_EXPONENT`] in magnitude.
    ExponentOutOfRange,
}

impl ParseRationalError {
    /// What was wrong with the text.
    pub fn kind(&self) -> &RationalErrorKind {
        &self.kind
    }
}

impl fmt::Display for ParseRationalError {
    /// Writes what was wrong.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            RationalErrorKind::Empty => "cannot parse rational from empty string",
            RationalErrorKind::InvalidDigit => "invalid digit found in string",
            RationalErrorKind::ZeroDenominator => "the denominator is zero",
            RationalErrorKind::ExponentOutOfRange => "exponent out of range",
        }
        .fmt(f)
    }
}

impl std::error::Error for ParseRationalError {}
