//! Rationals as text: [`Display`](fmt::Display) writes `-3/4`, or `5` for an integer, and
//! [`Debug`](fmt::Debug) writes the same; [`FromStr`] reads integers, fractions and decimals with
//! an exponent, exactly; [`Rational::from_str_radix`] and [`Rational::to_str_radix`] read and write
//! integers and fractions in the radices from 2 to 36; and the conversions from integers.
//!
//! The parts are read and written as [`Number`]s are, by divide and conquer from a few hundred
//! digits on, in time `O(M(n) log n)` for `n` digits; a fraction read is then reduced by the gcd of
//! its parts. A decimal `d₁…dₙ.f₁…fₘ e x` is the integer `d₁…dₙf₁…fₘ` (its trailing zeros dropped,
//! each adding one to the exponent) times `10^(x − m)`. A negative power of ten leaves the
//! fraction `N/10^k` in lowest terms but for the factors 2 or 5 of `N`, which its last digit tells
//! apart: one of `1, 3, 7, 9` shares neither with `10^k`, an even one only twos, which a shift
//! removes, and `5` only fives, which a remainder by `5^27` counts, a gcd with `5^k` past that.
//! Text of up to 19 digits, with an exponent of up to 19 in magnitude, is read in registers.
//! `Display` writes parts of up to 64 bits two digits at a time from a table, and fractions of up
//! to 320 characters on the stack, so that the only allocation is the text the formatter keeps.

use super::small::{self, Small, gcd};
use super::{Rational, is_one, one, trim};
use crate::math::numbers::{ByteOrder, Number};
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
        if is_one(&self.den) {
            return fmt::Display::fmt(&self.num, f);
        }
        let negative = self.num.is_negative() && !self.num.is_zero();
        if let Some(x) = Small::of(self) {
            let mut buf = [0u8; 41];
            let len = small_text(x.num, x.den, &mut buf);
            return f.pad_integral(!negative, "", ascii(&buf[..len]));
        }
        // At most `log10(256) < 2.41` digits a byte, and one more, for each part, and the sign.
        let bound = (self.num.value.len() + self.den.value.len()) * 5 / 2 + 4;
        if bound <= STACK_TEXT {
            // Written on the stack, and from there once to the formatter.
            let mut text = StackText {
                bytes: [0; STACK_TEXT],
                len: 0,
            };
            write!(text, "{}/{}", self.num, self.den)?;
            return f.pad_integral(!negative, "", text.as_str(usize::from(negative)));
        }
        let mut s = String::with_capacity(bound);
        write!(s, "{}/{}", self.num, self.den)?;
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

/// Writes the decimal digits of `x` at the end of `buf`, two at a time; returns where they start.
#[inline(always)]
fn digits_u64(mut x: u64, buf: &mut [u8; 20]) -> usize {
    let mut i = 20;
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

/// Writes `num/den` in decimal at the start of `out`; returns the length.
#[inline]
fn small_text(num: u64, den: u64, out: &mut [u8; 41]) -> usize {
    let mut buf = [0u8; 20];
    let i = digits_u64(num, &mut buf);
    let a = 20 - i;
    out[..a].copy_from_slice(&buf[i..]);
    out[a] = b'/';
    let j = digits_u64(den, &mut buf);
    out[a + 1..a + 1 + 20 - j].copy_from_slice(&buf[j..]);
    a + 21 - j
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
    /// The integer `n` as `n/1`, in new vectors.
    #[inline]
    fn from(n: &Number) -> Rational {
        Rational::from_integer(n.clone())
    }
}

macro_rules! from_primitive {
    ($($t:ty),*) => {$(
        impl From<$t> for Rational {
            #[doc = concat!("The `", stringify!($t), "` `v` as `v/1`, little-endian.")]
            #[inline]
            fn from(v: $t) -> Rational {
                let mut num = Number::from(v);
                trim(&mut num);
                Rational::from_reduced(num, one(ByteOrder::LittleEndian))
            }
        }
    )*};
}

from_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

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
        if is_one(&self.den) {
            return self.num.to_str_radix(radix);
        }
        if radix == 10 {
            if let Some(x) = Small::of(self) {
                let mut buf = [0u8; 41];
                let len = small_text(x.num, x.den, &mut buf);
                let mut s = String::with_capacity(len + usize::from(x.negative));
                if x.negative {
                    s.push('-');
                }
                s.push_str(ascii(&buf[..len]));
                return s;
            }
            // In one vector, from the buffers of the formatting of the parts.
            let bound = (self.num.value.len() + self.den.value.len()) * 5 / 2 + 4;
            let mut s = String::with_capacity(bound);
            let _ = write!(s, "{}/{}", self.num, self.den);
            return s;
        }
        let den = self.den.to_str_radix(radix);
        let mut s = self.num.to_str_radix(radix);
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

/// The value of up to 19 decimal digits, `None` for more, none, or a character that is not one.
#[inline(always)]
fn word(run: &[u8]) -> Option<u64> {
    if run.is_empty() || run.len() > 19 {
        return None;
    }
    let mut v = 0u64;
    for &c in run {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + u64::from(d);
    }
    Some(v)
}

/// The value of the decimal digits `high` followed by `low`, up to 19 of them in all, `None` for
/// none or a character that is not a digit.
#[inline(always)]
fn word_pair(high: &[u8], low: &[u8]) -> Option<u64> {
    let mut v = 0u64;
    for &c in high.iter().chain(low) {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + u64::from(d);
    }
    (!high.is_empty() || !low.is_empty()).then_some(v)
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

/// The integer of the sign `negative` and the magnitude `m`, little-endian.
#[inline(always)]
fn small_integer(negative: bool, m: u128) -> Rational {
    Rational::from_reduced(
        small::number(ByteOrder::LittleEndian, negative, m),
        one(ByteOrder::LittleEndian),
    )
}

/// The integer of the sign `negative` and the digits `run` in `radix`.
fn integer(run: &str, radix: u32, negative: bool) -> Result<Rational, ParseRationalError> {
    if radix == 10
        && let Some(v) = word(run.as_bytes())
    {
        return Ok(small_integer(negative, u128::from(v)));
    }
    let n = with_sign(magnitude(run, radix)?, negative);
    Ok(Rational::from_reduced(n, one(ByteOrder::LittleEndian)))
}

/// The fraction `±p/q` of the digits `p` and `q` in `radix`, in lowest terms.
fn fraction(p: &str, q: &str, radix: u32, negative: bool) -> Result<Rational, ParseRationalError> {
    if radix == 10
        && let (Some(x), Some(y)) = (word(p.as_bytes()), word(q.as_bytes()))
    {
        if y == 0 {
            return Err(error(RationalErrorKind::ZeroDenominator));
        }
        if x == 0 {
            return Ok(small_integer(false, 0));
        }
        let g = gcd(x, y);
        return Ok(Rational::from_reduced(
            small::number(ByteOrder::LittleEndian, negative, u128::from(x / g)),
            small::number(ByteOrder::LittleEndian, false, u128::from(y / g)),
        ));
    }
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
    // Digits of up to 19 and a power of ten of up to 19, in registers.
    if k.unsigned_abs() < 20
        && high.len() + low.len() <= 19
        && let Some(v) = word_pair(high.as_bytes(), low.as_bytes())
    {
        let ten = TENS[k.unsigned_abs() as usize];
        if k >= 0 {
            return Ok(small_integer(negative, u128::from(v) * u128::from(ten)));
        }
        let g = gcd(v, ten);
        return Ok(Rational::from_reduced(
            small::number(ByteOrder::LittleEndian, negative, u128::from(v / g)),
            small::number(ByteOrder::LittleEndian, false, u128::from(ten / g)),
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
