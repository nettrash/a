//! Floats as text and text as floats, correctly rounded, in any radix from 2 to 36: `Display`
//! (the shortest decimal that reads back as the float at its precision, or the digits a
//! formatter's precision asks for), `LowerExp` and `UpperExp`, [`BigFloat::to_string_radix`] and
//! [`BigFloat::to_digits`] (the digits of `mpfr_get_str`), and `FromStr` and
//! [`BigFloat::from_str_radix`] (the text of `mpfr_strtofr`).

use super::limbs;
use super::round::{round_mag, signed};
use super::scale::{self, Dir, round_int, round_value};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

/// The finite number `x`, not zero, as its sign, its mantissa's limbs and the exponent of their
/// lowest bit, and its exponent `e`, `2^(e-1) ≤ |x| < 2^e`.
fn finite(x: &BigFloat) -> Option<(bool, &[u64], i128, i64)> {
    match &x.kind {
        Kind::Finite {
            negative,
            exp,
            mant,
        } => Some((
            *negative,
            mant.as_slice(),
            i128::from(*exp) - 64 * mant.len() as i128,
            *exp,
        )),
        _ => None,
    }
}

/// The digits of the magnitude `d` in radix `b`, lower case; `0` for zero.
fn digit_text(d: &[u64], b: u32) -> Vec<u8> {
    match d.len() {
        0 => vec![b'0'],
        1 => word_digits(u128::from(d[0]), b),
        2 => word_digits(u128::from(d[0]) | u128::from(d[1]) << 64, b),
        _ => limbs::to_number(d, false, ByteOrder::LittleEndian)
            .to_str_radix(b)
            .into_bytes(),
    }
}

/// The digits of `v` in radix `b`, lower case: by chunks of the largest power of `b` in a word
/// while `v` is wider than a word, then by words, radix 10 with a constant divisor.
fn word_digits(mut v: u128, b: u32) -> Vec<u8> {
    let mut buf = [0u8; 130];
    let mut i = buf.len();
    let (chunk, t) = chunk_power(b);
    let write = |buf: &mut [u8; 130], i: &mut usize, mut x: u64, min: usize| {
        let start = *i;
        while x != 0 || start - *i < min {
            let d = if b == 10 {
                let d = (x % 10) as u8;
                x /= 10;
                d
            } else {
                let d = (x % u64::from(b)) as u8;
                x /= u64::from(b);
                d
            };
            *i -= 1;
            buf[*i] = if d < 10 { b'0' + d } else { b'a' + d - 10 };
        }
    };
    while v >> 64 != 0 {
        let r = (v % u128::from(chunk)) as u64;
        v /= u128::from(chunk);
        write(&mut buf, &mut i, r, t);
    }
    write(&mut buf, &mut i, v as u64, 1);
    buf[i..].to_vec()
}

/// The largest power `b^t` of the radix in a word, and `t`.
#[inline]
fn chunk_power(b: u32) -> (u64, usize) {
    let (mut p, mut t) = (u64::from(b), 1);
    while let Some(q) = p.checked_mul(u64::from(b)) {
        p = q;
        t += 1;
    }
    (p, t)
}

/// The `n ≥ 1` significant digits in radix `b` of the finite number `x`, not zero, rounded in
/// the mode `round`: their text, the exponent `E` of the first, `|x| ≈ d.ddd…·b^E`, and the
/// direction of the rounding.
///
/// The exponent is first taken as `floor((e - 1)·log_b(2))`, which is `E` or one less; the
/// integer `D` that `|x|·b^(n-1-E)` rounds to then has `n` digits, or `n + 1` when `E` was one
/// more, or `b^n` exactly when the rounding carried past the last `n`-digit integer, which is
/// `b^(n-1)` at the exponent `E + 1`.
pub(crate) fn digits(x: &BigFloat, b: u32, n: u64, round: Round) -> (Vec<u8>, i128, Ordering) {
    let (negative, m, lsb, e) = finite(x).expect("a finite number that is not zero");
    let dir = Dir::of(negative, round);
    let mut big_e = scale::floor_log(i128::from(e) - 1, b).0;
    loop {
        let k = i128::from(n) - 1 - big_e;
        let (d, o) = round_int(m, lsb, b, k, dir);
        let mut text = digit_text(&d, b);
        let len = text.len() as u64;
        if len == n + 1 {
            if o == Ordering::Greater && text[0] == b'1' && text[1..].iter().all(|&c| c == b'0') {
                text.truncate(n as usize);
                return (text, big_e + 1, signed(negative, o));
            }
            big_e += 1;
            continue;
        }
        if len < n || text[0] == b'0' {
            big_e -= 1;
            continue;
        }
        debug_assert_eq!(len, n);
        return (text, big_e, signed(negative, o));
    }
}

/// The number of digits in radix `b` that a number of `prec` bits needs to be read back exactly,
/// written to nearest and read to nearest, as `mpfr_get_str_ndigits`: `1 + ceil(p·log_b(2))`,
/// with `p = prec - 1` when `b` is a power of two and `prec` otherwise.
pub(crate) fn ndigits(b: u32, prec: u64) -> u64 {
    let (a, c) = scale::split(b);
    if c == 1 {
        return 1 + (prec - 1).div_ceil(u64::from(a));
    }
    // `p·log_b(2)` is not an integer, so its ceiling is one more than its floor.
    let (f, certain) = scale::floor_log(i128::from(prec), b);
    let f = if certain {
        f
    } else {
        // The floor is `f` or `f + 1`: `f + 1` when `b^(f+1) < 2^p`.
        let t = (f + 1) as u64;
        let bits = limbs::bit_len(&scale::pow_exact(c, t)) + u64::from(a) * t;
        if bits <= prec { f + 1 } else { f }
    };
    f as u64 + 2
}

/// The shortest decimal digits that read back to nearest at the precision of the finite `x`, not
/// zero, as `x`, the nearest to `x` of them, and the exponent `E` of the first, `|x| ≈ d.ddd…·10^E`.
///
/// The numbers that round to `x` lie between `lo = x - ulp/2` (`x - ulp/4` when the mantissa is
/// a power of two, its lower neighbour being nearer) and `hi = x + ulp/2`, both ends included when
/// the mantissa is even (the ties go to it). At a scale `10^k` where the interval's integers are
/// `L..=H` with `H - L < 10`, at most one is a multiple of ten: when there is one it is the
/// shortest, with its zeros dropped (any shorter decimal would be another multiple of ten);
/// otherwise every integer of the interval has as many digits, and the nearest to `x·10^k` is
/// taken.
pub(crate) fn shortest(x: &BigFloat) -> (Vec<u8>, i128) {
    let (_, mant, _, e) = finite(x).expect("a finite number that is not zero");
    let prec = x.prec;
    let unused = 64 * mant.len() as u64 - prec;
    let mp = limbs::shr(mant, unused);
    let lsb = i128::from(e) - i128::from(prec);
    let bottom = limbs::trailing_zeros(&mp) == prec - 1;
    let even = mp[0] & 1 == 0;
    let x4 = limbs::shl(&mp, 2);
    let lo4 = limbs::sub(&x4, &[if bottom { 1 } else { 2 }]);
    let hi4 = limbs::add(&x4, &[2]);
    // The interval is `2^lsb` wide (three quarters of it at a power of two): from `10^k` with
    // `k ≥ -lsb·log10(2)` on, it holds an integer.
    let mut k = scale::floor_log(-lsb, 10).0 + 1;
    // At a scale `10^k`, `k ≥ 0` and not too large, `x·10^k = N·2^s` with `N = 4·M·5^k` and the
    // ends `N ∓ 2·5^k` (`N - 5^k` at a power of two), one power for the three.
    let shared = |k: i128| -> Option<Scaled> {
        if k < 0 || k > 4 * (i128::from(prec) + 512) {
            return None;
        }
        let p = scale::pow_exact(5, k as u64);
        let n = limbs::mul(&x4, &p);
        let p2 = limbs::shl(&p, 1);
        let lo = limbs::sub(&n, if bottom { &p } else { &p2 });
        let hi = limbs::add(&n, &p2);
        Some((lo, n, hi, lsb - 2 + k))
    };
    let mut kept = None;
    let (mut lo, mut hi) = loop {
        let (l, h) = if let Some((nl, n, nh, s)) = shared(k) {
            let (mut l, _, z) = scale::int_parts(&nl, s);
            if !z || !even {
                limbs::increment(&mut l);
            }
            let (mut h, _, z) = scale::int_parts(&nh, s);
            if z && !even {
                h = limbs::sub(&h, &[1]);
            }
            kept = Some((k, n, s));
            (l, h)
        } else {
            let (mut l, o) = round_int(&lo4, lsb - 2, 10, k, Dir::Ceil);
            if o == Ordering::Equal && !even {
                limbs::increment(&mut l);
            }
            let (mut h, o) = round_int(&hi4, lsb - 2, 10, k, Dir::Floor);
            if o == Ordering::Equal && !even {
                h = limbs::sub(&h, &[1]);
            }
            (l, h)
        };
        if limbs::cmp(&l, &h) == Ordering::Greater {
            k += 1;
            continue;
        }
        break (l, h);
    };
    // Ten integers or more: a coarser scale holds one too.
    while limbs::cmp(&limbs::sub(&hi, &lo), &[10]) != Ordering::Less {
        let r = div_small(&mut lo, 10);
        if r != 0 {
            limbs::increment(&mut lo);
        }
        div_small(&mut hi, 10);
        k -= 1;
    }
    let r = rem_small(&hi, 10);
    let z = limbs::sub(&hi, &[r]);
    if limbs::cmp(&z, &lo) != Ordering::Less && !z.is_empty() {
        let mut text = digit_text(&z, 10);
        let len = text.len() as i128;
        while text.last() == Some(&b'0') {
            text.pop();
        }
        return (text, len - 1 - k);
    }
    let (mut d, _) = match kept {
        Some((kk, n, s)) if kk == k => {
            let (t, h, z) = scale::int_parts(&n, s);
            scale::decide(t, h, z, Dir::Nearest)
        }
        _ => round_int(&x4, lsb - 2, 10, k, Dir::Nearest),
    };
    if limbs::cmp(&d, &lo) == Ordering::Less {
        d = lo;
    } else if limbs::cmp(&d, &hi) == Ordering::Greater {
        d = hi;
    }
    let text = digit_text(&d, 10);
    let len = text.len() as i128;
    (text, len - 1 - k)
}

/// The ends of the interval and the number at a scale, `(lo, x, hi)·2^s`.
type Scaled = (Vec<u64>, Vec<u64>, Vec<u64>, i128);

/// `v /= d` for a small `d`, in place, without zero limbs at the top: the remainder.
fn div_small(v: &mut Vec<u64>, d: u64) -> u64 {
    let mut r = 0u64;
    for x in v.iter_mut().rev() {
        let t = u128::from(r) << 64 | u128::from(*x);
        *x = (t / u128::from(d)) as u64;
        r = (t % u128::from(d)) as u64;
    }
    limbs::trim(v);
    r
}

/// `v mod d` for a small `d`.
fn rem_small(v: &[u64], d: u64) -> u64 {
    let mut r = 0u64;
    for &x in v.iter().rev() {
        r = ((u128::from(r) << 64 | u128::from(x)) % u128::from(d)) as u64;
    }
    r
}

/// Writes `digits` (at least one) in scientific notation, `d.ddd` and the exponent after
/// `marker`, into `out`.
fn scientific(out: &mut String, digits: &[u8], exp: i128, marker: char) {
    out.push(digits[0] as char);
    if digits.len() > 1 {
        out.push('.');
        // ASCII digits.
        out.push_str(core::str::from_utf8(&digits[1..]).expect("ASCII digits"));
    }
    out.push(marker);
    out.push_str(&exp.to_string());
}

/// Writes the digits `digits` of a number `d.ddd…·10^exp` in positional notation into `out`,
/// with `frac` digits after the point at least (zeros added), all the digits kept.
fn positional(out: &mut String, digits: &[u8], exp: i128) {
    let text = core::str::from_utf8(digits).expect("ASCII digits");
    let n = digits.len() as i128;
    if exp < 0 {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        out.push_str(text);
    } else if exp + 1 >= n {
        out.push_str(text);
        for _ in 0..(exp + 1 - n) {
            out.push('0');
        }
    } else {
        let (int, frac) = text.split_at((exp + 1) as usize);
        out.push_str(int);
        out.push('.');
        out.push_str(frac);
    }
}

/// The decimal exponent past which `Display` without a precision writes scientific notation
/// for numbers of few digits, as JavaScript does: `10^21` and beyond.
const POSITIONAL_MAX: i128 = 21;
/// The decimal exponent below which `Display` without a precision writes scientific notation.
const POSITIONAL_MIN: i128 = -5;

impl BigFloat {
    /// Writes the special values as an `f64` of the value would be written, with the formatter's
    /// flags; `None` for a finite number that is not zero.
    fn fmt_special(
        &self,
        f: &mut fmt::Formatter<'_>,
        write: fn(&f64, &mut fmt::Formatter<'_>) -> fmt::Result,
    ) -> Option<fmt::Result> {
        let v = match self.kind {
            Kind::Nan => f64::NAN,
            Kind::Inf { negative } => {
                if negative {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
            Kind::Zero { negative } => {
                if negative {
                    -0.0
                } else {
                    0.0
                }
            }
            Kind::Finite { .. } => return None,
        };
        Some(write(&v, f))
    }

    /// The text of `Display` with `frac` digits after the point, positional when the number is
    /// below `10^max(21, digits of the precision)`, and scientific with as many digits after the
    /// point otherwise, without the sign.
    fn fixed_text(&self, frac: usize) -> String {
        let (_, m, lsb, e) = finite(self).expect("a finite number that is not zero");
        let limit = POSITIONAL_MAX.max(ndigits(10, self.prec) as i128);
        let est = scale::floor_log(i128::from(e) - 1, 10).0;
        let mut out = String::new();
        if est < limit {
            let (d, _) = round_int(m, lsb, 10, frac as i128, Dir::Nearest);
            let text = digit_text(&d, 10);
            let int_digits = text.len() as i128 - frac as i128;
            if int_digits <= limit || d.is_empty() {
                // The integer digits, then the fraction's, zeros in front as needed.
                let pad = (frac + 1).saturating_sub(text.len());
                let mut all = String::with_capacity(pad + text.len() + 1);
                for _ in 0..pad {
                    all.push('0');
                }
                all.push_str(core::str::from_utf8(&text).expect("ASCII digits"));
                let cut = all.len() - frac;
                out.push_str(&all[..cut]);
                if frac > 0 {
                    out.push('.');
                    out.push_str(&all[cut..]);
                }
                return out;
            }
        }
        let (text, exp, _) = digits(self, 10, frac as u64 + 1, Round::Nearest);
        scientific(&mut out, &text, exp, 'e');
        out
    }
}

impl fmt::Display for BigFloat {
    /// Writes the number in decimal, as `f64` does: without a precision, the shortest digits that
    /// read back to nearest at the number's precision as the number itself (`0.1` for
    /// `BigFloat::from(0.1)`), and with one, `{:.3}`, that many digits after the point, rounded
    /// to nearest from the exact value. Positional notation, or scientific (`1.5e30`, as
    /// [`LowerExp`](fmt::LowerExp)) when positional would add zeros past the digits for a number
    /// of `10^21` or more, or below `10^-5`; with a precision, for a number beyond
    /// `10^max(21, d)`, `d` the digits that the precision needs ([`BigFloat::to_digits`]).
    /// NaN, the infinities and the zeros as `f64`'s (`NaN`, `inf`, `-inf`, `0`, `-0`); width,
    /// fill, alignment, `+` and `0` as for numbers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(0.1).to_string(), "0.1");
    /// assert_eq!(BigFloat::from(-1.5e300).to_string(), "-1.5e300");
    /// assert_eq!(format!("{:.3}", BigFloat::from(2.0 / 3.0)), "0.667");
    /// assert_eq!(format!("{:+08.2}", BigFloat::from(3.14159)), "+0003.14");
    /// assert_eq!(BigFloat::from(1u128 << 100).to_string(), "1267650600228229401496703205376");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(r) = self.fmt_special(f, <f64 as fmt::Display>::fmt) {
            return r;
        }
        let text = match f.precision() {
            Some(frac) => self.fixed_text(frac),
            None => {
                let (digits, exp) = shortest(self);
                let mut out = String::new();
                let limit = POSITIONAL_MAX.max(digits.len() as i128);
                if (POSITIONAL_MIN..limit).contains(&exp) {
                    positional(&mut out, &digits, exp);
                } else {
                    scientific(&mut out, &digits, exp, 'e');
                }
                out
            }
        };
        f.pad_integral(!self.is_sign_negative(), "", &text)
    }
}

impl BigFloat {
    /// Writes the number in scientific notation with the exponent marker `marker`.
    fn fmt_exp(&self, f: &mut fmt::Formatter<'_>, marker: char) -> fmt::Result {
        let (digits, exp) = match f.precision() {
            Some(frac) => {
                let (d, e, _) = digits(self, 10, frac as u64 + 1, Round::Nearest);
                (d, e)
            }
            None => shortest(self),
        };
        let mut out = String::new();
        scientific(&mut out, &digits, exp, marker);
        f.pad_integral(!self.is_sign_negative(), "", &out)
    }
}

impl fmt::LowerExp for BigFloat {
    /// Writes the number in scientific notation, as `f64` does: `1.5e30`, `-2e-7`; without a
    /// precision the digits of [`Display`](fmt::Display), the shortest that read back, with one,
    /// `{:.3e}`, that many digits after the point, rounded to nearest.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(format!("{:e}", BigFloat::from(1234.5)), "1.2345e3");
    /// assert_eq!(format!("{:.2e}", BigFloat::from(1234.5)), "1.23e3");
    /// assert_eq!(format!("{:e}", BigFloat::from(-0.00025)), "-2.5e-4");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(r) = self.fmt_special(f, <f64 as fmt::LowerExp>::fmt) {
            return r;
        }
        self.fmt_exp(f, 'e')
    }
}

impl fmt::UpperExp for BigFloat {
    /// As [`LowerExp`](fmt::LowerExp), with `E`: `1.5E30`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(r) = self.fmt_special(f, <f64 as fmt::UpperExp>::fmt) {
            return r;
        }
        self.fmt_exp(f, 'E')
    }
}

/// Checks a radix of output, panicking outside 2..=36.
#[inline]
#[track_caller]
fn check_radix(radix: u32) {
    assert!((2..=36).contains(&radix), "radix {radix} is not in 2..=36");
}

impl BigFloat {
    /// The digits of the number in radix `radix` (from 2 to 36, lower-case letters) rounded in
    /// the mode `round`, as `mpfr_get_str`: the digit text, the exponent `E` of the number
    /// `0.d₁d₂…·radix^E` (the point before the first digit), and the direction of the rounding;
    /// `None` for NaN, the infinities and the zeros.
    ///
    /// With `digits` of `None`, as many digits as any number of the precision needs to be read
    /// back exactly, written to nearest and read to nearest: `1 + ceil(p·log_radix(2))`, `p` the
    /// precision, or one less than it when the radix is a power of two (`mpfr_get_str_ndigits`).
    /// To nearest, a number halfway between two outputs goes to the one whose digits are an even
    /// integer.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36, or `digits` is `Some(0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let x = BigFloat::from(-3.1416);
    /// let (d, e, o) = x.to_digits(10, Some(5), Round::Nearest).unwrap();
    /// assert_eq!((d.as_str(), e, o), ("31416", 1, Ordering::Less));
    /// assert_eq!(x.to_digits(10, None, Round::Nearest).unwrap().0, "31415999999999999");
    /// // 32 bits: 1 + ceil(31/4) = 9 hexadecimal digits.
    /// assert_eq!(BigFloat::from(255).to_digits(16, None, Round::Nearest).unwrap().0, "ff0000000");
    /// ```
    #[track_caller]
    pub fn to_digits(
        &self,
        radix: u32,
        digits: Option<usize>,
        round: Round,
    ) -> Option<(String, i64, Ordering)> {
        check_radix(radix);
        let n = match digits {
            Some(0) => panic!("a number of digits of zero"),
            Some(n) => n as u64,
            None => ndigits(radix, self.prec),
        };
        if !self.is_normal() {
            return None;
        }
        let (text, e, o) = self::digits(self, radix, n, round);
        let text = String::from_utf8(text).expect("ASCII digits");
        Some((text, (e + 1) as i64, o))
    }

    /// The number in radix `radix` (from 2 to 36, lower-case letters), its digits rounded in the
    /// mode `round`, in scientific notation: a sign for a negative number, the first digit, the
    /// point and the others when there are others, then `e` (radix 10 and below) or `@` (above)
    /// and the exponent of the radix in decimal, `-1.25e-3`, `f.fc@2`. As many digits as
    /// `digits`, or with `None` as any number of the precision needs to be read back exactly
    /// ([`to_digits`](Self::to_digits)). NaN, the infinities and the zeros are `nan`, `inf`,
    /// `-inf`, `0` and `-0`, or `@nan@` and `@inf@` above radix 16, where `nan` and `inf` could be
    /// digits; [`from_str_radix`](Self::from_str_radix) reads them all back.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36, or `digits` is `Some(0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let x = BigFloat::from(1.5);
    /// assert_eq!(x.to_string_radix(10, None, Round::Nearest), "1.5000000000000000e0");
    /// assert_eq!(x.to_string_radix(2, Some(3), Round::Nearest), "1.10e0");
    /// assert_eq!(BigFloat::from(-1000).to_string_radix(16, Some(2), Round::Up), "-3.e@2");
    /// let third = BigFloat::from(1.0) / BigFloat::from(3.0);
    /// assert_eq!(third.to_string_radix(10, Some(3), Round::Up), "3.34e-1");
    /// let (back, _) =
    ///     BigFloat::from_str_radix(&third.to_string_radix(7, None, Round::Nearest), 7, 53, Round::Nearest)
    ///         .unwrap();
    /// assert_eq!(back, third);
    /// ```
    #[track_caller]
    pub fn to_string_radix(&self, radix: u32, digits: Option<usize>, round: Round) -> String {
        check_radix(radix);
        let n = match digits {
            Some(0) => panic!("a number of digits of zero"),
            Some(n) => n as u64,
            None => ndigits(radix, self.prec),
        };
        let words = radix <= 16;
        match self.kind {
            Kind::Nan => return (if words { "nan" } else { "@nan@" }).to_string(),
            Kind::Inf { negative } => {
                let s = if words { "inf" } else { "@inf@" };
                return if negative {
                    format!("-{s}")
                } else {
                    s.to_string()
                };
            }
            Kind::Zero { negative } => return (if negative { "-0" } else { "0" }).to_string(),
            Kind::Finite { .. } => {}
        }
        let (text, e, _) = self::digits(self, radix, n, round);
        let mut out = String::with_capacity(text.len() + 24);
        if self.is_sign_negative() {
            out.push('-');
        }
        scientific(&mut out, &text, e, if radix <= 10 { 'e' } else { '@' });
        out
    }
}

// Reading

/// An error from [`BigFloat`]'s [`FromStr`] or [`BigFloat::from_str_radix`]: what was wrong with
/// the text, from [`kind`](Self::kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBigFloatError {
    kind: BigFloatErrorKind,
}

/// What [`ParseBigFloatError`] found wrong with the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BigFloatErrorKind {
    /// The text is empty.
    Empty,
    /// A character is not a digit of the radix where one should be or is out of place, or a part
    /// has no digits.
    InvalidDigit,
}

impl ParseBigFloatError {
    /// What was wrong with the text.
    pub fn kind(&self) -> &BigFloatErrorKind {
        &self.kind
    }
}

impl fmt::Display for ParseBigFloatError {
    /// Writes what was wrong.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            BigFloatErrorKind::Empty => "cannot parse float from empty string",
            BigFloatErrorKind::InvalidDigit => "invalid float literal",
        }
        .fmt(f)
    }
}

impl core::error::Error for ParseBigFloatError {}

/// The error of the kind `kind`.
#[cold]
fn error(kind: BigFloatErrorKind) -> ParseBigFloatError {
    ParseBigFloatError { kind }
}

/// The value of the digit `c` in a radix up to 36, letters in either case; 36 or more for any
/// other character.
#[inline]
fn digit_value(c: u8) -> u32 {
    match c {
        b'0'..=b'9' => u32::from(c - b'0'),
        b'a'..=b'z' => u32::from(c - b'a') + 10,
        b'A'..=b'Z' => u32::from(c - b'A') + 10,
        _ => 36,
    }
}

/// The text of a number, read.
enum Parsed<'a> {
    /// NaN.
    Nan,
    /// An infinity of the sign.
    Inf(bool),
    /// `±digits·radix^exp·2^bexp`, the digits (characters) without the point.
    Number {
        negative: bool,
        radix: u32,
        /// The digits from the first that is not zero to the last that is not zero, empty for
        /// zero.
        digits: Digits<'a>,
        /// The digits from the first that is not zero to the end of the text's digits.
        significant: u64,
        /// The exponent of the radix of the last digit of `digits`.
        exp: i128,
        /// The binary exponent of a `p` exponent.
        bexp: i128,
    },
}

/// Digits, in the text or copied out of it around the point.
enum Digits<'a> {
    Text(&'a [u8]),
    Owned(Vec<u8>),
}

impl Digits<'_> {
    #[inline]
    fn get(&self) -> &[u8] {
        match self {
            Digits::Text(t) => t,
            Digits::Owned(v) => v,
        }
    }
}

/// The bound at which an exponent in the text is held: past it every number overflows or
/// underflows, its digits being fewer than `2^60`.
const EXP_HOLD: i128 = 1 << 64;

/// Whether `s` starts with `word`, ignoring the case of ASCII letters.
#[inline]
fn starts_with_ci(s: &[u8], word: &[u8]) -> bool {
    s.len() >= word.len() && s[..word.len()].eq_ignore_ascii_case(word)
}

/// The optional `(n-char-sequence)` of a NaN, and nothing after it.
fn nan_tail(s: &[u8]) -> bool {
    if s.is_empty() {
        return true;
    }
    s[0] == b'('
        && s.len() >= 2
        && s[s.len() - 1] == b')'
        && s[1..s.len() - 1]
            .iter()
            .all(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

/// Reads the text of a number in `radix`, 0 for the radix its prefix gives.
fn parse(src: &str, radix: u32) -> Result<Parsed<'_>, ParseBigFloatError> {
    let invalid = || error(BigFloatErrorKind::InvalidDigit);
    let bytes = src.as_bytes();
    let (negative, mut s) = match bytes.first() {
        None => return Err(error(BigFloatErrorKind::Empty)),
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        Some(_) => (false, bytes),
    };
    if s.is_empty() {
        return Err(invalid());
    }
    // The special values.
    if starts_with_ci(s, b"@inf@") {
        return if s.len() == 5 {
            Ok(Parsed::Inf(negative))
        } else {
            Err(invalid())
        };
    }
    if starts_with_ci(s, b"@nan@") {
        return if nan_tail(&s[5..]) {
            Ok(Parsed::Nan)
        } else {
            Err(invalid())
        };
    }
    if radix <= 16 {
        if s.eq_ignore_ascii_case(b"inf") || s.eq_ignore_ascii_case(b"infinity") {
            return Ok(Parsed::Inf(negative));
        }
        if starts_with_ci(s, b"nan") {
            return if nan_tail(&s[3..]) {
                Ok(Parsed::Nan)
            } else {
                Err(invalid())
            };
        }
    }
    let mut radix = radix;
    if s.len() > 2 && s[0] == b'0' {
        if (radix == 0 || radix == 16) && (s[1] | 0x20) == b'x' {
            radix = 16;
            s = &s[2..];
        } else if (radix == 0 || radix == 2) && (s[1] | 0x20) == b'b' {
            radix = 2;
            s = &s[2..];
        }
    }
    if radix == 0 {
        radix = 10;
    }
    // The digits, the point, the digits.
    let mut i = 0;
    while i < s.len() && digit_value(s[i]) < radix {
        i += 1;
    }
    let int = &s[..i];
    let mut frac: &[u8] = &[];
    if i < s.len() && s[i] == b'.' {
        let start = i + 1;
        i = start;
        while i < s.len() && digit_value(s[i]) < radix {
            i += 1;
        }
        frac = &s[start..i];
    }
    if int.is_empty() && frac.is_empty() {
        return Err(invalid());
    }
    // The exponent.
    let (mut exp, mut bexp) = (0i128, 0i128);
    if i < s.len() {
        let c = s[i];
        let binary = (c | 0x20) == b'p' && (radix == 2 || radix == 16);
        if !(c == b'@' || (c | 0x20) == b'e' && radix <= 10 || binary) {
            return Err(invalid());
        }
        let mut t = &s[i + 1..];
        let en = match t.first() {
            Some(b'-') => {
                t = &t[1..];
                true
            }
            Some(b'+') => {
                t = &t[1..];
                false
            }
            _ => false,
        };
        if t.is_empty() || !t.iter().all(u8::is_ascii_digit) {
            return Err(invalid());
        }
        let mut x = 0i128;
        for &c in t {
            x = (x * 10 + i128::from(c - b'0')).min(EXP_HOLD);
        }
        let x = if en { -x } else { x };
        if binary {
            bexp = x;
        } else {
            exp = x;
        }
    }
    // The significant digits.
    let first_int = int.iter().position(|&c| c != b'0');
    let last_frac = frac.iter().rposition(|&c| c != b'0');
    let (digits, significant, last_exp) = match (first_int, last_frac) {
        (Some(f), None) => {
            let l = int
                .iter()
                .rposition(|&c| c != b'0')
                .expect("a digit that is not zero");
            (
                Digits::Text(&int[f..=l]),
                (int.len() - f + frac.len()) as u64,
                (int.len() - 1 - l) as i128,
            )
        }
        (Some(f), Some(l)) => {
            let mut v = Vec::with_capacity(int.len() - f + l + 1);
            v.extend_from_slice(&int[f..]);
            v.extend_from_slice(&frac[..=l]);
            (
                Digits::Owned(v),
                (int.len() - f + frac.len()) as u64,
                -(l as i128) - 1,
            )
        }
        (None, Some(l)) => {
            let f = frac
                .iter()
                .position(|&c| c != b'0')
                .expect("a digit that is not zero");
            (
                Digits::Text(&frac[f..=l]),
                (frac.len() - f) as u64,
                -(l as i128) - 1,
            )
        }
        (None, None) => (Digits::Text(&[]), 0, 0),
    };
    Ok(Parsed::Number {
        negative,
        radix,
        digits,
        significant,
        exp: (exp + last_exp).clamp(-EXP_HOLD, EXP_HOLD),
        bexp,
    })
}

/// The magnitude of the digits `d` (characters, at least one) in `radix`.
fn digits_value(d: &[u8], radix: u32) -> Vec<u64> {
    let bits = d.len() as u64 * u64::from(32 - (radix - 1).leading_zeros());
    if bits <= 128 {
        let mut v = 0u128;
        for &c in d {
            v = v * u128::from(radix) + u128::from(digit_value(c));
        }
        return scale::limbs_of(v);
    }
    if bits <= 4096 {
        // By chunks of the digits of a word: `v·b^t + chunk`.
        let (chunk, t) = chunk_power(radix);
        let mut v: Vec<u64> = Vec::with_capacity((bits / 64 + 2) as usize);
        let head = d.len() % t;
        let mut acc = 0u64;
        for &c in &d[..head] {
            acc = acc * u64::from(radix) + u64::from(digit_value(c));
        }
        if acc != 0 {
            v.push(acc);
        }
        for part in d[head..].chunks(t) {
            let mut x = 0u64;
            for &c in part {
                x = x * u64::from(radix) + u64::from(digit_value(c));
            }
            scale::mul_1(&mut v, chunk);
            if v.is_empty() {
                v.push(x);
            } else {
                let mut carry = x;
                for limb in v.iter_mut() {
                    let (s, c) = limb.overflowing_add(carry);
                    *limb = s;
                    carry = u64::from(c);
                    if carry == 0 {
                        break;
                    }
                }
                if carry != 0 {
                    v.push(carry);
                }
            }
        }
        limbs::trim(&mut v);
        return v;
    }
    // ASCII digits and letters.
    let text = core::str::from_utf8(d).expect("ASCII digits");
    let n = Number::from_str_radix(text, radix, ByteOrder::LittleEndian).expect("valid digits");
    limbs::of_number(&n)
}

/// The value of the number `±digits·radix^exp·2^bexp`, `digits` not empty, rounded to `prec`
/// bits in the mode `round`.
///
/// For a radix that is a power of two, the digits that hold two bits more than the precision,
/// and whether any digit after them is not zero, the sticky bit. Otherwise the first `m` digits
/// make `D`, and the number lies strictly between `D·radix^t` and `(D + 1)·radix^t`, `t` the
/// digits dropped, which are not all zeros: when both round to the same number, which is not
/// between them, it is the rounding of the number, and its direction is known; otherwise twice
/// as many digits, up to all of them.
fn value(
    negative: bool,
    digits: &[u8],
    radix: u32,
    exp: i128,
    bexp: i128,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let n = digits.len() as u64;
    let (a, c) = scale::split(radix);
    if c == 1 {
        let a = u64::from(a);
        let keep = (prec + 2).div_ceil(a) + 1;
        if n <= keep {
            let d = digits_value(digits, radix);
            return round_mag(negative, &d, i128::from(a) * exp + bexp, false, prec, round);
        }
        let d = digits_value(&digits[..keep as usize], radix);
        // `keep` digits, the first not zero, hold `(keep - 1)·a + 1 > prec + 2` bits.
        return round_mag(
            negative,
            &d,
            i128::from(a) * (exp + i128::from(n - keep)) + bexp,
            true,
            prec,
            round,
        );
    }
    debug_assert_eq!(bexp, 0);
    if prec <= 62
        && let Some(r) = small_value(negative, digits, radix, exp, prec, round)
    {
        return r;
    }
    // Digits for 8 bits more than the precision, at most a word's for 53 bits: the interval of
    // the truncation decides but for one number in about 2^8; then twice as many digits.
    let mut m = (scale::floor_log(i128::from(prec) + 8, radix).0 as u64 + 1).min(n);
    loop {
        if m == n {
            let d = digits_value(digits, radix);
            return round_value(negative, &d, radix, exp, prec, round);
        }
        // Held at the bound, past which every number overflows or underflows alike.
        let t = (exp + i128::from(n - m)).clamp(-EXP_HOLD, EXP_HOLD);
        let d = digits_value(&digits[..m as usize], radix);
        let (lo, ol) = round_value(negative, &d, radix, t, prec, round);
        let mut d1 = d;
        limbs::increment(&mut d1);
        let (hi, oh) = round_value(negative, &d1, radix, t, prec, round);
        if lo.kind == hi.kind {
            let (ml, mh) = if negative {
                (ol.reverse(), oh.reverse())
            } else {
                (ol, oh)
            };
            if mh != Ordering::Less {
                return (lo, signed(negative, Ordering::Greater));
            }
            if ml != Ordering::Greater {
                return (lo, signed(negative, Ordering::Less));
            }
        }
        m = (2 * m).min(n);
    }
}

/// `±m·2^lsb` (`m` not zero), or with `sticky` a number strictly between it and `±(m + 1)·2^lsb`
/// (`m` then of more than `prec` bits), rounded to `prec ≤ 64` bits in the mode `round`, in
/// registers: the mantissa at the top of a word, the exponent of `0.1…·2^e`, and the direction.
fn round_u128(
    negative: bool,
    m: u128,
    lsb: i128,
    sticky: bool,
    prec: u64,
    round: Round,
) -> (u64, i128, Ordering) {
    let bits = u64::from(128 - m.leading_zeros());
    let e0 = lsb + i128::from(bits);
    if bits <= prec {
        debug_assert!(!sticky);
        return ((m as u64) << (64 - bits), e0, Ordering::Equal);
    }
    let drop = bits - prec;
    let mut t = m >> drop;
    let half = m >> (drop - 1) & 1 == 1;
    let rest = sticky || m & ((1u128 << (drop - 1)) - 1) != 0;
    let up = match round {
        Round::Nearest => half && (rest || t & 1 == 1),
        _ => super::round::away(negative, round) && (half || rest),
    };
    let mut e = e0;
    if up {
        t += 1;
        if t >> prec != 0 {
            t >>= 1;
            e += 1;
        }
    }
    let o = if !(half || rest) {
        Ordering::Equal
    } else if up {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    ((t as u64) << (64 - prec), e, signed(negative, o))
}

/// The value of up to a word's digits times a power of the radix that fits a word, in registers:
/// `D·c^k·2^(a·k)` as a product of two words, or `D·2^s / c^j` as the quotient of a double word
/// by a word, of at least 63 bits, the remainder its sticky bit. A longer text is truncated to
/// the digits of a word, and taken when both ends of the truncation round alike and the
/// direction is known, as [`value`] does. `None` when the digits, the power or the result do not
/// fit, or the truncation does not decide.
fn small_value(
    negative: bool,
    digits: &[u8],
    radix: u32,
    exp: i128,
    prec: u64,
    round: Round,
) -> Option<(BigFloat, Ordering)> {
    let (a, c) = scale::split(radix);
    let (chunk, t) = chunk_power(radix);
    let _ = chunk;
    let n = digits.len();
    let m = n.min(t);
    let k = exp + (n - m) as i128;
    let mut d = 0u64;
    for &ch in &digits[..m] {
        d = d * u64::from(radix) + u64::from(digit_value(ch));
    }
    let j = k.unsigned_abs();
    if j > 40 {
        return None;
    }
    let pw = c.checked_pow(j as u32)?;
    let one = |d: u64| -> (u64, i128, Ordering) {
        if k >= 0 {
            let p = u128::from(d) * u128::from(pw);
            round_u128(negative, p, i128::from(a) * k, false, prec, round)
        } else {
            let s = u64::from(d.leading_zeros()) + 63;
            let num = u128::from(d) << s;
            let (q, r) = (num / u128::from(pw), num % u128::from(pw));
            round_u128(
                negative,
                q,
                -i128::from(s) + i128::from(a) * k,
                r != 0,
                prec,
                round,
            )
        }
    };
    let (mant, e, o) = if m == n {
        one(d)
    } else {
        // `D + 1` is at most `2^64`, which a word of the radix's digits never reaches.
        let lo = one(d);
        let hi = one(d.checked_add(1)?);
        if (lo.0, lo.1) != (hi.0, hi.1) {
            return None;
        }
        let (ml, mh) = if negative {
            (lo.2.reverse(), hi.2.reverse())
        } else {
            (lo.2, hi.2)
        };
        if mh != Ordering::Less {
            (lo.0, lo.1, signed(negative, Ordering::Greater))
        } else if ml != Ordering::Greater {
            (lo.0, lo.1, signed(negative, Ordering::Less))
        } else {
            return None;
        }
    };
    if !(i128::from(BigFloat::EXP_MIN)..=i128::from(BigFloat::EXP_MAX)).contains(&e) {
        return None;
    }
    Some((
        BigFloat {
            prec,
            kind: Kind::Finite {
                negative,
                exp: e as i64,
                mant: vec![mant],
            },
        },
        o,
    ))
}

impl BigFloat {
    /// Reads a number in radix `radix` and rounds it to `prec` bits in the mode `round`, with the
    /// direction of the rounding, as `mpfr_strtofr` (but the whole text, and no whitespace):
    ///
    /// - an optional sign, `+` or `-`, then digits with an optional point, `12`, `1.5`, `.5`,
    ///   `5.`, letters for the digits past 9 in either case, and an optional exponent: `e` or `E`
    ///   (radix 10 and below) or `@` (any radix), an optional sign and decimal digits, the power of
    ///   the radix that multiplies the digits, `1.5e-3`, `ff@-2`; in radix 2 and 16, `p` or `P`
    ///   and a power of two in decimal instead, `0x1.8p3` is 12;
    /// - a prefix `0x` or `0X` in radix 16 and `0b` or `0B` in radix 2; radix 0 takes its radix
    ///   from the prefix, and 10 without one;
    /// - `inf`, `infinity`, `nan` or `nan(…)` (letters, digits and `_` in the parentheses) in
    ///   radix 16 and below, `@inf@` and `@nan@` in any, letters in either case.
    ///
    /// The value is that of all the digits, however many, rounded once: `0.1` to 53 bits is the
    /// `f64` `0.1`. An exponent so large or small that the number is beyond the exponent range
    /// overflows or underflows by the mode, without the work growing with it. `-0` is `-0`.
    ///
    /// # Errors
    ///
    /// [`ParseBigFloatError`] of kind [`BigFloatErrorKind::Empty`] for empty text, and
    /// [`BigFloatErrorKind::InvalidDigit`] for a character that is not a digit of the radix where
    /// one should be, or out of place, or a part with no digits.
    ///
    /// # Panics
    ///
    /// When `radix` is neither 0 nor from 2 to 36, or `prec` is not a precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from_str_radix("0.1", 10, 53, Round::Nearest).unwrap();
    /// assert_eq!((x, o), (BigFloat::from(0.1), Ordering::Greater));
    /// let (x, o) = BigFloat::from_str_radix("-ff.8@-1", 16, 10, Round::TowardZero).unwrap();
    /// assert_eq!((x, o), (BigFloat::from(-15.96875), Ordering::Equal));
    /// let (x, _) = BigFloat::from_str_radix("0x1.8p3", 0, 24, Round::Nearest).unwrap();
    /// assert_eq!(x, BigFloat::from(12));
    /// let (x, o) = BigFloat::from_str_radix("1e999999999999999999999", 10, 53, Round::Down).unwrap();
    /// assert_eq!((x.is_infinite(), o), (false, Ordering::Less));
    /// assert!(BigFloat::from_str_radix("1.2.3", 10, 53, Round::Nearest).is_err());
    /// ```
    #[track_caller]
    pub fn from_str_radix(
        src: &str,
        radix: u32,
        prec: u64,
        round: Round,
    ) -> Result<(BigFloat, Ordering), ParseBigFloatError> {
        assert!(
            radix == 0 || (2..=36).contains(&radix),
            "radix {radix} is neither 0 nor in 2..=36"
        );
        let prec = BigFloat::check_prec(prec);
        Ok(match parse(src, radix)? {
            Parsed::Nan => (BigFloat::nan(prec), Ordering::Equal),
            Parsed::Inf(negative) => (
                BigFloat {
                    prec,
                    kind: Kind::Inf { negative },
                },
                Ordering::Equal,
            ),
            Parsed::Number {
                negative,
                radix,
                digits,
                exp,
                bexp,
                ..
            } => {
                let d = digits.get();
                if d.is_empty() {
                    (
                        BigFloat {
                            prec,
                            kind: Kind::Zero { negative },
                        },
                        Ordering::Equal,
                    )
                } else {
                    value(negative, d, radix, exp, bexp, prec, round)
                }
            }
        })
    }
}

/// The precision at which [`FromStr`] reads `n` significant decimal digits: 53 for up to 17,
/// those of an `f64` written to be read back, and `ceil(n·log2(10))` beyond.
fn decimal_prec(n: u64) -> u64 {
    if n <= 17 {
        return 53;
    }
    let (lo, hi) = scale::log2_pow(10, u128::from(n));
    // `n·log2(10)` is not an integer: its ceiling is one more than its floor.
    if lo == hi {
        return lo as u64 + 1;
    }
    // `10^n` has `ceil(n·log2(10))` bits, `10^n` not being a power of two.
    (limbs::bit_len(&scale::pow_exact(5, n)) + n).min(BigFloat::MAX_PREC)
}

impl FromStr for BigFloat {
    type Err = ParseBigFloatError;

    /// Reads a number as [`BigFloat::from_str_radix`] in radix 0 (decimal, or hexadecimal after
    /// `0x` and binary after `0b`), rounded to nearest, at a precision the text gives: 53 bits, an
    /// `f64`'s, for up to 17 significant decimal digits (those of any `f64` written to be read
    /// back, so that `"0.1"` is the `f64` 0.1), and `ceil(n·log2(10))` bits for `n` more, which
    /// keeps them; for hexadecimal and binary digits, 53 bits or their bits if more, which holds
    /// them exactly. The significant digits run from the first that is not zero to the last,
    /// zeros included. Any other precision or mode is [`from_str_radix`](Self::from_str_radix)'s.
    ///
    /// # Errors
    ///
    /// As [`from_str_radix`](Self::from_str_radix).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// let x: BigFloat = "0.1".parse().unwrap();
    /// assert_eq!((x.prec(), x), (53, BigFloat::from(0.1)));
    /// let pi: BigFloat = "3.14159265358979323846264338327950288".parse().unwrap();
    /// assert_eq!(pi.prec(), 120);
    /// let x: BigFloat = "-0x1.8p-1".parse().unwrap();
    /// assert_eq!(x, BigFloat::from(-0.75));
    /// assert!("inf".parse::<BigFloat>().unwrap().is_infinite());
    /// ```
    fn from_str(src: &str) -> Result<BigFloat, ParseBigFloatError> {
        let parsed = parse(src, 0)?;
        let prec = match &parsed {
            Parsed::Number {
                radix: 10,
                significant,
                ..
            } => decimal_prec(*significant),
            Parsed::Number {
                radix, significant, ..
            } => (significant * u64::from(radix.trailing_zeros())).clamp(53, BigFloat::MAX_PREC),
            _ => 53,
        };
        Ok(match parsed {
            Parsed::Nan => BigFloat::nan(prec),
            Parsed::Inf(negative) => BigFloat {
                prec,
                kind: Kind::Inf { negative },
            },
            Parsed::Number {
                negative,
                radix,
                digits,
                exp,
                bexp,
                ..
            } => {
                let d = digits.get();
                if d.is_empty() {
                    BigFloat {
                        prec,
                        kind: Kind::Zero { negative },
                    }
                } else {
                    value(negative, d, radix, exp, bexp, prec, Round::Nearest).0
                }
            }
        })
    }
}
