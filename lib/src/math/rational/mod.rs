//! Rational numbers: a [`Rational`] is a fraction `p/q` of two [`Number`]s, always in lowest
//! terms with a positive denominator, so that every rational number has one form and two
//! rationals are equal when their parts are.
//!
//! They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide
//! with `/` and `/=`, each operand owned or borrowed, and take a [`Number`] or a primitive integer
//! as an operand on either side. They negate with `-`, compare by value with `==`, `<` and the
//! rest (with numbers and primitive integers too), hash by value, add up with `Sum`, multiply with
//! `Product`, and are written as `-3/4`, or `5` when they are integers, by `Display`.
//!
//! The sums and products reduce by the greatest common divisors of the parts as Knuth's
//! *The Art of Computer Programming*, volume 2, section 4.5.1, has it, so that the numbers they
//! multiply stay as short as they can be: `a/b + c/d` divides `b` and `d` by `g = gcd(b, d)`
//! first, and its numerator then only by the divisors of `g`; `(a/b)·(c/d)` divides `a` by
//! `gcd(a, d)` and `c` by `gcd(c, b)` before it multiplies. When every part fits in 64 bits the
//! arithmetic is done in registers, and otherwise on limbs, the parts read once and the result
//! written once, into the vectors of the operand that is assigned to or taken by value. `Sum` and
//! `Product` combine their terms in a balanced tree.
//!
//! [`Rational::floor`], [`Rational::ceil`], [`Rational::round`], [`Rational::round_ties_even`] and
//! [`Rational::trunc`] give integers as [`Number`]s, and [`Rational::fract`] what `trunc` leaves;
//! [`Rational::recip`], [`Rational::abs`], [`Rational::signum`] and [`Rational::pow`] the rest of
//! the arithmetic.
//!
//! A finite `f64` or `f32` converts to a rational exactly with `TryFrom`, and a rational to the
//! nearest float with [`Rational::to_f64`] and [`Rational::to_f32`], correctly rounded, in constant
//! time but for one fraction in 700; an integer converts to a [`Number`] or a primitive integer
//! with `TryFrom`. [`FromStr`](core::str::FromStr) reads `-3/4`, `5` and decimals such as
//! `-1.5e-3`, exactly, and [`Rational::from_str_radix`] and [`Rational::to_str_radix`] read and
//! write fractions in the radices from 2 to 36.
//!
//! Each part of a rational is in the byte order of its numerator: the constructors take it from
//! the numerator they are given, and the result of an operator is in the byte order of its left
//! operand, as for [`Number`]s.

mod approx;
mod arith;
mod cmp;
mod convert;
mod iter;
mod round;
mod sequences;
mod small;
mod text;

pub use convert::{TryFromFloatError, TryFromRationalError};
pub use text::{ParseRationalError, RationalErrorKind};

use crate::math::numbers::{ByteOrder, Magnitude, Number, fraction};
use small::Small;
use std::sync::OnceLock;

/// The rational number `numer/denom`, in lowest terms with a positive denominator.
///
/// Zero is `0/1`, and an integer `n` is `n/1`. The parts are kept private, so that the form
/// holds: [`Rational::new`] reduces any fraction to it, and [`numer`](Rational::numer),
/// [`denom`](Rational::denom) and [`into_parts`](Rational::into_parts) give the parts back.
///
/// A rational whose parts both fit in 64 bits holds them in two words, with no allocation, and
/// makes the [`Number`]s that `numer` and `denom` return the first time they are asked for; a
/// longer one holds two `Number`s.
///
/// # Examples
///
/// ```
/// use a::math::numbers::Number;
/// use a::math::rational::Rational;
///
/// let a = Rational::new(Number::from(6), Number::from(-8));
/// assert_eq!(a.numer(), &Number::from(-3));
/// assert_eq!(a.denom(), &Number::from(4));
///
/// let b = Rational::from(1) / 3u32;
/// assert_eq!(&a + &b, Rational::new(Number::from(-5), Number::from(12)));
/// assert_eq!((a * b).to_string(), "-1/4");
/// ```
pub struct Rational {
    repr: Repr,
}

/// The two forms of a rational: a value whose parts both fit in 64 bits is always in words, and
/// any other in numbers, so that each value has one form whatever made it.
enum Repr {
    /// Both parts of at most 64 bits.
    Word(Word),
    /// A part of more than 64 bits.
    Big(Big),
}

/// A rational in numbers: at least one part of more than 64 bits, both in the shortest form and
/// in one byte order.
struct Big {
    /// The numerator, of the sign of the number.
    num: Number,
    /// The denominator: positive and prime to the numerator.
    den: Number,
}

/// A rational in words.
struct Word {
    /// The magnitude of the numerator.
    num: u64,
    /// The denominator: positive and prime to the numerator; 1 for zero.
    den: u64,
    /// Whether the number is below zero; never for zero.
    negative: bool,
    /// The byte order of the parts as numbers.
    order: ByteOrder,
    /// The parts as numbers, made the first time they are asked for.
    numbers: OnceLock<Box<(Number, Number)>>,
}

impl Word {
    /// The rational `±num/den`, with nothing made yet.
    #[inline(always)]
    fn new(negative: bool, num: u64, den: u64, order: ByteOrder) -> Word {
        debug_assert!(den != 0 && (num != 0 || den == 1 && !negative));
        Word {
            num,
            den,
            negative,
            order,
            numbers: OnceLock::new(),
        }
    }

    /// The parts as numbers.
    #[inline]
    fn numbers(&self) -> &(Number, Number) {
        self.numbers.get_or_init(|| Box::new(self.make_numbers()))
    }

    /// The parts as new numbers.
    #[cold]
    #[inline(never)]
    fn make_numbers(&self) -> (Number, Number) {
        (
            small::number(self.order, self.negative, u128::from(self.num)),
            small::number(self.order, false, u128::from(self.den)),
        )
    }
}

impl Clone for Rational {
    /// A copy: the words of a short rational, without the numbers made for it, or the two numbers
    /// of a long one.
    #[inline]
    fn clone(&self) -> Rational {
        Rational {
            repr: match &self.repr {
                Repr::Word(w) => Repr::Word(Word::new(w.negative, w.num, w.den, w.order)),
                Repr::Big(b) => Repr::Big(Big {
                    num: b.num.clone(),
                    den: b.den.clone(),
                }),
            },
        }
    }
}

impl Rational {
    /// The fraction `numer/denom` in lowest terms, with the sign on the numerator and both parts in
    /// the byte order of `numer`.
    ///
    /// Divides both parts by their greatest common divisor; takes `O(n²)` time for parts of `n`
    /// limbs at most, less for the longest (the gcd's half gcd), and reuses the vectors of the
    /// parts where the results fit.
    ///
    /// # Panics
    ///
    /// When `denom` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::new(Number::from(-10), Number::from(-4));
    /// assert_eq!(r.to_string(), "5/2");
    /// assert_eq!(Rational::new(Number::from(0), Number::from(-7)).to_string(), "0");
    /// ```
    #[track_caller]
    pub fn new(numer: Number, denom: Number) -> Rational {
        let order = numer.byte_order();
        if numer.value.len() <= 8 && denom.value.len() <= 8 {
            // Both in words, read straight from their vectors, each in its own byte order.
            let (x, y) = (small::short_vector(&numer), small::short_vector(&denom));
            if y == 0 {
                zero_denominator();
            }
            if x == 0 {
                return Rational::from_word(false, 0, 1, order);
            }
            let g = small::gcd(x, y);
            return Rational::from_word(numer.negative != denom.negative, x / g, y / g, order);
        }
        if denom.is_zero() {
            zero_denominator();
        }
        normalize(numer, in_order(denom, order))
    }

    /// The integer `n` as the rational `n/1`, in the byte order of `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from_integer(Number::from(-7));
    /// assert!(r.is_integer());
    /// assert_eq!(r.to_string(), "-7");
    /// ```
    pub fn from_integer(n: Number) -> Rational {
        let order = n.byte_order();
        if n.value.len() <= 8 {
            let m = small::short_vector(&n);
            return Rational::from_word(n.negative && m != 0, m, 1, order);
        }
        let m = Magnitude::of(&n);
        if m.bytes.len() <= 8 {
            return Rational::from_word(m.negative, m.small() as u64, 1, order);
        }
        let mut num = n;
        trim(&mut num);
        Rational::big(num, one(order))
    }

    /// The numerator, of the sign of the number.
    #[inline]
    pub fn numer(&self) -> &Number {
        match &self.repr {
            Repr::Word(w) => &w.numbers().0,
            Repr::Big(b) => &b.num,
        }
    }

    /// The denominator, positive and prime to the numerator.
    #[inline]
    pub fn denom(&self) -> &Number {
        match &self.repr {
            Repr::Word(w) => &w.numbers().1,
            Repr::Big(b) => &b.den,
        }
    }

    /// The numerator and the denominator, taken out of the rational.
    #[inline]
    pub fn into_parts(self) -> (Number, Number) {
        match self.repr {
            Repr::Word(w) => match w.numbers.into_inner() {
                Some(parts) => *parts,
                None => Word::new(w.negative, w.num, w.den, w.order).make_numbers(),
            },
            Repr::Big(b) => (b.num, b.den),
        }
    }

    /// The byte order of both parts.
    #[inline]
    pub fn byte_order(&self) -> ByteOrder {
        match &self.repr {
            Repr::Word(w) => w.order,
            Repr::Big(b) => b.num.byte_order(),
        }
    }

    /// Whether the number is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        match &self.repr {
            Repr::Word(w) => w.num == 0,
            Repr::Big(b) => b.num.is_zero(),
        }
    }

    /// Whether the number is below zero.
    #[inline]
    pub fn is_negative(&self) -> bool {
        match &self.repr {
            Repr::Word(w) => w.negative,
            Repr::Big(b) => b.num.is_negative(),
        }
    }

    /// Whether the number is above zero.
    #[inline]
    pub fn is_positive(&self) -> bool {
        match &self.repr {
            Repr::Word(w) => !w.negative && w.num != 0,
            Repr::Big(b) => b.num.is_positive(),
        }
    }

    /// Whether the number is an integer: whether its denominator is 1.
    #[inline]
    pub fn is_integer(&self) -> bool {
        match &self.repr {
            Repr::Word(w) => w.den == 1,
            Repr::Big(b) => is_one(&b.den),
        }
    }

    /// The absolute value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(-3) / 4u32).abs().to_string(), "3/4");
    /// ```
    pub fn abs(&self) -> Rational {
        match &self.repr {
            Repr::Word(w) => Rational::from_word(false, w.num, w.den, w.order),
            Repr::Big(b) => {
                let mut num = b.num.clone();
                num.negative = false;
                Rational::big(num, b.den.clone())
            }
        }
    }

    /// `-1`, `0` or `1`, as the number is below zero, zero or above it, in its byte order.
    pub fn signum(&self) -> Rational {
        let order = self.byte_order();
        let m = u64::from(!self.is_zero());
        Rational::from_word(self.is_negative(), m, 1, order)
    }

    /// The reciprocal `1/x`.
    ///
    /// # Panics
    ///
    /// When the number is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(-3) / 4u32;
    /// assert_eq!(r.recip().to_string(), "-4/3");
    /// ```
    #[track_caller]
    pub fn recip(&self) -> Rational {
        if let Repr::Word(w) = &self.repr
            && w.num != 0
        {
            return Rational::from_word(w.negative, w.den, w.num, w.order);
        }
        self.clone().into_recip()
    }

    /// The reciprocal `1/x`, in the vectors of `self`.
    ///
    /// # Panics
    ///
    /// When the number is zero.
    #[track_caller]
    pub fn into_recip(mut self) -> Rational {
        self.invert();
        self
    }

    /// `self` raised to the power `exp`, which may be negative: `x⁻ⁿ = (1/x)ⁿ`, and `x⁰ = 1`, for
    /// `x = 0` too, as for the primitive types. The parts are raised one by one, as a power of a
    /// fraction in lowest terms is in lowest terms.
    ///
    /// # Panics
    ///
    /// When the number is zero and `exp` is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let r = Rational::from(-2) / 3u32;
    /// assert_eq!(r.pow(3).to_string(), "-8/27");
    /// assert_eq!(r.pow(-2).to_string(), "9/4");
    /// assert_eq!(r.pow(0).to_string(), "1");
    /// ```
    #[track_caller]
    pub fn pow(&self, exp: i32) -> Rational {
        let k = exp.unsigned_abs();
        let r = match &self.repr {
            Repr::Word(w) => {
                let negative = w.negative && k % 2 == 1 && w.num != 0;
                // The bits of the parts: a power of `x` of `b` bits lies in `[2^((b-1)k), 2^(bk))`.
                let (bn, bd) = (64 - w.num.leading_zeros(), 64 - w.den.leading_zeros());
                let (k64, short) = (u64::from(k), |b: u32| u64::from(b) * u64::from(k) <= 64);
                if short(bn) && short(bd) {
                    let (n, d) = (w.num.wrapping_pow(k), w.den.wrapping_pow(k));
                    Rational::from_word(negative, n, d, w.order)
                } else {
                    // Straight from the words when the powers take up to 512 bits.
                    let power = |m: u64, negative: bool| {
                        Number::power_of_small(u128::from(m), k, negative, w.order)
                            .unwrap_or_else(|| small::number(w.order, negative, m.into()).pow(k))
                    };
                    let (n, d) = (power(w.num, negative), power(w.den, false));
                    if u64::from(bn.max(bd) - 1) * k64 >= 64 {
                        // A part of at least `2^64`: in numbers.
                        Rational::big(n, d)
                    } else {
                        Rational::from_reduced(n, d)
                    }
                }
            }
            Repr::Big(b) => Rational::from_reduced(b.num.pow(k), b.den.pow(k)),
        };
        if exp < 0 { r.into_recip() } else { r }
    }

    /// Turns the sign of the number over in place; zero stays as it is.
    #[inline]
    pub(crate) fn negate(&mut self) {
        match &mut self.repr {
            Repr::Word(w) => {
                w.negative = !w.negative && w.num != 0;
                if let Some(parts) = w.numbers.get_mut() {
                    parts.0.negative = w.negative;
                }
            }
            Repr::Big(b) => b.num.negative = !b.num.negative,
        }
    }

    /// Turns the number into its reciprocal in place, the sign moved to the new numerator.
    #[track_caller]
    pub(crate) fn invert(&mut self) {
        if self.is_zero() {
            panic!("attempt to divide by zero");
        }
        match &mut self.repr {
            Repr::Word(w) => *w = Word::new(w.negative, w.den, w.num, w.order),
            Repr::Big(b) => {
                let negative = b.num.negative;
                core::mem::swap(&mut b.num, &mut b.den);
                b.num.negative = negative;
                b.den.negative = false;
            }
        }
    }

    /// The numerator as a new number, without making the numbers of a rational in words.
    #[inline]
    pub(crate) fn numer_number(&self) -> Number {
        match &self.repr {
            Repr::Word(w) => small::number(w.order, w.negative, u128::from(w.num)),
            Repr::Big(b) => b.num.clone(),
        }
    }

    /// The parts when they fit in 64 bits: the sign, the numerator's magnitude and the
    /// denominator.
    #[inline(always)]
    pub(crate) fn word(&self) -> Option<Small> {
        match &self.repr {
            Repr::Word(w) => Some(Small {
                negative: w.negative,
                num: w.num,
                den: w.den,
            }),
            Repr::Big(_) => None,
        }
    }

    /// The parts as numbers when one of them does not fit in 64 bits.
    #[inline(always)]
    pub(crate) fn numbers(&self) -> Option<(&Number, &Number)> {
        match &self.repr {
            Repr::Word(_) => None,
            Repr::Big(b) => Some((&b.num, &b.den)),
        }
    }

    /// The parts as numbers, to be written in place, when one of them does not fit in 64 bits;
    /// [`settle`](Rational::settle) must follow when they may have become short.
    #[inline(always)]
    pub(crate) fn numbers_mut(&mut self) -> Option<(&mut Number, &mut Number)> {
        match &mut self.repr {
            Repr::Word(_) => None,
            Repr::Big(b) => Some((&mut b.num, &mut b.den)),
        }
    }

    /// The magnitudes of the parts, the sign on the numerator's: of the words of a short rational
    /// in place, with nothing allocated, or of the numbers of a long one.
    #[inline(always)]
    pub(crate) fn magnitudes(&self) -> (Magnitude<'_>, Magnitude<'_>) {
        match &self.repr {
            Repr::Word(w) => (
                small::magnitude(&w.num, w.negative),
                small::magnitude(&w.den, false),
            ),
            Repr::Big(b) => (Magnitude::of(&b.num), Magnitude::of(&b.den)),
        }
    }

    /// The rational `±num/den` in words: in lowest terms, `den` not zero, and zero as `0/1`, not
    /// negative.
    #[inline(always)]
    pub(crate) fn from_word(negative: bool, num: u64, den: u64, order: ByteOrder) -> Rational {
        Rational {
            repr: Repr::Word(Word::new(negative, num, den, order)),
        }
    }

    /// Sets the number to `±num/den` in words (in lowest terms, `den` not zero, and zero as `0/1`,
    /// not negative): in the words of a rational in words, with no drop but of the numbers made
    /// for the old value, if any.
    #[inline(always)]
    pub(crate) fn set_word(&mut self, negative: bool, num: u64, den: u64, order: ByteOrder) {
        debug_assert!(den != 0 && (num != 0 || den == 1 && !negative));
        match &mut self.repr {
            Repr::Word(w) => {
                w.num = num;
                w.den = den;
                w.negative = negative;
                w.order = order;
                if w.numbers.get().is_some() {
                    w.numbers = OnceLock::new();
                }
            }
            Repr::Big(_) => *self = Rational::from_word(negative, num, den, order),
        }
    }

    /// The rational `±num/den` from parts of up to 128 bits in lowest terms, `den` not zero, and
    /// zero as `0/1`, not negative: in words when both fit in 64 bits.
    #[inline]
    pub(crate) fn from_u128(negative: bool, num: u128, den: u128, order: ByteOrder) -> Rational {
        if (num | den) >> 64 == 0 {
            return Rational::from_word(negative, num as u64, den as u64, order);
        }
        Rational::big(
            small::number(order, negative, num),
            small::number(order, false, den),
        )
    }

    /// A rational in numbers, of parts with one of more than 64 bits.
    #[inline(always)]
    pub(crate) fn big(num: Number, den: Number) -> Rational {
        debug_assert!(num.value.len() > 8 || den.value.len() > 8);
        Rational {
            repr: Repr::Big(Big { num, den }),
        }
    }

    /// A rational from parts already in lowest terms, with a positive denominator, both in the
    /// shortest form and in one byte order: in words when both fit in 64 bits.
    #[inline]
    pub(crate) fn from_reduced(num: Number, den: Number) -> Rational {
        debug_assert!(!den.is_zero() && !den.is_negative());
        debug_assert!(num.byte_order() == den.byte_order());
        let (x, y) = (Magnitude::of(&num), Magnitude::of(&den));
        if x.bytes.len() <= 8 && y.bytes.len() <= 8 {
            return Rational::from_word(
                x.negative,
                x.small() as u64,
                y.small() as u64,
                num.byte_order,
            );
        }
        Rational::big(num, den)
    }

    /// Brings a rational in numbers to words when its parts have become short, after they were
    /// written in place.
    #[inline]
    pub(crate) fn settle(&mut self) {
        if let Repr::Big(b) = &self.repr {
            let (x, y) = (Magnitude::of(&b.num), Magnitude::of(&b.den));
            if x.bytes.len() <= 8 && y.bytes.len() <= 8 {
                *self = Rational::from_word(
                    x.negative,
                    x.small() as u64,
                    y.small() as u64,
                    b.num.byte_order,
                );
            }
        }
    }
}

/// Brings a fraction with a non-zero denominator, both parts in one byte order, to lowest terms
/// with a positive denominator.
///
/// Parts of up to 8 bytes each are reduced in registers by the greatest common divisor of two
/// limbs into words, parts of up to 16 by that of two `u128`s and Hensel's exact division modulo
/// `2^128`; longer ones on limbs (see [`fraction::reduce`]), in the vectors of the parts. A divisor
/// of 1, the common case of random parts, leaves the parts as they are. Measured on an Apple M4
/// Max, `Rational::new` of random 64-bit parts took 260 cycles so against 341 on numbers.
fn normalize(mut num: Number, mut den: Number) -> Rational {
    let order = num.byte_order;
    let (x, y) = (Magnitude::of(&num), Magnitude::of(&den));
    // The sign of the fraction, on the numerator; zero is `0/1`.
    let negative = x.negative != y.negative;
    if x.bytes.is_empty() {
        return Rational::from_word(false, 0, 1, order);
    }
    let (ln, ld) = (x.bytes.len(), y.bytes.len());
    if ln <= 16 && ld <= 16 {
        let (x, y) = (x.small(), y.small());
        if ln <= 8 && ld <= 8 {
            let (x, y) = (x as u64, y as u64);
            let g = small::gcd(x, y);
            return Rational::from_word(negative, x / g, y / g, order);
        }
        let g = fraction::gcd_wide(x, y);
        let (p, q) = if g == 1 { (x, y) } else { exact_pair(x, y, g) };
        if (p | q) >> 64 == 0 {
            return Rational::from_word(negative, p as u64, q as u64, order);
        }
        // Still long: in the vectors given.
        small::put(&mut num, order, negative, p);
        small::put(&mut den, order, false, q);
        return Rational::big(num, den);
    }
    trim(&mut num);
    trim(&mut den);
    num.negative = negative;
    den.negative = false;
    if !is_one(&den) {
        fraction::reduce(&mut num, &mut den);
    }
    Rational::from_reduced(num, den)
}

/// Panics for a zero denominator.
#[cold]
#[inline(never)]
#[track_caller]
fn zero_denominator() -> ! {
    panic!("the denominator of a rational is zero")
}

impl Default for Rational {
    /// Zero, `0/1`, little-endian.
    #[inline]
    fn default() -> Rational {
        Rational::from_word(false, 0, 1, ByteOrder::LittleEndian)
    }
}

/// `x / g` and `y / g` for a `g` that divides both, by Hensel's method: the quotients of `x` and
/// `y` shifted right by the zero bits of `g`, times the inverse of its odd part modulo `2^128`.
#[inline]
fn exact_pair(x: u128, y: u128, g: u128) -> (u128, u128) {
    let z = g.trailing_zeros();
    let odd = g >> z;
    // The inverse modulo `2^64`, then one Newton step for `2^128`.
    let v = u128::from(fraction::inverse_word(odd as u64));
    let v = v.wrapping_mul(2u128.wrapping_sub(odd.wrapping_mul(v)));
    ((x >> z).wrapping_mul(v), (y >> z).wrapping_mul(v))
}

/// 1 in the byte order `order`.
#[inline]
pub(crate) fn one(order: ByteOrder) -> Number {
    Number {
        value: vec![1],
        negative: false,
        byte_order: order,
    }
}

/// Whether `n` is 1.
#[inline]
pub(crate) fn is_one(n: &Number) -> bool {
    !n.negative && n.value.len() == 1 && n.value[0] == 1 || n.value.len() > 1 && *n == 1u8
}

/// `x` in the byte order `order`: its bytes turned around when the orders differ.
#[inline]
pub(crate) fn in_order(x: Number, order: ByteOrder) -> Number {
    if x.byte_order() == order {
        return x;
    }
    let mut value = x.value;
    value.reverse();
    Number {
        value,
        negative: x.negative,
        byte_order: order,
    }
}

/// Trims `n` to the shortest form in place: no zero bytes above the most significant one, zero as
/// `[0]` and never negative.
#[inline]
pub(crate) fn trim(n: &mut Number) {
    let order = n.byte_order();
    let v = &mut n.value;
    let s = match order {
        ByteOrder::BigEndian => v.len() - v.iter().position(|&b| b != 0).unwrap_or(v.len()),
        ByteOrder::LittleEndian => v.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1),
    };
    if s == 0 {
        v.clear();
        v.push(0);
        n.negative = false;
    } else if s != v.len() {
        match order {
            ByteOrder::BigEndian => {
                let len = v.len();
                v.drain(..len - s);
            }
            ByteOrder::LittleEndian => v.truncate(s),
        }
    }
}
