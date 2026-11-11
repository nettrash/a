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
//! arithmetic is done in registers, and the result written into the vectors of the operand that
//! is assigned to or taken by value.
//!
//! [`Rational::floor`], [`Rational::ceil`], [`Rational::round`], [`Rational::round_ties_even`] and
//! [`Rational::trunc`] give integers as [`Number`]s, and [`Rational::fract`] what `trunc` leaves;
//! [`Rational::recip`], [`Rational::abs`], [`Rational::signum`] and [`Rational::pow`] the rest of
//! the arithmetic.
//!
//! Each part of a rational is in the byte order of its numerator: the constructors take it from
//! the numerator they are given, and the result of an operator is in the byte order of its left
//! operand, as for [`Number`]s.

mod arith;
mod cmp;
mod iter;
mod round;
mod small;
mod text;

use crate::math::numbers::{ByteOrder, Number};

/// The rational number `numer/denom`, in lowest terms with a positive denominator.
///
/// Zero is `0/1`, and an integer `n` is `n/1`. The parts are kept private, so that the form
/// holds: [`Rational::new`] reduces any fraction to it, and [`numer`](Rational::numer),
/// [`denom`](Rational::denom) and [`into_parts`](Rational::into_parts) give the parts back.
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
#[derive(Clone)]
pub struct Rational {
    /// The numerator, of the sign of the number, in the shortest form.
    num: Number,
    /// The denominator: positive, prime to the numerator, in the shortest form and the byte order
    /// of the numerator.
    den: Number,
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
        if denom.is_zero() {
            panic!("the denominator of a rational is zero");
        }
        let order = numer.byte_order();
        let mut r = Rational {
            num: numer,
            den: in_order(denom, order),
        };
        r.normalize();
        r
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
        let mut num = n;
        trim(&mut num);
        Rational {
            num,
            den: one(order),
        }
    }

    /// The numerator, of the sign of the number.
    #[inline]
    pub fn numer(&self) -> &Number {
        &self.num
    }

    /// The denominator, positive and prime to the numerator.
    #[inline]
    pub fn denom(&self) -> &Number {
        &self.den
    }

    /// The numerator and the denominator, taken out of the rational.
    #[inline]
    pub fn into_parts(self) -> (Number, Number) {
        (self.num, self.den)
    }

    /// The byte order of both parts.
    #[inline]
    pub fn byte_order(&self) -> ByteOrder {
        self.num.byte_order()
    }

    /// Whether the number is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// Whether the number is below zero.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.num.is_negative()
    }

    /// Whether the number is above zero.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.num.is_positive()
    }

    /// Whether the number is an integer: whether its denominator is 1.
    #[inline]
    pub fn is_integer(&self) -> bool {
        is_one(&self.den)
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
        let mut r = self.clone();
        r.num.negative = false;
        r
    }

    /// `-1`, `0` or `1`, as the number is below zero, zero or above it, in its byte order.
    pub fn signum(&self) -> Rational {
        Rational::from_integer(self.num.signum())
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
        let r = Rational {
            num: self.num.pow(k),
            den: self.den.pow(k),
        };
        if exp < 0 { r.into_recip() } else { r }
    }

    /// Turns the number into its reciprocal in place, the sign moved to the new numerator.
    #[track_caller]
    pub(crate) fn invert(&mut self) {
        if self.num.is_zero() {
            panic!("attempt to divide by zero");
        }
        let negative = self.num.negative;
        core::mem::swap(&mut self.num, &mut self.den);
        self.num.negative = negative;
        self.den.negative = false;
    }

    /// Brings a fraction with a non-zero denominator, both parts in one byte order, to lowest terms
    /// with a positive denominator.
    fn normalize(&mut self) {
        trim(&mut self.num);
        trim(&mut self.den);
        if self.den.negative {
            self.den.negative = false;
            self.num.negative = !self.num.negative && !self.num.is_zero();
        }
        if self.num.is_zero() {
            self.num.negative = false;
            self.den = one(self.num.byte_order());
            return;
        }
        if is_one(&self.den) {
            return;
        }
        let g = self.num.gcd(&self.den);
        if !is_one(&g) {
            self.num = self.num.div_exact(&g);
            self.den = self.den.div_exact(&g);
        }
    }

    /// A rational from parts already in lowest terms, with a positive denominator, both in the
    /// shortest form and in one byte order.
    #[inline]
    pub(crate) fn from_reduced(num: Number, den: Number) -> Rational {
        debug_assert!(!den.is_zero() && !den.is_negative());
        debug_assert!(num.byte_order() == den.byte_order());
        Rational { num, den }
    }
}

impl Default for Rational {
    /// Zero, `0/1`, little-endian.
    fn default() -> Rational {
        Rational {
            num: Number::default(),
            den: one(ByteOrder::LittleEndian),
        }
    }
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
