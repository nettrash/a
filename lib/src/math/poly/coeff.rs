//! The types a polynomial can have as coefficients.

use crate::math::complex::Complex;
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::fmt;

/// What the algorithms on polynomials need of their coefficients, sealed.
pub(crate) mod sealed {
    use super::*;

    /// The ring operations, by reference so that a [`Number`] need not be copied, and the
    /// kernels the polynomials' arithmetic calls, which a type may take its own way.
    pub trait Coefficient: Clone + PartialEq + fmt::Debug {
        /// 0.
        fn zero() -> Self;
        /// 1.
        fn one() -> Self;
        /// Whether it is 0.
        fn is_zero(&self) -> bool;
        /// Whether it is 1.
        fn is_one(&self) -> bool;
        /// The integer `n`.
        fn from_u64(n: u64) -> Self;
        /// `a + b`.
        fn add(a: &Self, b: &Self) -> Self;
        /// `a - b`.
        fn sub(a: &Self, b: &Self) -> Self;
        /// `a · b`.
        fn mul(a: &Self, b: &Self) -> Self;
        /// `-a`.
        fn neg(a: &Self) -> Self;
        /// `*a += b`.
        fn add_assign(a: &mut Self, b: &Self);
        /// `*a -= b`.
        fn sub_assign(a: &mut Self, b: &Self);
        /// `*a *= b`.
        fn mul_assign(a: &mut Self, b: &Self);
        /// `*c += a · b`.
        fn mul_add(c: &mut Self, a: &Self, b: &Self) {
            if !a.is_zero() && !b.is_zero() {
                Self::add_assign(c, &Self::mul(a, b));
            }
        }
        /// `*c -= a · b`.
        fn mul_sub(c: &mut Self, a: &Self, b: &Self) {
            if !a.is_zero() && !b.is_zero() {
                Self::sub_assign(c, &Self::mul(a, b));
            }
        }
        /// The product of the polynomials of the coefficients `a` and `b`, the constant first, both
        /// not empty: `a.len() + b.len() - 1` coefficients, the top one possibly 0 (for a ring
        /// with zero divisors).
        fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
            schoolbook(a, b)
        }
        /// The square of the polynomial of the coefficients `a`, not empty.
        fn poly_sqr(a: &[Self]) -> Vec<Self> {
            Self::poly_mul(a, a)
        }
        /// Whether the coefficient is written with a minus sign in front, which the text of a
        /// polynomial takes for the sign of its term.
        fn is_negative(&self) -> bool;
        /// Writes the coefficient, as `Display` writes it; with `abs`, its absolute value (for a
        /// coefficient with [`is_negative`](Coefficient::is_negative)); a complex number, a sum,
        /// in parentheses.
        fn write(&self, f: &mut fmt::Formatter<'_>, abs: bool) -> fmt::Result;
    }

    /// The division of a field.
    pub trait Field: Coefficient {
        /// `1 / a`.
        ///
        /// # Panics
        ///
        /// Panics if `a` is 0, for an exact type; a float gives an infinity.
        fn inv(a: &Self) -> Self;
        /// `a / b`.
        fn div(a: &Self, b: &Self) -> Self {
            Self::mul(a, &Self::inv(b))
        }
    }

    /// The schoolbook product of `a` and `b`, both not empty.
    pub(crate) fn schoolbook<T: Coefficient>(a: &[T], b: &[T]) -> Vec<T> {
        let mut c = vec![T::zero(); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            for (c, y) in c[i..].iter_mut().zip(b) {
                T::mul_add(c, x, y);
            }
        }
        c
    }
}

/// A type a [`Polynomial`](super::Polynomial) can have as coefficients: [`Number`], [`Rational`],
/// `f64` and [`Complex<f64>`]. The trait is sealed. (Polynomials modulo an integer are
/// [`ModPolynomial`](super::ModPolynomial)s.)
pub trait Coefficient: sealed::Coefficient {}

/// A coefficient type that is a field, whose polynomials divide with a remainder and have
/// greatest common divisors by Euclid's algorithm: [`Rational`], `f64` and [`Complex<f64>`]. The
/// trait is sealed.
pub trait Field: Coefficient + sealed::Field {}

impl Coefficient for Number {}
impl Coefficient for Rational {}
impl Coefficient for f64 {}
impl Coefficient for Complex<f64> {}
impl Field for Rational {}
impl Field for f64 {}
impl Field for Complex<f64> {}

/// The ring operations of a type of exact numbers by its operators on references.
macro_rules! exact {
    () => {
        #[inline]
        fn is_zero(&self) -> bool {
            Self::is_zero(self)
        }
        #[inline]
        fn add(a: &Self, b: &Self) -> Self {
            a + b
        }
        #[inline]
        fn sub(a: &Self, b: &Self) -> Self {
            a - b
        }
        #[inline]
        fn mul(a: &Self, b: &Self) -> Self {
            a * b
        }
        #[inline]
        fn neg(a: &Self) -> Self {
            -a
        }
        #[inline]
        fn add_assign(a: &mut Self, b: &Self) {
            *a += b;
        }
        #[inline]
        fn sub_assign(a: &mut Self, b: &Self) {
            *a -= b;
        }
        #[inline]
        fn mul_assign(a: &mut Self, b: &Self) {
            *a *= b;
        }
        #[inline]
        fn is_negative(&self) -> bool {
            Self::is_negative(self)
        }
        fn write(&self, f: &mut fmt::Formatter<'_>, abs: bool) -> fmt::Result {
            if abs && Self::is_negative(self) {
                fmt::Display::fmt(&self.abs(), f)
            } else {
                fmt::Display::fmt(self, f)
            }
        }
    };
}

impl sealed::Coefficient for Number {
    #[inline]
    fn zero() -> Self {
        Number::default()
    }
    #[inline]
    fn one() -> Self {
        Number::from(1u8)
    }
    #[inline]
    fn is_one(&self) -> bool {
        crate::math::rational::is_one(self)
    }
    #[inline]
    fn from_u64(n: u64) -> Self {
        Number::from(n)
    }
    exact!();
}

impl sealed::Coefficient for Rational {
    #[inline]
    fn zero() -> Self {
        Rational::default()
    }
    #[inline]
    fn one() -> Self {
        Rational::from(1u8)
    }
    #[inline]
    fn is_one(&self) -> bool {
        crate::math::rational::is_one(self.numer()) && crate::math::rational::is_one(self.denom())
    }
    #[inline]
    fn from_u64(n: u64) -> Self {
        Rational::from(n)
    }
    exact!();
}

impl sealed::Field for Rational {
    #[inline]
    #[track_caller]
    fn inv(a: &Self) -> Self {
        a.recip()
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        a / b
    }
}

impl sealed::Coefficient for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
    #[inline]
    fn from_u64(n: u64) -> Self {
        n as f64
    }
    #[inline]
    fn add(a: &Self, b: &Self) -> Self {
        a + b
    }
    #[inline]
    fn sub(a: &Self, b: &Self) -> Self {
        a - b
    }
    #[inline]
    fn mul(a: &Self, b: &Self) -> Self {
        a * b
    }
    #[inline]
    fn neg(a: &Self) -> Self {
        -a
    }
    #[inline]
    fn add_assign(a: &mut Self, b: &Self) {
        *a += b;
    }
    #[inline]
    fn sub_assign(a: &mut Self, b: &Self) {
        *a -= b;
    }
    #[inline]
    fn mul_assign(a: &mut Self, b: &Self) {
        *a *= b;
    }
    #[inline]
    fn mul_add(c: &mut Self, a: &Self, b: &Self) {
        *c += a * b;
    }
    #[inline]
    fn mul_sub(c: &mut Self, a: &Self, b: &Self) {
        *c -= a * b;
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.is_sign_negative() && *self != 0.0
    }
    fn write(&self, f: &mut fmt::Formatter<'_>, abs: bool) -> fmt::Result {
        let x = if abs { self.abs() } else { *self };
        fmt::Display::fmt(&x, f)
    }
}

impl sealed::Field for f64 {
    #[inline]
    fn inv(a: &Self) -> Self {
        1.0 / a
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        a / b
    }
}

impl sealed::Coefficient for Complex<f64> {
    #[inline]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.re == 1.0 && self.im == 0.0
    }
    #[inline]
    fn from_u64(n: u64) -> Self {
        Complex::new(n as f64, 0.0)
    }
    #[inline]
    fn add(a: &Self, b: &Self) -> Self {
        *a + *b
    }
    #[inline]
    fn sub(a: &Self, b: &Self) -> Self {
        *a - *b
    }
    #[inline]
    fn mul(a: &Self, b: &Self) -> Self {
        *a * *b
    }
    #[inline]
    fn neg(a: &Self) -> Self {
        -*a
    }
    #[inline]
    fn add_assign(a: &mut Self, b: &Self) {
        *a += *b;
    }
    #[inline]
    fn sub_assign(a: &mut Self, b: &Self) {
        *a -= *b;
    }
    #[inline]
    fn mul_assign(a: &mut Self, b: &Self) {
        *a *= *b;
    }
    #[inline]
    fn mul_add(c: &mut Self, a: &Self, b: &Self) {
        *c += *a * *b;
    }
    #[inline]
    fn mul_sub(c: &mut Self, a: &Self, b: &Self) {
        *c -= *a * *b;
    }
    #[inline]
    fn is_negative(&self) -> bool {
        false
    }
    fn write(&self, f: &mut fmt::Formatter<'_>, _abs: bool) -> fmt::Result {
        f.write_str("(")?;
        fmt::Display::fmt(self, f)?;
        f.write_str(")")
    }
}

impl sealed::Field for Complex<f64> {
    #[inline]
    fn inv(a: &Self) -> Self {
        a.recip()
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        *a / *b
    }
}
