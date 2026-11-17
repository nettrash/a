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
        /// The value of the polynomial of the coefficients `c` at the point `x`, by Horner's rule.
        fn poly_eval(c: &[Self], x: &Self) -> Self {
            let mut it = c.iter().rev();
            let Some(top) = it.next() else {
                return Self::zero();
            };
            let mut v = top.clone();
            for a in it {
                Self::mul_assign(&mut v, x);
                Self::add_assign(&mut v, a);
            }
            v
        }
        /// The values of the polynomial of the coefficients `c` at the points `xs`.
        fn poly_eval_many(c: &[Self], xs: &[Self]) -> Vec<Self> {
            crate::math::poly::eval::tree::horner_many(c, xs)
        }
        /// The coefficients of `Π (x - rᵢ)` for the roots `r`.
        fn poly_from_roots(r: &[Self]) -> Vec<Self> {
            crate::math::poly::eval::tree::from_roots(r, Self::poly_mul)
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
        /// The coefficients of the interpolating polynomial of the points `xs`, distinct, and the
        /// values `ys`, as many, possibly with zeros at the top.
        fn poly_interpolate(xs: &[Self], ys: &[Self]) -> Vec<Self> {
            crate::math::poly::eval::newton(xs, ys)
        }
        /// The quotient and the remainder of `a` by `d` (not zero, `inv` the inverse of its
        /// leading coefficient) by a method faster than the schoolbook one for these lengths, or
        /// `None` to leave them to it.
        /// `b^e mod f` (`f` not zero) by a method faster than the generic one for this field, or
        /// `None` to leave it to it.
        fn poly_pow_mod(
            b: &super::super::Polynomial<Self>,
            e: &crate::math::numbers::Number,
            f: &super::super::Polynomial<Self>,
        ) -> Option<super::super::Polynomial<Self>> {
            let _ = (b, e, f);
            None
        }
        /// The inverse of the power series `a` (its constant not zero) modulo `x^n` by a method
        /// faster than the generic one for this field and length, or `None` to leave it to it.
        fn poly_inv_series(
            a: &super::super::Polynomial<Self>,
            n: usize,
        ) -> Option<super::super::Polynomial<Self>> {
            let _ = (a, n);
            None
        }
        fn poly_div_rem(
            a: &super::super::Polynomial<Self>,
            d: &super::super::Polynomial<Self>,
            inv: &Self,
        ) -> Option<(
            super::super::Polynomial<Self>,
            super::super::Polynomial<Self>,
        )> {
            let _ = (a, d, inv);
            None
        }
        /// The monic greatest common divisor of the polynomials of the coefficients `a` and `b`.
        fn poly_gcd(a: &[Self], b: &[Self]) -> Vec<Self>;
        /// `(g, s, t)` of [`Polynomial::xgcd`](crate::math::poly::Polynomial::xgcd) on vectors.
        #[allow(clippy::type_complexity)]
        fn poly_xgcd(a: &[Self], b: &[Self]) -> (Vec<Self>, Vec<Self>, Vec<Self>);
        /// The resultant of the polynomials of the coefficients `a` and `b`.
        fn poly_resultant(a: &[Self], b: &[Self]) -> Self;
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
    fn poly_eval(c: &[Self], x: &Self) -> Self {
        // From a few coefficients, the evaluation on limbs (word blocks, divide and conquer past
        // n·bits(x) of a few thousand bits) instead of Horner's rule on numbers, whose every step
        // builds a number as long as the value.
        if c.len() <= super::arith::EVAL_HORNER {
            let mut it = c.iter().rev();
            let Some(top) = it.next() else {
                return Number::default();
            };
            let mut v = top.clone();
            for a in it {
                <Number as sealed::Coefficient>::mul_assign(&mut v, x);
                <Number as sealed::Coefficient>::add_assign(&mut v, a);
            }
            return v;
        }
        crate::math::poly::eval::number_eval_many(c, core::slice::from_ref(x))
            .pop()
            .expect("one point")
    }
    fn poly_eval_many(c: &[Self], xs: &[Self]) -> Vec<Self> {
        crate::math::poly::eval::number_eval_many(c, xs)
    }
    fn poly_from_roots(r: &[Self]) -> Vec<Self> {
        crate::math::poly::eval::number_from_roots(r)
    }
    fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::arith::zmul::mul(a, b)
    }
    fn poly_sqr(a: &[Self]) -> Vec<Self> {
        super::arith::zmul::sqr(a)
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
    fn poly_eval_many(c: &[Self], xs: &[Self]) -> Vec<Self> {
        crate::math::poly::eval::rational_eval_many(c, xs)
    }
    fn poly_from_roots(r: &[Self]) -> Vec<Self> {
        crate::math::poly::eval::rational_from_roots(r)
    }
    fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::arith::field::rat_mul(a, b)
    }
    fn poly_sqr(a: &[Self]) -> Vec<Self> {
        super::arith::field::rat_sqr(a)
    }
    exact!();
}

impl sealed::Field for Rational {
    fn poly_pow_mod(
        b: &super::Polynomial<Self>,
        e: &Number,
        f: &super::Polynomial<Self>,
    ) -> Option<super::Polynomial<Self>> {
        super::arith::field::rat_pow_mod(b.coeffs(), e, f.coeffs()).map(super::Polynomial::new)
    }
    fn poly_inv_series(a: &super::Polynomial<Self>, n: usize) -> Option<super::Polynomial<Self>> {
        super::arith::field::rat_inv_series(a.coeffs(), n).map(super::Polynomial::new)
    }
    fn poly_div_rem(
        a: &super::Polynomial<Self>,
        d: &super::Polynomial<Self>,
        _inv: &Self,
    ) -> Option<(super::Polynomial<Self>, super::Polynomial<Self>)> {
        super::arith::field::rat_div_rem(a, d)
    }
    #[inline]
    #[track_caller]
    fn inv(a: &Self) -> Self {
        a.recip()
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        a / b
    }
    #[track_caller]
    fn poly_interpolate(xs: &[Self], ys: &[Self]) -> Vec<Self> {
        crate::math::poly::eval::rational_interpolate(xs, ys)
    }
    fn poly_gcd(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::gcd::rational_gcd(a, b)
    }
    fn poly_xgcd(a: &[Self], b: &[Self]) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
        super::gcd::rational_xgcd(a, b)
    }
    fn poly_resultant(a: &[Self], b: &[Self]) -> Self {
        super::gcd::rational_resultant(a, b)
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
    fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::arith::field::f64_mul(a, b)
    }
    fn poly_sqr(a: &[Self]) -> Vec<Self> {
        super::arith::field::f64_sqr(a)
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
    fn poly_div_rem(
        a: &super::Polynomial<Self>,
        d: &super::Polynomial<Self>,
        inv: &Self,
    ) -> Option<(super::Polynomial<Self>, super::Polynomial<Self>)> {
        super::arith::field_div_rem(a, d, inv)
    }
    #[inline]
    fn inv(a: &Self) -> Self {
        1.0 / a
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        a / b
    }
    fn poly_gcd(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::gcd::field_gcd(a, b)
    }
    fn poly_xgcd(a: &[Self], b: &[Self]) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
        super::gcd::field_xgcd(a, b)
    }
    fn poly_resultant(a: &[Self], b: &[Self]) -> Self {
        super::gcd::field_resultant(a, b)
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
    fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::arith::field::complex_mul(a, b)
    }
    fn poly_sqr(a: &[Self]) -> Vec<Self> {
        super::arith::field::complex_sqr(a)
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
    fn poly_div_rem(
        a: &super::Polynomial<Self>,
        d: &super::Polynomial<Self>,
        inv: &Self,
    ) -> Option<(super::Polynomial<Self>, super::Polynomial<Self>)> {
        super::arith::field_div_rem(a, d, inv)
    }
    #[inline]
    fn inv(a: &Self) -> Self {
        a.recip()
    }
    #[inline]
    fn div(a: &Self, b: &Self) -> Self {
        *a / *b
    }
    fn poly_gcd(a: &[Self], b: &[Self]) -> Vec<Self> {
        super::gcd::field_gcd(a, b)
    }
    fn poly_xgcd(a: &[Self], b: &[Self]) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
        super::gcd::field_xgcd(a, b)
    }
    fn poly_resultant(a: &[Self], b: &[Self]) -> Self {
        super::gcd::field_resultant(a, b)
    }
}
