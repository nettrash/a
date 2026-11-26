//! Polynomials in one variable: a [`Polynomial<T>`] has coefficients of the type `T`, which is
//! [`Number`](crate::math::numbers::Number), [`Rational`](crate::math::rational::Rational), `f64`
//! or [`Complex<f64>`](crate::math::complex::Complex); a [`ModPolynomial`] has coefficients
//! modulo an integer, which it holds once.
//!
//! Both are dense: a vector of the coefficients, the constant first, without zeros at the top, so
//! that the zero polynomial has none and every other its leading coefficient last.

mod arith;
mod basic;
mod coeff;
mod div;
mod eval;
mod factor;
mod gcd;
mod modular;
mod ops;
mod roots;
mod text;

pub use coeff::{Coefficient, Field};
pub use modular::ModPolynomial;
pub use roots::ComplexRoot;

/// A polynomial in one variable with coefficients of the type `T`, held densely: the
/// coefficients, the constant first, without zeros at the top.
///
/// # Examples
///
/// ```
/// use a::math::poly::Polynomial;
/// use a::math::numbers::Number;
///
/// // (x + 1)(x - 1) = x² - 1
/// let p = Polynomial::new(vec![Number::from(1), Number::from(1)]);
/// let q = Polynomial::new(vec![Number::from(-1), Number::from(1)]);
/// let r = &p * &q;
/// assert_eq!(r.degree(), Some(2));
/// assert_eq!(r.to_string(), "x^2 - 1");
/// assert_eq!(r.eval(&Number::from(3)), Number::from(8));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Polynomial<T> {
    /// The coefficients, the constant first, without zeros at the top.
    c: Vec<T>,
}
