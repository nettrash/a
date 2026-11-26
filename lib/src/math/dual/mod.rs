//! Dual numbers for forward-mode automatic differentiation: [`Dual<T>`] is `re + eps·ε` with
//! `ε² = 0`, so a function evaluated at `x + ε` gives `f(x) + f′(x)·ε`, and [`HyperDual<T>`] is
//! `re + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂` with `ε₁² = ε₂² = 0` (Fike and Alonso, AIAA
//! 2011-886), so `f(x + ε₁ + ε₂)` gives `f″(x)` as well — each exact up to rounding, with no step
//! to choose and nothing to cancel, but for the one rule below at `0·∞`.
//!
//! Both are generic over the trait [`DualNum`], implemented by `f32`, `f64` and both of them for
//! every `T: DualNum`, so that they nest: `Dual<Dual<f64>>` and `Dual<HyperDual<f64>>` carry
//! second and third derivatives. A function written once over `D: DualNum` gives its value for
//! `D = f64` and its derivatives for the dual types. The arithmetic is that of the algebra of
//! each (`(a + bε)(c + dε) = ac + (ad + bc)ε`); the elementary functions — powers and roots,
//! exponentials and logarithms, the circular and hyperbolic functions and their inverses, `abs`,
//! `hypot`, `atan2`, `mul_add`, `min`, `max` and `signum` — take the value from the function of
//! the standard library and the derivative parts from the derivative of the function at the
//! value, formed so that it keeps its digits: `1/((1 − x)(1 + x))` rather than `1/(1 − x²)`,
//! `1/cosh²x` rather than `1 − tanh²x`.
//!
//! In the elementary functions a tangent part that is zero stays zero, even multiplied by an
//! infinite derivative, so that the variables that do not move in a pass do not turn the others'
//! derivatives into NaN: `∂/∂x (x + √y)` at `y = 0` is 1. The rule cannot tell such a variable
//! from an intermediate whose derivative happens to be 0 at the point (or underflows), and where
//! that intermediate meets an infinite derivative the true limit of `0·∞` is replaced by 0. Where
//! the composite is smooth that 0 is wrong, and nothing signals it: `cbrt(x³)`, `(x³)^(1/3)` and
//! `cbrt(sin³x)` have the derivative 0 at 0 here, not 1 (and `cbrt(x³)` at `10⁻³⁰⁰`, where `x³`
//! underflows); `cbrt(x⁶)` has the second derivative 0 at 0, not 2; `cbrt(x³ + y³)` the gradient
//! `(0, 0)` at the origin, not `(1, 1)`. num-dual gives NaN at those points. At a kink, as of
//! `√(x²)` at 0, the 0 is the choice the functions make for `abs`.
//!
//! The functions [`derivative`], [`second_derivative`], [`gradient`], [`jacobian`] and
//! [`hessian`], and their `value_and_…` forms, run the passes: one dual pass for a derivative,
//! one hyper-dual pass for a second derivative, `n` dual passes for a gradient or a Jacobian of
//! `n` variables, a column each, and `n(n + 1)/2` hyper-dual passes for a Hessian, an entry each.
//! Their results are what the solvers of [`numerics`](crate::math::numerics) take: a gradient
//! as an `n × 1` [`Matrix`](crate::math::matrix::Matrix), a Jacobian as `m × n`.
//!
//! Comparisons look at the values alone, so that a branch takes the same path for the value and
//! for its derivatives. Nothing here is constant-time.

mod derivatives;
mod functions;
mod hyper;
mod num;
mod ops;

pub use derivatives::{
    derivative, gradient, hessian, jacobian, second_derivative, value_and_derivative,
    value_and_derivatives, value_and_gradient, value_and_jacobian, value_gradient_and_hessian,
};
pub use hyper::HyperDual;
pub use num::DualNum;

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

/// The dual number `re + eps·ε`, `ε² = 0`.
///
/// A function `f` of it is `f(re) + f′(re)·eps·ε`: evaluated at `x + 1·ε` it carries the
/// derivative of `f` at `x` in `eps`, and at a point of several variables each `xᵢ + vᵢ·ε` it
/// carries the derivative along the direction `v`. The parts are of any [`DualNum`] type, so
/// that `Dual<Dual<f64>>` gives second derivatives and `Dual<HyperDual<f64>>` third. The
/// methods of [`DualNum`] state the derivative each function takes, at its edges too.
///
/// `+`, `-`, `*` and `/` take two dual numbers, owned or borrowed, or one and a float of its
/// underlying type on either side; the assignments, `Sum` and `Product` too. `==` and the order
/// compare the values alone: `Dual::new(1.0, 2.0) == Dual::new(1.0, 3.0)`, so that a branch on a
/// comparison takes the same path whatever the derivatives, and `partial_cmp` agrees with `==`.
/// `Display` writes `1.5-2ε`. Nothing is constant-time.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, DualNum};
///
/// // f(x) = x² + 3x at x = 2: f = 10, f′ = 7.
/// let x = Dual::new(2.0_f64, 1.0);
/// let f = x * x + Dual::new(3.0, 0.0) * x;
/// assert_eq!((f.re, f.eps), (10.0, 7.0));
///
/// // Floats on either side, and the elementary functions with their derivatives.
/// let g = (2.0 * x - 1.0) / x;
/// assert_eq!((g.re, g.eps), (1.5, 0.25));
/// let s = x.sin();
/// assert_eq!((s.re, s.eps), (2.0f64.sin(), 2.0f64.cos()));
/// assert_eq!(format!("{}", Dual::new(1.5, -2.0)), "1.5-2ε");
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Dual<T> {
    /// The value.
    pub re: T,
    /// The coefficient of `ε`: the derivative.
    pub eps: T,
}

impl<T> Dual<T> {
    /// The dual number `re + eps·ε`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::dual::Dual;
    ///
    /// let d = Dual::new(1.5, -2.0);
    /// assert_eq!((d.re, d.eps), (1.5, -2.0));
    /// ```
    #[inline]
    pub const fn new(re: T, eps: T) -> Self {
        Dual { re, eps }
    }
}

impl<T: DualNum> Dual<T> {
    /// The number of the value `re` and a zero derivative.
    #[inline]
    fn from_value(re: T) -> Self {
        Dual::new(re, T::ZERO)
    }
}

impl<T: Add<Output = T>> Add for Dual<T> {
    type Output = Self;

    /// The sum.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual::new(self.re + rhs.re, self.eps + rhs.eps)
    }
}

impl<T: Sub<Output = T>> Sub for Dual<T> {
    type Output = Self;

    /// The difference.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual::new(self.re - rhs.re, self.eps - rhs.eps)
    }
}

impl<T: Neg<Output = T>> Neg for Dual<T> {
    type Output = Self;

    /// The negation.
    #[inline]
    fn neg(self) -> Self {
        Dual::new(-self.re, -self.eps)
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>> Mul for Dual<T> {
    type Output = Self;

    /// The product: `(a + bε)(c + dε) = ac + (ad + bc)ε`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Dual::new(self.re * rhs.re, self.re * rhs.eps + self.eps * rhs.re)
    }
}

impl<T: DualNum> fmt::Display for Dual<T> {
    /// Writes `re+epsε` or `re-epsε`, each part as `Display` writes it, with the precision given,
    /// and a part that is itself a dual number in parentheses: `1.5-2ε`, `1.50-2.00ε` with
    /// `{:.2}`, `(1+1ε)+(1+0ε)ε`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision();
        write_part(f, p, &self.re, "")?;
        write_part(f, p, &self.eps, "ε")
    }
}

/// Writes one part of a number with the unit `unit` after it: the first part (`unit` empty) as it
/// is, a later one with its sign in front, and a part that is itself a dual number, whose text
/// has an `ε` in it, in parentheses after a `+`.
fn write_part<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    precision: Option<usize>,
    part: &T,
    unit: &str,
) -> fmt::Result {
    let text = match precision {
        Some(p) => format!("{part:.p$}"),
        None => format!("{part}"),
    };
    if text.contains('ε') {
        let plus = if unit.is_empty() { "" } else { "+" };
        write!(f, "{plus}({text}){unit}")
    } else if unit.is_empty() {
        f.write_str(&text)
    } else if let Some(abs) = text.strip_prefix('-') {
        write!(f, "-{abs}{unit}")
    } else {
        write!(f, "+{text}{unit}")
    }
}
