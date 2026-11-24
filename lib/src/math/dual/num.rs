//! The trait [`DualNum`] of the numbers a function to be differentiated is written over, and its
//! implementations for `f32` and `f64`, whose methods are the functions of the standard library.

use crate::math::numerics::Real;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub(super) mod sealed {
    /// The types the crate implements [`DualNum`](super::DualNum) for.
    pub trait Sealed {
        /// Whether every part is zero. A tangent that is zero contributes nothing to a derivative,
        /// even where the derivative it multiplies is infinite or NaN, as `√x` at 0 is: the
        /// variables that do not move in a pass of a gradient must not turn it into NaN.
        #[doc(hidden)]
        fn all_parts_zero(&self) -> bool;
    }
}

/// The numbers a function is written over to be differentiated: `f32` and `f64`, and
/// [`Dual<T>`](super::Dual) and [`HyperDual<T>`](super::HyperDual) of any `T: DualNum`, so that
/// they nest, as `Dual<Dual<f64>>` and `Dual<HyperDual<f64>>` do for higher derivatives.
///
/// A function written once, generic over `D: DualNum`, gives its value for `D = f64` and its
/// derivatives for `D = Dual<f64>` or `HyperDual<f64>`. The trait carries the arithmetic (with
/// another `D` and with the underlying float [`Float`](DualNum::Float) on the right of `+`, `-`,
/// `*` and `/`), the comparisons, which look at the values alone ([`re`](DualNum::re)), and the
/// elementary functions, each with its exact derivative. The methods of `f32` and `f64` are
/// those of the standard library, but for [`atanh`](DualNum::atanh), whose negative arguments
/// are taken as `−atanh(−x)`; for the dual numbers the value is the same function of the
/// value, and the derivative parts follow from the derivatives of the function at the value, as
/// [`Dual`](super::Dual) and [`HyperDual`](super::HyperDual) describe. Nothing is
/// constant-time.
///
/// In the elementary functions a tangent part that is zero stays zero whatever the derivative it
/// multiplies, so that a variable that does not move contributes nothing even at a point where a
/// derivative is infinite: `∂/∂x (x + √y)` at `y = 0` is 1, not NaN. The arithmetic is the
/// floats', in which an infinite value times a zero tangent part is NaN. **The rule is a limit
/// too**: it cannot tell a variable that does not move from an intermediate whose derivative is 0
/// at the point (or underflows), and where such an intermediate meets an infinite derivative the
/// true limit of `0·∞` is replaced by 0, a wrong finite derivative where the composite is smooth,
/// with nothing to signal it. `cbrt(x³)`, `(x³)^(1/3)` and `cbrt(sin³x)` have the derivative 0 at
/// 0 here, not 1, and `cbrt(x³)` at `10⁻³⁰⁰`, where `x³` underflows; `cbrt(x⁶)` the second
/// derivative 0 at 0, not 2; `cbrt(x³ + y³)` the gradient `(0, 0)` at the origin, not `(1, 1)`.
/// num-dual gives NaN at these points.
///
/// With the float fixed, `D: DualNum<Float = f64>`, a float literal can stand on the right of an
/// operator, `x * 2.0 - 1.0`, and in a comparison, `x > 0.0` or `x.re() > 0.0`. On the left it
/// cannot by itself: `2.0 * x` needs `f64: Mul<D>`, which the trait cannot promise (a bound on
/// its associated type would hold of `D::Float`, and is not used once that is known to be
/// `f64`), and the orphan rule forbids an implementation of it for every `D`. A function may ask
/// for it, `where f64: Mul<D, Output = D>`, which every dual type of `f64` meets; or a float on the
/// left is written `D::constant(2.0) * x`. With the float generic too, `D: DualNum`, the literals are
/// `D::from_f64(2.0)`. The concrete types take a float on either side: `2.0 * Dual::new(1.0, 1.0)`.
/// `Dual<T>` and `HyperDual<T>` are `DualNum` for every `T: DualNum`, with no further condition,
/// so code generic over the float can build its own dual numbers and pass them on.
///
/// The trait is sealed: the crate may add methods to it.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, DualNum, HyperDual};
///
/// // Written once: the value, and the derivatives by dual and hyper-dual numbers.
/// fn f<D: DualNum<Float = f64>>(x: D) -> D {
///     if x > 0.0 { x.sin() * x + 1.0 } else { D::constant(1.0) - x }
/// }
/// assert_eq!(f(2.0), 2.0f64.sin() * 2.0 + 1.0);
/// let d = f(Dual::new(2.0, 1.0));
/// assert_eq!(d.re, f(2.0));
/// assert!((d.eps - (2.0f64.cos() * 2.0 + 2.0f64.sin())).abs() < 1e-15);
/// let h = f(HyperDual::new(2.0, 1.0, 1.0, 0.0));
/// assert!((h.eps1eps2 - (2.0 * 2.0f64.cos() - 2.0 * 2.0f64.sin())).abs() < 1e-15);
/// assert_eq!(f(Dual::new(-1.0, 1.0)).eps, -1.0);
/// ```
pub trait DualNum:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + PartialEq<<Self as DualNum>::Float>
    + PartialOrd<<Self as DualNum>::Float>
    + fmt::Debug
    + fmt::Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Add<<Self as DualNum>::Float, Output = Self>
    + Sub<<Self as DualNum>::Float, Output = Self>
    + Mul<<Self as DualNum>::Float, Output = Self>
    + Div<<Self as DualNum>::Float, Output = Self>
    + AddAssign<<Self as DualNum>::Float>
    + SubAssign<<Self as DualNum>::Float>
    + MulAssign<<Self as DualNum>::Float>
    + DivAssign<<Self as DualNum>::Float>
    + Sum
    + Product
    + Send
    + Sync
    + 'static
    + sealed::Sealed
{
    /// The underlying float, `f32` or `f64`: the type of the value and of every part, however
    /// deep the nesting.
    type Float: Real;

    /// Zero, with zero derivative parts.
    const ZERO: Self;
    /// One, with zero derivative parts.
    const ONE: Self;

    /// The constant `x`: the value `x` and zero derivative parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::dual::{Dual, DualNum};
    ///
    /// let c = Dual::<f64>::constant(2.5);
    /// assert_eq!((c.re, c.eps), (2.5, 0.0));
    /// ```
    fn constant(x: Self::Float) -> Self;

    /// The constant nearest `x`, for literals in code generic over the float too.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::dual::{Dual, DualNum};
    ///
    /// fn half<D: DualNum>(x: D) -> D {
    ///     x * D::from_f64(0.5)
    /// }
    /// assert_eq!(half(Dual::new(3.0f32, 1.0)).eps, 0.5);
    /// assert_eq!(f32::from_f64(0.1), 0.1f32);
    /// ```
    fn from_f64(x: f64) -> Self;

    /// The value: the number itself for a float, the innermost value of a dual number. Branches
    /// on it take the same path for the value and for its derivatives.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::dual::{Dual, DualNum};
    ///
    /// let x = Dual::new(Dual::new(1.5, 1.0), Dual::new(1.0, 0.0));
    /// assert_eq!(x.re(), 1.5);
    /// assert_eq!(2.0f64.re(), 2.0);
    /// ```
    fn re(self) -> Self::Float;

    /// `1 / self`; the derivative is `−1/x²`.
    fn recip(self) -> Self;
    /// The square root; the derivative `1/(2√x)` is `+∞` at 0, for either zero (the derivative
    /// from the right), and NaN for `x < 0`.
    fn sqrt(self) -> Self;
    /// The cube root, of the sign of `self`; the derivative `1/(3∛x²)` is `+∞` at either zero.
    fn cbrt(self) -> Self;
    /// `self` to the integer power `n`; the derivative `n·xⁿ⁻¹` is 0 for `n = 0`, also at 0. Where
    /// the power `xⁿ⁻¹` alone falls below the normal numbers though `n·xⁿ⁻¹` need not (subnormal,
    /// or 0 where the standard library forms a negative power as `1/x^|m|` and `x^|m|`
    /// overflows), it is formed in two halves with the coefficient between them,
    /// `(x^(m − ⌊m/2⌋)·n)·x^⌊m/2⌋`, `m = n − 1`; the second derivative alike.
    fn powi(self, n: i32) -> Self;
    /// `self` to the real power `e`; the derivative `e·xᵉ⁻¹` is 0 for `e = 0`, at 0 it is 0 for
    /// `e > 1`, 1 for `e = 1` and infinite for `e < 1`, and for `x < 0` it is finite where
    /// `xᵉ⁻¹` is, that is where `e` is an integer. The power is `x^e₁·(1 + δ·ln x)`, `e₁` the
    /// rounded `e − 1` and `δ` its exact remainder, so that the rounding of `e − 1`, which `ln x`
    /// would multiply (145 ulps of `0.1·x^(−0.9)` at `10³⁰⁰`), does not enter; and where `x^e₁`
    /// alone leaves the range though `e·xᵉ⁻¹` does not, `(e·xᵉ)/x`. The second derivative alike.
    /// For `|e| ≥ 2ᵖ − 2` (`2⁵³ − 2` in `f64`, `2²⁴ − 2` in `f32`) and an infinite `e`, where every
    /// float is an integer and `e − 1` or `e − 2` need not be a float, the powers are taken from
    /// the value, `xᵉ/x` and `xᵉ/x²`, with the parity of `e − 1` and `e − 2` and the sign of an
    /// overflow, and at `x = ±0` and `±∞` as `|x|^(e − k)` with that parity: the derivative of
    /// `x^(2⁵³ + 2)` at −1 is `−(2⁵³ + 2)`, the second derivative of `x^(−(2⁵³ − 1))` there is
    /// negative, that of `x^(2⁶⁰)` at 3 is `+∞`, at 0 they are 0, and at `±∞` they are infinite with
    /// the sign of `e·xᵉ⁻¹` and `e(e − 1)·xᵉ⁻²` (the first `−∞` at −∞ for an even `e`); at `x = 1`
    /// an infinite `e` gives `f′ = e` and `f″ = +∞`.
    ///
    /// Where the power `xᵉ⁻¹` (of `f′`) or `xᵉ⁻²` (of `f″`) is subnormal though the derivative is a
    /// normal number — within a factor `|e|` or `|e(e − 1)|` of the smallest normal number — the
    /// derivative has only the subnormal's digits, by every form; where that power, or `xᵉ` for
    /// the exponents beyond `2ᵖ − 2`, underflows to 0, the derivative is 0: the second derivative
    /// of `x^(−2⁶⁰)` at `1 + 3·2⁻⁵²` is 0, not `3.8·10⁻²⁹⁸`.
    fn powf(self, e: Self::Float) -> Self;
    /// `self` to the power `y`, which may vary too: the derivative with respect to `x` is formed
    /// as by [`powf`](DualNum::powf); that with respect to `y` is `xʸ·ln x`, taken as 0 at `x = 0`
    /// where `xʸ` is 0 (its limit for `y > 0`), formed where `xʸ` only underflows, and NaN for
    /// `x < 0`, where only an exponent that does not vary has a derivative. A hyper-dual number
    /// takes its second derivatives in `y` as 0 there too: the limit of `xʸ·ln²x`, but a choice for
    /// the mixed `∂²/∂x∂y = xʸ⁻¹·(1 + y ln x)`, whose limit is 0 for `y > 1` and `−∞` for
    /// `0 < y ≤ 1`.
    fn pow(self, y: Self) -> Self;
    /// `eˣ`.
    fn exp(self) -> Self;
    /// `2ˣ`; the derivative is `2ˣ·ln 2`.
    fn exp2(self) -> Self;
    /// `eˣ − 1`, accurate near zero; the derivative is `eˣ`.
    fn exp_m1(self) -> Self;
    /// The natural logarithm; the derivative is `1/x`, `+∞` at either zero.
    fn ln(self) -> Self;
    /// The base-2 logarithm; the derivative is `1/(x·ln 2)`, `+∞` at either zero.
    fn log2(self) -> Self;
    /// The base-10 logarithm; the derivative is `1/(x·ln 10)`, `+∞` at either zero.
    fn log10(self) -> Self;
    /// `ln(1 + x)`, accurate near zero; the derivative is `1/(1 + x)`.
    fn ln_1p(self) -> Self;
    /// The logarithm to the base `base`; the derivative is `1/(x·ln base)`, infinite at either
    /// zero.
    fn log(self, base: Self::Float) -> Self;
    /// The sine.
    fn sin(self) -> Self;
    /// The cosine.
    fn cos(self) -> Self;
    /// The tangent; the derivative is `1 + tan²x`.
    fn tan(self) -> Self;
    /// The sine and the cosine.
    fn sin_cos(self) -> (Self, Self);
    /// The arcsine; the derivative `1/√(1 − x²)`, its radicand formed as `(1 − x)(1 + x)` near ±1,
    /// keeps its digits there, where it is infinite.
    fn asin(self) -> Self;
    /// The arccosine; the derivative `−1/√(1 − x²)`, its radicand formed as `(1 − x)(1 + x)` near
    /// ±1, keeps its digits there, where it is infinite.
    fn acos(self) -> Self;
    /// The arctangent; the derivative `1/(1 + x²)` is formed as `r²/(1 + r²)`, `r = 1/x`, for
    /// `|x| > 1`, so that it does not overflow.
    fn atan(self) -> Self;
    /// The angle of the point `(x, self)`, as `f64::atan2`; the derivatives
    /// `(x·dy − y·dx)/(x² + y²)` are formed as `(x/r)/r` and `(y/r)/r` with `r = hypot(x, y)`, so
    /// that no square overflows or underflows, and are NaN at the origin, where the angle is not
    /// continuous. The mixed second partial `(y² − x²)/r⁴` is formed from `(y − x)(y + x)`, which
    /// does not cancel next to the diagonals (by `HyperDual`; the nested `Dual<Dual<_>>`, which
    /// differentiates the first partials, does cancel there), and from the halves of `y` and `x`
    /// beyond half the largest float, where `y ± x` would overflow.
    fn atan2(self, x: Self) -> Self;
    /// The hyperbolic sine.
    fn sinh(self) -> Self;
    /// The hyperbolic cosine.
    fn cosh(self) -> Self;
    /// The hyperbolic tangent; the derivative is `1/cosh²x`, which keeps its digits for large
    /// `|x|`, where `1 − tanh²x` would be 0, with `1/cosh x` as `2e^(−|x|)` beyond 20, where
    /// `cosh x` could overflow.
    fn tanh(self) -> Self;
    /// The inverse hyperbolic sine; the derivative is `1/hypot(1, x)`, which does not overflow.
    fn asinh(self) -> Self;
    /// The inverse hyperbolic cosine; the derivative `1/(√(x − 1)·√(x + 1))` keeps its digits
    /// near 1, where it is infinite, and does not overflow for large `x`.
    fn acosh(self) -> Self;
    /// The inverse hyperbolic tangent; the derivative is `1/(1 − x²)`, its denominator formed as
    /// `(1 − x)(1 + x)` near ±1. For `f32` and `f64` a negative argument is taken as
    /// `−atanh(−x)`: the standard library's formula loses its digits near −1 (by about 10¹⁴ ulps
    /// at `−1 + 2⁻⁵³`) though not near 1.
    fn atanh(self) -> Self;
    /// The absolute value; the derivative is the sign of the value, and 0 at 0 (for either
    /// zero), the middle of the slopes −1 and 1 there.
    fn abs(self) -> Self;
    /// `√(self² + y²)` without an overflow or underflow in the squares; at the origin, where it
    /// is not differentiable, every derivative part is 0. The second partials, as `y²/h³`, are
    /// formed as `(y/h)·((y/h)/h)`, so that the square of a small ratio does not underflow first.
    fn hypot(self, y: Self) -> Self;
    /// `self · a + b`, with one rounding of the value for the floats.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// The smaller value, ignoring a NaN, with the derivatives of the one chosen; of `self` where
    /// the values are equal.
    fn min(self, other: Self) -> Self;
    /// The larger value, ignoring a NaN, with the derivatives of the one chosen; of `self` where
    /// the values are equal.
    fn max(self, other: Self) -> Self;
    /// `1` or `−1` as the sign of the value (`±1` for `±0`, NaN for NaN), with zero derivatives,
    /// also at the jump at 0.
    fn signum(self) -> Self;
}

macro_rules! float {
    ($t:ident) => {
        impl sealed::Sealed for $t {
            #[inline]
            fn all_parts_zero(&self) -> bool {
                *self == 0.0
            }
        }

        impl DualNum for $t {
            type Float = $t;

            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline]
            fn constant(x: $t) -> Self {
                x
            }
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn re(self) -> $t {
                self
            }
            #[inline]
            fn recip(self) -> Self {
                1.0 / self
            }
            #[inline]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            #[inline]
            fn cbrt(self) -> Self {
                $t::cbrt(self)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                $t::powi(self, n)
            }
            #[inline]
            fn powf(self, e: $t) -> Self {
                $t::powf(self, e)
            }
            #[inline]
            fn pow(self, y: Self) -> Self {
                $t::powf(self, y)
            }
            #[inline]
            fn exp(self) -> Self {
                $t::exp(self)
            }
            #[inline]
            fn exp2(self) -> Self {
                $t::exp2(self)
            }
            #[inline]
            fn exp_m1(self) -> Self {
                $t::exp_m1(self)
            }
            #[inline]
            fn ln(self) -> Self {
                $t::ln(self)
            }
            #[inline]
            fn log2(self) -> Self {
                $t::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                $t::log10(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                $t::ln_1p(self)
            }
            #[inline]
            fn log(self, base: $t) -> Self {
                $t::log(self, base)
            }
            #[inline]
            fn sin(self) -> Self {
                $t::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                $t::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                $t::tan(self)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                $t::sin_cos(self)
            }
            #[inline]
            fn asin(self) -> Self {
                $t::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                $t::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                $t::atan(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                $t::atan2(self, x)
            }
            #[inline]
            fn sinh(self) -> Self {
                $t::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                $t::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                $t::tanh(self)
            }
            #[inline]
            fn asinh(self) -> Self {
                $t::asinh(self)
            }
            #[inline]
            fn acosh(self) -> Self {
                $t::acosh(self)
            }
            #[inline]
            fn atanh(self) -> Self {
                // The standard library's `0.5·ln_1p(2x/(1 − x))` is exact in `1 − x` near 1 but
                // not near −1, where it is wrong by up to 10¹⁴ ulps: the function is odd, so a
                // negative argument takes the side that is.
                if self < 0.0 {
                    -$t::atanh(-self)
                } else {
                    $t::atanh(self)
                }
            }
            #[inline]
            fn abs(self) -> Self {
                $t::abs(self)
            }
            #[inline]
            fn hypot(self, y: Self) -> Self {
                $t::hypot(self, y)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $t::mul_add(self, a, b)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                $t::min(self, other)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                $t::max(self, other)
            }
            #[inline]
            fn signum(self) -> Self {
                $t::signum(self)
            }
        }
    };
}

float!(f32);
float!(f64);

/// `d·t`, or zero where every part of the tangent `t` is zero: a variable that does not move
/// contributes nothing, even where the derivative `d` is infinite or NaN.
#[inline]
pub(super) fn scale<T: DualNum>(d: T, t: T) -> T {
    if t.all_parts_zero() { T::ZERO } else { d * t }
}

/// The float `x` as the type `F`.
#[inline]
pub(super) fn float<F: Real>(x: f64) -> F {
    F::from_f64(x)
}
