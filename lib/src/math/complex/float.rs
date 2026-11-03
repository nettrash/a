//! The float types of the parts.

use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// `f32` or `f64`: the type of the parts of a [`Complex`](super::Complex), and of the real numbers
/// a matrix of either gives, such as its norms and singular values. The methods are those of the
/// two types, for code generic over both. The trait is sealed.
pub trait Float:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + fmt::Debug
    + fmt::Display
    + fmt::LowerExp
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
    + Product
    + Send
    + Sync
    + 'static
    + sealed::Sealed
{
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
    /// 2.
    const TWO: Self;
    /// 1/2.
    const HALF: Self;
    /// π.
    const PI: Self;
    /// Infinity.
    const INFINITY: Self;
    /// Not a number.
    const NAN: Self;
    /// The difference between 1 and the next larger float.
    const EPSILON: Self;
    /// The smallest positive normal float.
    const MIN_POSITIVE: Self;
    /// The largest finite float.
    const MAX: Self;
    /// The number of significant bits, the hidden one included.
    const MANTISSA_DIGITS: u32;

    /// The nearest float to `x`.
    fn from_f64(x: f64) -> Self;
    /// The value as an `f64`, exactly.
    fn to_f64(self) -> f64;
    /// The absolute value.
    fn abs(self) -> Self;
    /// The square root.
    fn sqrt(self) -> Self;
    /// `√(self² + other²)` without an overflow or underflow in the squares.
    fn hypot(self, other: Self) -> Self;
    /// The angle of the point `(other, self)`, in `[-π, π]`.
    fn atan2(self, other: Self) -> Self;
    /// The sine and the cosine.
    fn sin_cos(self) -> (Self, Self);
    /// `e^self`.
    fn exp(self) -> Self;
    /// The natural logarithm.
    fn ln(self) -> Self;
    /// `ln(1 + self)`, accurate near zero.
    fn ln_1p(self) -> Self;
    /// `self` to the integer power `n`.
    fn powi(self, n: i32) -> Self;
    /// `self · a + b` with one rounding.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// The absolute value of `self` with the sign of `sign`.
    fn copysign(self, sign: Self) -> Self;
    /// The larger of the two, ignoring a NaN.
    fn max(self, other: Self) -> Self;
    /// The smaller of the two, ignoring a NaN.
    fn min(self, other: Self) -> Self;
    /// Whether it is neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// Whether it is infinite.
    fn is_infinite(self) -> bool;
    /// Whether it is NaN.
    fn is_nan(self) -> bool;
    /// Whether its sign bit is set, as for `-0.0`.
    fn is_sign_negative(self) -> bool;
}

macro_rules! float {
    ($t:ident) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const PI: Self = core::$t::consts::PI;
            const INFINITY: Self = $t::INFINITY;
            const NAN: Self = $t::NAN;
            const EPSILON: Self = $t::EPSILON;
            const MIN_POSITIVE: Self = $t::MIN_POSITIVE;
            const MAX: Self = $t::MAX;
            const MANTISSA_DIGITS: u32 = $t::MANTISSA_DIGITS;

            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn abs(self) -> Self {
                $t::abs(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            #[inline]
            fn hypot(self, other: Self) -> Self {
                $t::hypot(self, other)
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                $t::atan2(self, other)
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                $t::sin_cos(self)
            }
            #[inline]
            fn exp(self) -> Self {
                $t::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                $t::ln(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                $t::ln_1p(self)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                $t::powi(self, n)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $t::mul_add(self, a, b)
            }
            #[inline]
            fn copysign(self, sign: Self) -> Self {
                $t::copysign(self, sign)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                $t::max(self, other)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                $t::min(self, other)
            }
            #[inline]
            fn is_finite(self) -> bool {
                $t::is_finite(self)
            }
            #[inline]
            fn is_infinite(self) -> bool {
                $t::is_infinite(self)
            }
            #[inline]
            fn is_nan(self) -> bool {
                $t::is_nan(self)
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                $t::is_sign_negative(self)
            }
        }
    };
}

float!(f32);
float!(f64);
