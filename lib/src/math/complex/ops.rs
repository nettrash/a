//! The operators of complex numbers.

use super::{Complex, Float};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Float> Complex<T> {
    #[inline]
    fn plus(a: Self, b: Self) -> Self {
        Complex::new(a.re + b.re, a.im + b.im)
    }

    #[inline]
    fn plus_real(a: Self, b: T) -> Self {
        Complex::new(a.re + b, a.im)
    }

    #[inline]
    fn real_plus(a: T, b: Self) -> Self {
        Complex::new(a + b.re, b.im)
    }

    #[inline]
    fn minus(a: Self, b: Self) -> Self {
        Complex::new(a.re - b.re, a.im - b.im)
    }

    #[inline]
    fn minus_real(a: Self, b: T) -> Self {
        Complex::new(a.re - b, a.im)
    }

    #[inline]
    fn real_minus(a: T, b: Self) -> Self {
        Complex::new(a - b.re, -b.im)
    }

    #[inline]
    fn times(a: Self, b: Self) -> Self {
        Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }

    #[inline]
    fn times_real(a: Self, b: T) -> Self {
        Complex::new(a.re * b, a.im * b)
    }

    #[inline]
    fn real_times(a: T, b: Self) -> Self {
        Complex::new(a * b.re, a * b.im)
    }

    /// `a / b` by Smith's method: the larger part `p` of `b` divides the smaller `q`, so that
    /// nothing overflows where the quotient doesn't. The two cases differ only in which parts
    /// play which role and in a sign, chosen by selects rather than a branch, which random
    /// operands would mispredict half the time. A zero `b` takes `r = 0`, which leaves `a.re / 0`
    /// and `a.im / 0`, infinities or NaNs as for floats, by a select too. The sums are at most
    /// twice the largest part, so all four parts are halved first when one of them is at least
    /// half the largest float, which leaves the quotient as it was: `(1e308 + 1e308·i) /
    /// (1e308 + 1e308·i)` is 1, not a NaN from `∞ / ∞`. A part that halving makes inexact is
    /// subnormal, beside one near the largest float, so the quotient underflows or overflows
    /// anyway. With no branch at all a quotient took 1.4 ns on an Apple M4 Max, against 1.6 with
    /// a branch for the zero divisor; four comparisons rather than a maximum of the parts saved
    /// 0.2 of it.
    #[inline]
    fn over(a: Self, b: Self) -> Self {
        let limit = T::MAX * T::HALF;
        let big = (a.re.abs() >= limit)
            | (a.im.abs() >= limit)
            | (b.re.abs() >= limit)
            | (b.im.abs() >= limit);
        let s = if big { T::HALF } else { T::ONE };
        let (ar, ai, br, bi) = (a.re * s, a.im * s, b.re * s, b.im * s);
        let swap = br.abs() < bi.abs();
        let (p, q) = if swap { (bi, br) } else { (br, bi) };
        let (x, y) = if swap { (ai, ar) } else { (ar, ai) };
        let r = if p == T::ZERO { T::ZERO } else { q / p };
        let d = p + q * r;
        let im = (y - x * r) / d;
        Complex::new((x + y * r) / d, if swap { -im } else { im })
    }

    #[inline]
    fn over_real(a: Self, b: T) -> Self {
        Complex::new(a.re / b, a.im / b)
    }

    #[inline]
    fn real_over(a: T, b: Self) -> Self {
        Complex::over(Complex::from(a), b)
    }
}

/// An operator between two complex numbers, and between a complex number and a float of its
/// type on the right, owned or borrowed, with its assignments.
macro_rules! operators {
    ($op:ident, $method:ident, $assign:ident, $assign_method:ident, $both:ident, $real:ident) => {
        impl<T: Float> $op for Complex<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Complex::$both(self, rhs)
            }
        }
        impl<T: Float> $op<&Complex<T>> for Complex<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                Complex::$both(self, *rhs)
            }
        }
        impl<T: Float> $op<Complex<T>> for &Complex<T> {
            type Output = Complex<T>;
            #[inline]
            fn $method(self, rhs: Complex<T>) -> Complex<T> {
                Complex::$both(*self, rhs)
            }
        }
        impl<T: Float> $op<&Complex<T>> for &Complex<T> {
            type Output = Complex<T>;
            #[inline]
            fn $method(self, rhs: &Complex<T>) -> Complex<T> {
                Complex::$both(*self, *rhs)
            }
        }
        impl<T: Float> $op<T> for Complex<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: T) -> Self {
                Complex::$real(self, rhs)
            }
        }
        impl<T: Float> $op<&T> for Complex<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &T) -> Self {
                Complex::$real(self, *rhs)
            }
        }
        impl<T: Float> $op<T> for &Complex<T> {
            type Output = Complex<T>;
            #[inline]
            fn $method(self, rhs: T) -> Complex<T> {
                Complex::$real(*self, rhs)
            }
        }
        impl<T: Float> $op<&T> for &Complex<T> {
            type Output = Complex<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Complex<T> {
                Complex::$real(*self, *rhs)
            }
        }
        impl<T: Float> $assign for Complex<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = Complex::$both(*self, rhs);
            }
        }
        impl<T: Float> $assign<&Complex<T>> for Complex<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &Self) {
                *self = Complex::$both(*self, *rhs);
            }
        }
        impl<T: Float> $assign<T> for Complex<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
                *self = Complex::$real(*self, rhs);
            }
        }
        impl<T: Float> $assign<&T> for Complex<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &T) {
                *self = Complex::$real(*self, *rhs);
            }
        }
    };
}

operators!(Add, add, AddAssign, add_assign, plus, plus_real);
operators!(Sub, sub, SubAssign, sub_assign, minus, minus_real);
operators!(Mul, mul, MulAssign, mul_assign, times, times_real);
operators!(Div, div, DivAssign, div_assign, over, over_real);

/// An operator with a float on the left of a complex number of its type, which the orphan rule
/// leaves to one implementation for each float type.
macro_rules! left {
    ($t:ident, $op:ident, $method:ident, $left:ident) => {
        impl $op<Complex<$t>> for $t {
            type Output = Complex<$t>;
            #[inline]
            fn $method(self, rhs: Complex<$t>) -> Complex<$t> {
                Complex::$left(self, rhs)
            }
        }
        impl $op<&Complex<$t>> for $t {
            type Output = Complex<$t>;
            #[inline]
            fn $method(self, rhs: &Complex<$t>) -> Complex<$t> {
                Complex::$left(self, *rhs)
            }
        }
        impl $op<Complex<$t>> for &$t {
            type Output = Complex<$t>;
            #[inline]
            fn $method(self, rhs: Complex<$t>) -> Complex<$t> {
                Complex::$left(*self, rhs)
            }
        }
        impl $op<&Complex<$t>> for &$t {
            type Output = Complex<$t>;
            #[inline]
            fn $method(self, rhs: &Complex<$t>) -> Complex<$t> {
                Complex::$left(*self, *rhs)
            }
        }
    };
}

macro_rules! lefts {
    ($t:ident) => {
        left!($t, Add, add, real_plus);
        left!($t, Sub, sub, real_minus);
        left!($t, Mul, mul, real_times);
        left!($t, Div, div, real_over);
    };
}

lefts!(f32);
lefts!(f64);

impl<T: Float> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: Float> Neg for &Complex<T> {
    type Output = Complex<T>;
    #[inline]
    fn neg(self) -> Complex<T> {
        -*self
    }
}

impl<T: Float> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |s, z| s + z)
    }
}

impl<'a, T: Float> Sum<&'a Complex<T>> for Complex<T> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |s, z| s + z)
    }
}

impl<T: Float> Product for Complex<T> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |p, z| p * z)
    }
}

impl<'a, T: Float> Product<&'a Complex<T>> for Complex<T> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |p, z| p * z)
    }
}
