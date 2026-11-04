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
    /// and `a.im / 0`, infinities or NaNs as for floats, by a select too.
    ///
    /// At the ends of the range, as LAPACK's `ladiv` (Baudin and Smith), all four parts are
    /// scaled by one power of two, which leaves the quotient as it was: halved when one of them
    /// is at least half the largest float, as the sums are at most twice the largest part, so
    /// that `(1e308 + 1e308·i) / (1e308 + 1e308·i)` is 1, not a NaN from `∞ / ∞`; and scaled
    /// up by `8/ε²` when the parts of either operand are all below `4·MIN/ε`, so that Smith's
    /// products keep their digits and `(u + u·i) / (u + 2u·i)` for the smallest subnormal `u`
    /// is `0.6 - 0.2i`. Where that makes the other operand overflow or vanish, the quotient
    /// overflows or underflows anyway. Where `r` itself falls below the normal range, the
    /// products with it are taken as `(y·q)/p`, so that `1e300 / (1e100 + 1e-250·i)` keeps its
    /// imaginary part `-1e-150`. All of it took a quotient from 1.1 ns to 2.0 on an Apple M4
    /// Max; separate factors for the two operands took 2.3, and the rare cases out of line in a
    /// function of their own 4.1, its call alone keeping the loop from overlapping quotients.
    #[inline]
    fn over(a: Self, b: Self) -> Self {
        let zero = T::ZERO;
        let (aa, ab, ba, bb) = (a.re.abs(), a.im.abs(), b.re.abs(), b.im.abs());
        let limit = T::MAX * T::HALF;
        let tiny = T::MIN_POSITIVE * T::from_f64(4.0) / T::EPSILON;
        let big = (aa >= limit) | (ab >= limit) | (ba >= limit) | (bb >= limit);
        let small = ((aa <= tiny) & (ab <= tiny)) | ((ba <= tiny) & (bb <= tiny));
        let s = if big {
            T::HALF
        } else if small {
            T::from_f64(8.0) / (T::EPSILON * T::EPSILON)
        } else {
            T::ONE
        };
        let (ar, ai, br, bi) = (a.re * s, a.im * s, b.re * s, b.im * s);
        let swap = br.abs() < bi.abs();
        let (p, q) = if swap { (bi, br) } else { (br, bi) };
        let (x, y) = if swap { (ai, ar) } else { (ar, ai) };
        let r = if p == zero { zero } else { q / p };
        let d = p + q * r;
        let (yr, xr) = if r.abs() < T::MIN_POSITIVE && q != zero && p != zero {
            ((y * q) / p, (x * q) / p)
        } else {
            (y * r, x * r)
        };
        let im = (y - xr) / d;
        Complex::new((x + yr) / d, if swap { -im } else { im })
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
