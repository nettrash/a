//! The operators of dual and hyper-dual numbers: with another of the same type and with the
//! underlying float on either side, owned or borrowed, with their assignments; negation; `Sum`
//! and `Product`; and the comparisons, by value.

use super::num::DualNum;
use super::{Dual, HyperDual};
use crate::math::numerics::Real;
use core::cmp::Ordering;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: DualNum> Dual<T> {
    #[inline]
    fn plus(a: Self, b: Self) -> Self {
        Dual::new(a.re + b.re, a.eps + b.eps)
    }

    #[inline]
    fn minus(a: Self, b: Self) -> Self {
        Dual::new(a.re - b.re, a.eps - b.eps)
    }

    #[inline]
    fn times(a: Self, b: Self) -> Self {
        Dual::new(a.re * b.re, a.re * b.eps + a.eps * b.re)
    }

    /// `a / b` as the solution `q` of `q·b = a`: `q₀ = a₀/b₀` and `q₁ = (a₁ − q₀·b₁)/b₀`, which,
    /// unlike `(a₁·b₀ − a₀·b₁)/b₀²`, does not overflow in the square.
    #[inline]
    fn over(a: Self, b: Self) -> Self {
        let q = a.re / b.re;
        Dual::new(q, (a.eps - q * b.eps) / b.re)
    }

    #[inline]
    fn plus_real(a: Self, b: T::Float) -> Self {
        Dual::new(a.re + b, a.eps)
    }

    #[inline]
    fn minus_real(a: Self, b: T::Float) -> Self {
        Dual::new(a.re - b, a.eps)
    }

    #[inline]
    fn times_real(a: Self, b: T::Float) -> Self {
        Dual::new(a.re * b, a.eps * b)
    }

    #[inline]
    fn over_real(a: Self, b: T::Float) -> Self {
        Dual::new(a.re / b, a.eps / b)
    }

    #[inline]
    fn real_plus(a: T::Float, b: Self) -> Self {
        Dual::new(b.re + a, b.eps)
    }

    #[inline]
    fn real_minus(a: T::Float, b: Self) -> Self {
        Dual::new(-b.re + a, -b.eps)
    }

    #[inline]
    fn real_times(a: T::Float, b: Self) -> Self {
        Dual::new(b.re * a, b.eps * a)
    }

    /// `a / b` for a float `a`: `q₀ = a/b₀`, `q₁ = −q₀·b₁/b₀`.
    #[inline]
    fn real_over(a: T::Float, b: Self) -> Self {
        let q = T::constant(a) / b.re;
        Dual::new(q, -(q * b.eps) / b.re)
    }

    #[inline]
    fn negate(a: Self) -> Self {
        Dual::new(-a.re, -a.eps)
    }
}

impl<T: DualNum> HyperDual<T> {
    #[inline]
    fn plus(a: Self, b: Self) -> Self {
        HyperDual::new(
            a.re + b.re,
            a.eps1 + b.eps1,
            a.eps2 + b.eps2,
            a.eps1eps2 + b.eps1eps2,
        )
    }

    #[inline]
    fn minus(a: Self, b: Self) -> Self {
        HyperDual::new(
            a.re - b.re,
            a.eps1 - b.eps1,
            a.eps2 - b.eps2,
            a.eps1eps2 - b.eps1eps2,
        )
    }

    /// The product, with `ε₁² = ε₂² = 0`: the `ε₁ε₂` part is `a₀b₁₂ + a₁b₂ + a₂b₁ + a₁₂b₀`.
    #[inline]
    fn times(a: Self, b: Self) -> Self {
        HyperDual::new(
            a.re * b.re,
            a.re * b.eps1 + a.eps1 * b.re,
            a.re * b.eps2 + a.eps2 * b.re,
            a.re * b.eps1eps2 + a.eps1 * b.eps2 + a.eps2 * b.eps1 + a.eps1eps2 * b.re,
        )
    }

    /// `a / b` as the solution `q` of `q·b = a`, part after part: each part of `q` is the
    /// remainder of that part of `a` divided by `b₀`, so nothing is squared that could overflow.
    #[inline]
    fn over(a: Self, b: Self) -> Self {
        let q = a.re / b.re;
        let q1 = (a.eps1 - q * b.eps1) / b.re;
        let q2 = (a.eps2 - q * b.eps2) / b.re;
        let q12 = (a.eps1eps2 - q * b.eps1eps2 - q1 * b.eps2 - q2 * b.eps1) / b.re;
        HyperDual::new(q, q1, q2, q12)
    }

    #[inline]
    fn plus_real(a: Self, b: T::Float) -> Self {
        HyperDual::new(a.re + b, a.eps1, a.eps2, a.eps1eps2)
    }

    #[inline]
    fn minus_real(a: Self, b: T::Float) -> Self {
        HyperDual::new(a.re - b, a.eps1, a.eps2, a.eps1eps2)
    }

    #[inline]
    fn times_real(a: Self, b: T::Float) -> Self {
        HyperDual::new(a.re * b, a.eps1 * b, a.eps2 * b, a.eps1eps2 * b)
    }

    #[inline]
    fn over_real(a: Self, b: T::Float) -> Self {
        HyperDual::new(a.re / b, a.eps1 / b, a.eps2 / b, a.eps1eps2 / b)
    }

    #[inline]
    fn real_plus(a: T::Float, b: Self) -> Self {
        HyperDual::new(b.re + a, b.eps1, b.eps2, b.eps1eps2)
    }

    #[inline]
    fn real_minus(a: T::Float, b: Self) -> Self {
        HyperDual::new(-b.re + a, -b.eps1, -b.eps2, -b.eps1eps2)
    }

    #[inline]
    fn real_times(a: T::Float, b: Self) -> Self {
        HyperDual::new(b.re * a, b.eps1 * a, b.eps2 * a, b.eps1eps2 * a)
    }

    #[inline]
    fn real_over(a: T::Float, b: Self) -> Self {
        Self::over(Self::from_value(T::constant(a)), b)
    }

    #[inline]
    fn negate(a: Self) -> Self {
        HyperDual::new(-a.re, -a.eps1, -a.eps2, -a.eps1eps2)
    }
}

/// An operator between two numbers of the type, owned or borrowed, and with an owned float on the
/// right, with its assignments; `owned` adds the one for two owned numbers, which the core's
/// `+`, `-` and `*` of `Dual` already have.
macro_rules! operators {
    ($s:ident, $op:ident, $method:ident, $assign:ident, $assign_method:ident, $both:ident, $real:ident, owned) => {
        impl<T: DualNum> $op for $s<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                $s::$both(self, rhs)
            }
        }
        operators!($s, $op, $method, $assign, $assign_method, $both, $real);
    };
    ($s:ident, $op:ident, $method:ident, $assign:ident, $assign_method:ident, $both:ident, $real:ident) => {
        impl<T: DualNum> $op<&$s<T>> for $s<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                $s::$both(self, *rhs)
            }
        }
        impl<T: DualNum> $op<$s<T>> for &$s<T> {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: $s<T>) -> $s<T> {
                $s::$both(*self, rhs)
            }
        }
        impl<T: DualNum> $op<&$s<T>> for &$s<T> {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: &$s<T>) -> $s<T> {
                $s::$both(*self, *rhs)
            }
        }
        impl<T: DualNum<Float = F>, F: Real> $op<F> for $s<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: F) -> Self {
                $s::$real(self, rhs)
            }
        }
        impl<T: DualNum<Float = F>, F: Real> $op<F> for &$s<T> {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: F) -> $s<T> {
                $s::$real(*self, rhs)
            }
        }
        impl<T: DualNum> $assign for $s<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = $s::$both(*self, rhs);
            }
        }
        impl<T: DualNum> $assign<&$s<T>> for $s<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &Self) {
                *self = $s::$both(*self, *rhs);
            }
        }
        impl<T: DualNum<Float = F>, F: Real> $assign<F> for $s<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: F) {
                *self = $s::$real(*self, rhs);
            }
        }
    };
}

/// An operator with a borrowed float on the right, and its assignment, for each float type: one
/// implementation for every float type `F` would overlap with the one for an owned `F`, as `&F`
/// could be such a type for all the compiler can tell.
macro_rules! borrowed {
    ($s:ident, $t:ident, $op:ident, $method:ident, $assign:ident, $assign_method:ident, $real:ident) => {
        impl<T: DualNum<Float = $t>> $op<&$t> for $s<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &$t) -> Self {
                $s::$real(self, *rhs)
            }
        }
        impl<T: DualNum<Float = $t>> $op<&$t> for &$s<T> {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: &$t) -> $s<T> {
                $s::$real(*self, *rhs)
            }
        }
        impl<T: DualNum<Float = $t>> $assign<&$t> for $s<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &$t) {
                *self = $s::$real(*self, *rhs);
            }
        }
    };
}

macro_rules! borrowed_all {
    ($s:ident, $t:ident) => {
        borrowed!($s, $t, Add, add, AddAssign, add_assign, plus_real);
        borrowed!($s, $t, Sub, sub, SubAssign, sub_assign, minus_real);
        borrowed!($s, $t, Mul, mul, MulAssign, mul_assign, times_real);
        borrowed!($s, $t, Div, div, DivAssign, div_assign, over_real);
    };
}

borrowed_all!(Dual, f32);
borrowed_all!(Dual, f64);
borrowed_all!(HyperDual, f32);
borrowed_all!(HyperDual, f64);

// The core's `+`, `-` and `*` of two owned dual numbers stay as they are (in `mod.rs`).
operators!(Dual, Add, add, AddAssign, add_assign, plus, plus_real);
operators!(Dual, Sub, sub, SubAssign, sub_assign, minus, minus_real);
operators!(Dual, Mul, mul, MulAssign, mul_assign, times, times_real);
operators!(
    Dual, Div, div, DivAssign, div_assign, over, over_real, owned
);
operators!(
    HyperDual, Add, add, AddAssign, add_assign, plus, plus_real, owned
);
operators!(
    HyperDual, Sub, sub, SubAssign, sub_assign, minus, minus_real, owned
);
operators!(
    HyperDual, Mul, mul, MulAssign, mul_assign, times, times_real, owned
);
operators!(
    HyperDual, Div, div, DivAssign, div_assign, over, over_real, owned
);

/// An operator with a float on the left of a number of its float type, which the orphan rule
/// leaves to one implementation for each float type.
macro_rules! left {
    ($s:ident, $t:ident, $op:ident, $method:ident, $left:ident) => {
        impl<T: DualNum<Float = $t>> $op<$s<T>> for $t {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: $s<T>) -> $s<T> {
                $s::$left(self, rhs)
            }
        }
        impl<T: DualNum<Float = $t>> $op<&$s<T>> for $t {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: &$s<T>) -> $s<T> {
                $s::$left(self, *rhs)
            }
        }
        impl<T: DualNum<Float = $t>> $op<$s<T>> for &$t {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: $s<T>) -> $s<T> {
                $s::$left(*self, rhs)
            }
        }
        impl<T: DualNum<Float = $t>> $op<&$s<T>> for &$t {
            type Output = $s<T>;
            #[inline]
            fn $method(self, rhs: &$s<T>) -> $s<T> {
                $s::$left(*self, *rhs)
            }
        }
    };
}

macro_rules! lefts {
    ($s:ident, $t:ident) => {
        left!($s, $t, Add, add, real_plus);
        left!($s, $t, Sub, sub, real_minus);
        left!($s, $t, Mul, mul, real_times);
        left!($s, $t, Div, div, real_over);
    };
}

lefts!(Dual, f32);
lefts!(Dual, f64);
lefts!(HyperDual, f32);
lefts!(HyperDual, f64);

/// Negation of a borrowed number, `Sum` and `Product` of owned and borrowed ones, and the
/// comparisons by value, with each other and with the float.
macro_rules! common {
    ($s:ident) => {
        impl<T: DualNum> Neg for &$s<T> {
            type Output = $s<T>;
            #[inline]
            fn neg(self) -> $s<T> {
                $s::negate(*self)
            }
        }

        impl<T: DualNum> Sum for $s<T> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($s::from_value(T::ZERO), |s, z| s + z)
            }
        }

        impl<'a, T: DualNum> Sum<&'a $s<T>> for $s<T> {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold($s::from_value(T::ZERO), |s, z| s + z)
            }
        }

        impl<T: DualNum> Product for $s<T> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($s::from_value(T::ONE), |p, z| p * z)
            }
        }

        impl<'a, T: DualNum> Product<&'a $s<T>> for $s<T> {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold($s::from_value(T::ONE), |p, z| p * z)
            }
        }

        impl<T: DualNum> PartialEq for $s<T> {
            /// Whether the values are equal; the derivative parts are not compared.
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.re == other.re
            }
        }

        impl<T: DualNum<Float = F>, F: Real> PartialEq<F> for $s<T> {
            /// Whether the value is `other`; the derivative parts are not compared.
            #[inline]
            fn eq(&self, other: &F) -> bool {
                self.re == *other
            }
        }

        impl<T: DualNum> PartialOrd for $s<T> {
            /// The order of the values; the derivative parts are not compared.
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.re.partial_cmp(&other.re)
            }
        }

        impl<T: DualNum<Float = F>, F: Real> PartialOrd<F> for $s<T> {
            /// The order of the value and `other`; the derivative parts are not compared.
            #[inline]
            fn partial_cmp(&self, other: &F) -> Option<Ordering> {
                self.re.partial_cmp(other)
            }
        }
    };
}

common!(Dual);
common!(HyperDual);

impl<T: DualNum> Neg for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        HyperDual::negate(self)
    }
}
