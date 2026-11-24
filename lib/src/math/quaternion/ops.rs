//! The operators of quaternions.

use super::Quaternion;
use super::scale::{exponent, in_range};
use crate::math::complex::Float;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Float> Quaternion<T> {
    #[inline]
    fn plus(a: Self, b: Self) -> Self {
        Quaternion::new(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z)
    }

    #[inline]
    fn plus_real(a: Self, b: T) -> Self {
        Quaternion::new(a.w + b, a.x, a.y, a.z)
    }

    #[inline]
    fn real_plus(a: T, b: Self) -> Self {
        Quaternion::new(a + b.w, b.x, b.y, b.z)
    }

    #[inline]
    fn minus(a: Self, b: Self) -> Self {
        Quaternion::new(a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z)
    }

    #[inline]
    fn minus_real(a: Self, b: T) -> Self {
        Quaternion::new(a.w - b, a.x, a.y, a.z)
    }

    #[inline]
    fn real_minus(a: T, b: Self) -> Self {
        Quaternion::new(a - b.w, -b.x, -b.y, -b.z)
    }

    /// Hamilton's product, `(w₁w₂ − v₁·v₂) + (w₁v₂ + w₂v₁ + v₁ × v₂)`. Each part is a sum of four
    /// products whose absolute values add up to at most `|l|·|r|`, so that the error of a part is
    /// a few units of `ε` times the sum of the absolute values of its four products, and nothing
    /// overflows where the norm of the product does not. Where finite operands give a part that
    /// is not finite, a product in it overflowed and the formula met `∞ − ∞`, `∞·0` or the
    /// overflow itself; [`times_cold`](Self::times_cold) takes those parts again from scaled
    /// operands.
    #[inline]
    fn times(l: Self, r: Self) -> Self {
        let p = Self::hamilton(l, r);
        if p.is_finite() || !(l.is_finite() && r.is_finite()) {
            p
        } else {
            Self::times_cold(l, r, p)
        }
    }

    /// The formula of the product as it is, for operands that cannot overflow and for operands
    /// with infinite or NaN parts, where it gives the NaNs of `0·∞` and `∞ − ∞` as floats do.
    #[inline]
    pub(super) fn hamilton(l: Self, r: Self) -> Self {
        Quaternion::new(
            l.w * r.w - l.x * r.x - l.y * r.y - l.z * r.z,
            l.w * r.x + l.x * r.w + l.y * r.z - l.z * r.y,
            l.w * r.y - l.x * r.z + l.y * r.w + l.z * r.x,
            l.w * r.z + l.x * r.y - l.y * r.x + l.z * r.w,
        )
    }

    /// The product of finite operands that overflows, from its plain value `p`. A part that `p`
    /// has finite had no product in it overflow, and is kept as it is: accurate as any part of a
    /// product is. The others are taken from the product of the operands scaled exactly by the
    /// powers of two that bring their largest parts into `[1, 2)`, scaled back by the sum of the
    /// powers, rounded once: the sum of the absolute values of such a part's products is at
    /// least `2¹⁰²⁴`, beside which what the scaling loses of the smaller parts of the operands (at
    /// most `2⁻¹⁰⁷⁵` in each scaled part, and `2^(kl+kr)` times that once scaled back, with
    /// `kl + kr ≤ 2046`) is at most `8ε`, so that the part is within a few units of `ε` times
    /// that sum, as a part of any product is, and is `±∞` where it overflows by more.
    #[cold]
    #[inline(never)]
    fn times_cold(l: Self, r: Self, p: Self) -> Self {
        let (kl, kr) = (exponent(l.max_abs()), exponent(r.max_abs()));
        let s = Self::hamilton(l.ldexp(-kl), r.ldexp(-kr)).ldexp(kl + kr);
        let pick = |plain: T, scaled: T| if plain.is_finite() { plain } else { scaled };
        Quaternion::new(
            pick(p.w, s.w),
            pick(p.x, s.x),
            pick(p.y, s.y),
            pick(p.z, s.z),
        )
    }

    #[inline]
    fn times_real(a: Self, b: T) -> Self {
        Quaternion::new(a.w * b, a.x * b, a.y * b, a.z * b)
    }

    #[inline]
    fn real_times(a: T, b: Self) -> Self {
        Quaternion::new(a * b.w, a * b.x, a * b.y, a * b.z)
    }

    /// `a / b = a·b⁻¹ = a·conj(b) / |b|²`, the division on the right. Where the squared norms of
    /// both operands are in `[MIN/ε, ε/MIN]`, the product `a·conj(b)`, at most `|a|·|b|` in
    /// each part, neither overflows nor loses its digits, and the quotient is that product divided
    /// by `|b|²`; elsewhere [`over_cold`](Self::over_cold) scales the operands first.
    #[inline]
    fn over(a: Self, b: Self) -> Self {
        let sb = b.norm_sqr();
        if in_range(a.norm_sqr()) && in_range(sb) {
            let p = Self::hamilton(a, b.conj());
            Quaternion::new(p.w / sb, p.x / sb, p.y / sb, p.z / sb)
        } else {
            Self::over_cold(a, b)
        }
    }

    /// The quotient at the ends of the range and of special values. Finite operands are each
    /// scaled by a power of two that brings their largest part into `[1, 2)`, exactly; the
    /// quotient of the scaled operands, whose parts are below 4, is then scaled back by the
    /// difference of the powers, rounded once. A zero dividend gives zeros.
    ///
    /// The special values follow the limits along the direction of the infinite operand: a NaN
    /// part in either gives NaN in every part; a zero divisor gives the parts of the dividend
    /// divided by its real part, a signed zero, as floats are divided; a finite dividend over a
    /// divisor with an infinite part gives zeros, with the signs of the quotient by the
    /// divisor's direction (its infinite parts as `±1`, the others 0); an infinite dividend over
    /// a finite nonzero divisor gives `±∞` where the quotient of the dividend's direction is not
    /// zero and that zero elsewhere; both infinite give NaN.
    #[cold]
    #[inline(never)]
    pub(super) fn over_cold(a: Self, b: Self) -> Self {
        if a.is_nan() || b.is_nan() {
            return Self::NAN;
        }
        let (fa, fb) = (a.is_finite(), b.is_finite());
        if !fb {
            return if fa {
                Self::times_real(Self::hamilton(a, b.infinite_direction().conj()), T::ZERO)
            } else {
                Self::NAN
            };
        }
        if b.is_zero() {
            return Self::over_real(a, b.w);
        }
        if !fa {
            let d = Self::over_cold(a.infinite_direction(), b);
            let inf = |t: T| {
                if t == T::ZERO {
                    t
                } else {
                    T::INFINITY.copysign(t)
                }
            };
            return Quaternion::new(inf(d.w), inf(d.x), inf(d.y), inf(d.z));
        }
        let ka = if a.is_zero() {
            0
        } else {
            exponent(a.max_abs())
        };
        let kb = exponent(b.max_abs());
        let (a, b) = (a.ldexp(-ka), b.ldexp(-kb));
        let s = b.norm_sqr();
        let p = Self::hamilton(a, b.conj());
        Quaternion::new(p.w / s, p.x / s, p.y / s, p.z / s).ldexp(ka - kb)
    }

    /// The direction of a quaternion with an infinite part: `±1` for its infinite parts, `±0` for
    /// the others.
    pub(super) fn infinite_direction(self) -> Self {
        let d = |t: T| {
            if t.is_infinite() {
                T::ONE.copysign(t)
            } else {
                T::ZERO.copysign(t)
            }
        };
        Quaternion::new(d(self.w), d(self.x), d(self.y), d(self.z))
    }

    #[inline]
    fn over_real(a: Self, b: T) -> Self {
        Quaternion::new(a.w / b, a.x / b, a.y / b, a.z / b)
    }

    #[inline]
    fn real_over(a: T, b: Self) -> Self {
        Self::over(Quaternion::from(a), b)
    }
}

/// An operator between two quaternions, and between a quaternion and a float of its type on the
/// right, owned or borrowed, with its assignments.
macro_rules! operators {
    ($op:ident, $method:ident, $assign:ident, $assign_method:ident, $both:ident, $real:ident) => {
        impl<T: Float> $op for Quaternion<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Quaternion::$both(self, rhs)
            }
        }
        impl<T: Float> $op<&Quaternion<T>> for Quaternion<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                Quaternion::$both(self, *rhs)
            }
        }
        impl<T: Float> $op<Quaternion<T>> for &Quaternion<T> {
            type Output = Quaternion<T>;
            #[inline]
            fn $method(self, rhs: Quaternion<T>) -> Quaternion<T> {
                Quaternion::$both(*self, rhs)
            }
        }
        impl<T: Float> $op<&Quaternion<T>> for &Quaternion<T> {
            type Output = Quaternion<T>;
            #[inline]
            fn $method(self, rhs: &Quaternion<T>) -> Quaternion<T> {
                Quaternion::$both(*self, *rhs)
            }
        }
        impl<T: Float> $op<T> for Quaternion<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: T) -> Self {
                Quaternion::$real(self, rhs)
            }
        }
        impl<T: Float> $op<&T> for Quaternion<T> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &T) -> Self {
                Quaternion::$real(self, *rhs)
            }
        }
        impl<T: Float> $op<T> for &Quaternion<T> {
            type Output = Quaternion<T>;
            #[inline]
            fn $method(self, rhs: T) -> Quaternion<T> {
                Quaternion::$real(*self, rhs)
            }
        }
        impl<T: Float> $op<&T> for &Quaternion<T> {
            type Output = Quaternion<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Quaternion<T> {
                Quaternion::$real(*self, *rhs)
            }
        }
        impl<T: Float> $assign for Quaternion<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = Quaternion::$both(*self, rhs);
            }
        }
        impl<T: Float> $assign<&Quaternion<T>> for Quaternion<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &Self) {
                *self = Quaternion::$both(*self, *rhs);
            }
        }
        impl<T: Float> $assign<T> for Quaternion<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
                *self = Quaternion::$real(*self, rhs);
            }
        }
        impl<T: Float> $assign<&T> for Quaternion<T> {
            #[inline]
            fn $assign_method(&mut self, rhs: &T) {
                *self = Quaternion::$real(*self, *rhs);
            }
        }
    };
}

operators!(Add, add, AddAssign, add_assign, plus, plus_real);
operators!(Sub, sub, SubAssign, sub_assign, minus, minus_real);
operators!(Mul, mul, MulAssign, mul_assign, times, times_real);
operators!(Div, div, DivAssign, div_assign, over, over_real);

/// An operator with a float on the left of a quaternion of its type, which the orphan rule leaves
/// to one implementation for each float type.
macro_rules! left {
    ($t:ident, $op:ident, $method:ident, $left:ident) => {
        impl $op<Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            #[inline]
            fn $method(self, rhs: Quaternion<$t>) -> Quaternion<$t> {
                Quaternion::$left(self, rhs)
            }
        }
        impl $op<&Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            #[inline]
            fn $method(self, rhs: &Quaternion<$t>) -> Quaternion<$t> {
                Quaternion::$left(self, *rhs)
            }
        }
        impl $op<Quaternion<$t>> for &$t {
            type Output = Quaternion<$t>;
            #[inline]
            fn $method(self, rhs: Quaternion<$t>) -> Quaternion<$t> {
                Quaternion::$left(*self, rhs)
            }
        }
        impl $op<&Quaternion<$t>> for &$t {
            type Output = Quaternion<$t>;
            #[inline]
            fn $method(self, rhs: &Quaternion<$t>) -> Quaternion<$t> {
                Quaternion::$left(*self, *rhs)
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

impl<T: Float> Neg for Quaternion<T> {
    type Output = Self;

    /// The negation of every part.
    #[inline]
    fn neg(self) -> Self {
        Quaternion::new(-self.w, -self.x, -self.y, -self.z)
    }
}

impl<T: Float> Neg for &Quaternion<T> {
    type Output = Quaternion<T>;

    /// The negation of every part.
    #[inline]
    fn neg(self) -> Quaternion<T> {
        -*self
    }
}

impl<T: Float> Sum for Quaternion<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |s, q| s + q)
    }
}

impl<'a, T: Float> Sum<&'a Quaternion<T>> for Quaternion<T> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |s, q| s + q)
    }
}

impl<T: Float> Product for Quaternion<T> {
    /// The product in the order of the iterator, `q₁·q₂·…·qₙ`, multiplied from the left.
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |p, q| p * q)
    }
}

impl<'a, T: Float> Product<&'a Quaternion<T>> for Quaternion<T> {
    /// The product in the order of the iterator, `q₁·q₂·…·qₙ`, multiplied from the left.
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |p, q| p * q)
    }
}
