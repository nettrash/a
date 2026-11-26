//! The operators of quaternions.

use super::Quaternion;
use super::scale::{exponent, in_range};
use crate::math::complex::Float;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use core::any::Any;
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
    /// operands. On aarch64 the four parts are the lanes of NEON vectors (one for `f32`, two for
    /// `f64`), the same products added in the same order, and the test is of their sum.
    #[inline]
    fn times(l: Self, r: Self) -> Self {
        // On aarch64: the same product in the lanes of NEON vectors, bit for bit.
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if let Some(p) = Self::times_neon(l, r) {
            return p;
        }
        let p = Self::hamilton(l, r);
        // The sum of the parts is finite only if they all are: one test for the common case, the
        // rest out of line.
        if (p.w + p.x + p.y + p.z).is_finite() {
            p
        } else {
            Self::times_unusual(l, r, p)
        }
    }

    /// The product by [`neon`] for `f32` and `f64`, `None` where a part is not finite or for
    /// another type (there is none: the trait is sealed). The types are known at compilation, so
    /// the checks of the type fold away.
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    #[inline(always)]
    fn times_neon(l: Self, r: Self) -> Option<Self> {
        let (a, b) = (&l as &dyn Any, &r as &dyn Any);
        if let (Some(&a), Some(&b)) = (
            a.downcast_ref::<Quaternion<f32>>(),
            b.downcast_ref::<Quaternion<f32>>(),
        ) {
            let p = neon::times_f32(a, b)?;
            return (&p as &dyn Any).downcast_ref::<Self>().copied();
        }
        if let (Some(&a), Some(&b)) = (
            a.downcast_ref::<Quaternion<f64>>(),
            b.downcast_ref::<Quaternion<f64>>(),
        ) {
            let p = neon::times_f64(a, b)?;
            return (&p as &dyn Any).downcast_ref::<Self>().copied();
        }
        None
    }

    #[cold]
    #[inline(never)]
    fn times_unusual(l: Self, r: Self, p: Self) -> Self {
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

/// Hamilton's product in the lanes of NEON vectors: four `f32` parts in one, four `f64` parts in
/// two.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod neon {
    use super::Quaternion;
    use core::arch::aarch64::{
        float32x4_t, float64x2_t, vaddq_f32, vaddq_f64, vaddvq_f32, vaddvq_f64, vextq_f32,
        vextq_f64, vmulq_f32, vmulq_f64, vmulq_laneq_f32, vmulq_laneq_f64, vnegq_f64, vrev64q_f32,
    };
    use core::mem::transmute;

    /// `l·r` in the lanes `(w, x, y, z)` as `l.w·r + l.x·(−r.x, r.w, −r.z, r.y) +
    /// l.y·(−r.y, r.z, r.w, −r.x) + l.z·(−r.z, −r.y, r.x, r.w)`, added in that order: each lane
    /// takes the products of [`Quaternion::hamilton`] in its order, so the result is the same
    /// bit for bit, with no fused operation. `None` where the sum of the parts is not finite,
    /// which it is when they all are but for an overflow of the sum, for the generic path to take
    /// the product again.
    #[inline(always)]
    pub(super) fn times_f32(l: Quaternion<f32>, r: Quaternion<f32>) -> Option<Quaternion<f32>> {
        const SIGNS: [[f32; 4]; 3] = [
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ];
        // SAFETY: NEON is part of the baseline of the aarch64 targets. `Quaternion<f32>` is
        // `#[repr(C)]` with four `f32` fields, 16 bytes like `float32x4_t` and `[f32; 4]`, and
        // every bit pattern is valid for all three, so the transmutes move the parts in order into
        // the lanes and back. The lane indices are constants in range.
        unsafe {
            let lv = transmute::<Quaternion<f32>, float32x4_t>(l);
            let rv = transmute::<Quaternion<f32>, float32x4_t>(r);
            let sign = |k: usize| transmute::<[f32; 4], float32x4_t>(SIGNS[k]);
            // (x, w, z, y), (y, z, w, x) and (z, y, x, w) of r, with their signs.
            let r_xwzy = vmulq_f32(vrev64q_f32(rv), sign(0));
            let yzwx = vextq_f32::<2>(rv, rv);
            let r_yzwx = vmulq_f32(yzwx, sign(1));
            let r_zyxw = vmulq_f32(vrev64q_f32(yzwx), sign(2));
            let mut p = vmulq_laneq_f32::<0>(rv, lv);
            p = vaddq_f32(p, vmulq_laneq_f32::<1>(r_xwzy, lv));
            p = vaddq_f32(p, vmulq_laneq_f32::<2>(r_yzwx, lv));
            p = vaddq_f32(p, vmulq_laneq_f32::<3>(r_zyxw, lv));
            vaddvq_f32(p)
                .is_finite()
                .then(|| transmute::<float32x4_t, Quaternion<f32>>(p))
        }
    }

    /// The same for `f64`, the parts `(w, x)` in one vector and `(y, z)` in the other.
    #[inline(always)]
    pub(super) fn times_f64(l: Quaternion<f64>, r: Quaternion<f64>) -> Option<Quaternion<f64>> {
        // SAFETY: as in `times_f32`: `Quaternion<f64>` is `#[repr(C)]` with four `f64` fields,
        // 32 bytes like two `float64x2_t` and `[f64; 2]` pairs, every bit pattern valid.
        unsafe {
            let [l01, l23] = transmute::<Quaternion<f64>, [float64x2_t; 2]>(l);
            let [r01, r23] = transmute::<Quaternion<f64>, [float64x2_t; 2]>(r);
            let minus_plus = transmute::<[f64; 2], float64x2_t>([-1.0, 1.0]);
            let plus_minus = transmute::<[f64; 2], float64x2_t>([1.0, -1.0]);
            // (x, w) and (z, y) of r.
            let r10 = vextq_f64::<1>(r01, r01);
            let r32 = vextq_f64::<1>(r23, r23);
            let mut p01 = vmulq_laneq_f64::<0>(r01, l01);
            let mut p23 = vmulq_laneq_f64::<0>(r23, l01);
            // l.x·(−r.x, r.w | −r.z, r.y)
            p01 = vaddq_f64(p01, vmulq_laneq_f64::<1>(vmulq_f64(r10, minus_plus), l01));
            p23 = vaddq_f64(p23, vmulq_laneq_f64::<1>(vmulq_f64(r32, minus_plus), l01));
            // l.y·(−r.y, r.z | r.w, −r.x)
            p01 = vaddq_f64(p01, vmulq_laneq_f64::<0>(vmulq_f64(r23, minus_plus), l23));
            p23 = vaddq_f64(p23, vmulq_laneq_f64::<0>(vmulq_f64(r01, plus_minus), l23));
            // l.z·(−r.z, −r.y | r.x, r.w)
            p01 = vaddq_f64(p01, vmulq_laneq_f64::<1>(vnegq_f64(r32), l23));
            p23 = vaddq_f64(p23, vmulq_laneq_f64::<1>(r10, l23));
            vaddvq_f64(vaddq_f64(p01, p23))
                .is_finite()
                .then(|| transmute::<[float64x2_t; 2], Quaternion<f64>>([p01, p23]))
        }
    }
}
