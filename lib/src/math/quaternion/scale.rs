//! Exact scaling by powers of two, which keeps the squares of the parts in the range of the type
//! where a quaternion's parts are very large or very small, and the rare paths that need it.

use super::Quaternion;
use crate::math::complex::Float;

/// `2⁶⁴`, which lifts any subnormal `f32` or `f64` into the normal range.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

/// The unbiased exponent of a positive normal `f64`.
#[inline]
fn biased(b: f64) -> i32 {
    ((b.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// The exponent `e` of a finite nonzero `x`: `2^e ≤ |x| < 2^(e+1)`. Through the `f64` the value
/// converts to exactly, so that one function serves both types.
#[inline]
pub(super) fn exponent<T: Float>(x: T) -> i32 {
    let b = x.to_f64().abs();
    if b < f64::MIN_POSITIVE {
        biased(b * TWO_64) - 64
    } else {
        biased(b)
    }
}

/// `2^e` as an `f64`, for `e` from −1074 (the smallest subnormal) to 1023.
#[inline]
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// The exponents of the largest power of two of the type and of its smallest subnormal: 127 and
/// −149 for `f32`, 1023 and −1074 for `f64`.
#[inline]
fn limits<T: Float>() -> (i32, i32) {
    let emax = exponent(T::MAX);
    let emin = exponent(T::MIN_POSITIVE) - (T::MANTISSA_DIGITS as i32 - 1);
    (emax, emin)
}

/// `x·2^e`, rounded once, for a finite `x`. A power of two of the type is a single product. Above
/// the type's largest power the factor is applied in steps of that power, each exact until the
/// product overflows, which it then does as the exact one would. Below the smallest subnormal
/// it is two halves, each a power of the type: where the first half leaves `x` subnormal the
/// result is far below the smallest subnormal, so that rounding twice gives the zero that
/// rounding once gives.
pub(super) fn ldexp<T: Float>(x: T, e: i32) -> T {
    let (emax, emin) = limits::<T>();
    if (emin..=emax).contains(&e) {
        return x * T::from_f64(pow2(e));
    }
    if e > emax {
        let (mut x, mut e) = (x, e);
        let step = T::from_f64(pow2(emax));
        while e > emax {
            x *= step;
            e -= emax;
        }
        return x * T::from_f64(pow2(e));
    }
    if e < 2 * emin {
        return x * T::ZERO;
    }
    let h = e / 2;
    x * T::from_f64(pow2(h)) * T::from_f64(pow2(e - h))
}

/// The bounds of the squared norm inside which the squares of the parts neither overflow nor lose
/// digits to underflow: `MIN/ε` and its reciprocal, `2^∓970` for `f64` and `2^∓103` for `f32`.
/// Inside them the plain formulas are used; outside, the operands are scaled first.
#[inline]
pub(super) fn in_range<T: Float>(s: T) -> bool {
    s >= T::MIN_POSITIVE / T::EPSILON && s <= T::EPSILON / T::MIN_POSITIVE
}

/// The length of `v`, as [`direction`] gives it, without the direction: `None` for the zero
/// vector, the `hypot` of the parts where one is not finite, the root of the squares where they
/// are in range, else the same scaled exactly by a power of two.
#[inline]
pub(super) fn length<T: Float>(v: [T; 3]) -> Option<T> {
    let [a, b, c] = v;
    let s = a * a + b * b + c * c;
    if in_range(s) {
        return Some(s.sqrt());
    }
    length_cold(v)
}

#[cold]
#[inline(never)]
fn length_cold<T: Float>(v: [T; 3]) -> Option<T> {
    let [a, b, c] = v;
    if a == T::ZERO && b == T::ZERO && c == T::ZERO {
        return None;
    }
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return Some(a.hypot(b).hypot(c));
    }
    let k = exponent(a.abs().max(b.abs()).max(c.abs()));
    let (a, b, c) = (ldexp(a, -k), ldexp(b, -k), ldexp(c, -k));
    Some(ldexp((a * a + b * b + c * c).sqrt(), k))
}

/// The length of a 3-vector without overflow or underflow in the squares, and the unit vector of
/// its direction: `None` for the zero vector, NaN parts for one with a part that is not finite
/// (and the length then `∞` for an infinite part, as `hypot` gives it, or NaN).
pub(super) fn direction<T: Float>(v: [T; 3]) -> Option<([T; 3], T)> {
    let [a, b, c] = v;
    if a == T::ZERO && b == T::ZERO && c == T::ZERO {
        return None;
    }
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return Some(([T::NAN; 3], a.hypot(b).hypot(c)));
    }
    let s = a * a + b * b + c * c;
    if s == T::ONE {
        // A unit vector already: its parts divided by 1 are themselves.
        return Some((v, T::ONE));
    }
    if in_range(s) {
        let r = s.sqrt();
        return Some(([a / r, b / r, c / r], r));
    }
    let k = exponent(a.abs().max(b.abs()).max(c.abs()));
    let (a, b, c) = (ldexp(a, -k), ldexp(b, -k), ldexp(c, -k));
    let r = (a * a + b * b + c * c).sqrt();
    Some(([a / r, b / r, c / r], ldexp(r, k)))
}

impl<T: Float> Quaternion<T> {
    /// The largest absolute value of a part, a NaN part ignored unless all are.
    #[inline]
    pub(super) fn max_abs(self) -> T {
        self.w
            .abs()
            .max(self.x.abs())
            .max(self.y.abs())
            .max(self.z.abs())
    }

    /// Every part times `2^e`, each rounded once.
    #[inline]
    pub(super) fn ldexp(self, e: i32) -> Self {
        Quaternion::new(
            ldexp(self.w, e),
            ldexp(self.x, e),
            ldexp(self.y, e),
            ldexp(self.z, e),
        )
    }

    /// The quaternion as it is when its squared norm is in range, else, when it is finite and
    /// not zero, scaled by a power of two so that its largest part is in `[1, 2)`: the same
    /// rotation, with squares in range. Zero and non-finite quaternions as they are.
    #[inline]
    pub(super) fn balanced(self) -> Self {
        if in_range(self.norm_sqr()) {
            self
        } else {
            self.balanced_cold()
        }
    }

    #[cold]
    #[inline(never)]
    fn balanced_cold(self) -> Self {
        if self.is_finite() && !self.is_zero() {
            self.ldexp(-exponent(self.max_abs()))
        } else {
            self
        }
    }

    /// Whether every part is a zero of either sign.
    #[inline]
    pub(super) fn is_zero(self) -> bool {
        self.w == T::ZERO && self.x == T::ZERO && self.y == T::ZERO && self.z == T::ZERO
    }

    /// NaN in every part.
    pub(super) const NAN: Self = Quaternion::new(T::NAN, T::NAN, T::NAN, T::NAN);
}
