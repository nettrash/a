//! Quaternions: a [`Quaternion<T>`] is `w + x·i + y·j + z·k` with parts of the float type `T`,
//! `f32` or `f64`, and Hamilton's rules `i² = j² = k² = ijk = −1`.
//!
//! They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` (Hamilton's
//! product, which is not commutative) and divide with `/` and `/=` on the right, `a / b = a·b⁻¹`,
//! each operand owned or borrowed, and take a float of their own type as an operand on either
//! side. They negate with `-`, compare with `==`, add up with `Sum` and multiply with `Product`,
//! and are written as `1+2i+3j+4k` by `Display`.
//!
//! [`Quaternion::norm`] is the length without overflow or underflow in its squares, and
//! [`Quaternion::normalize`], [`Quaternion::recip`] and the quotient keep their digits at the ends
//! of the range by scaling by a power of two. In a product of finite quaternions that overflows,
//! the parts the plain formula gives finite are kept, and the others are taken from operands
//! scaled by powers of two, so that every part is within a few units of `ε` times the sum of the
//! absolute values of its four products, as in any product, and those that overflow by more are
//! `±∞`, rather than NaNs from `∞ − ∞`. A power of a finite quaternion that overflows is formed
//! again from the scaled base: accurate normwise, with no NaN part.
//!
//! [`Quaternion::exp`], [`Quaternion::ln`], [`Quaternion::sqrt`], [`Quaternion::powi`],
//! [`Quaternion::powf`] and [`Quaternion::powq`] are the principal values. On the complex numbers
//! `w + x·i` they, and the arithmetic, agree with [`Complex`]'s to rounding wherever the operands
//! and the results are finite; at infinities and NaNs, and where a result overflows, the special
//! values stated with each function differ from `Complex`'s in places: a NaN part makes every
//! part NaN (where `Complex` follows C99 and may keep an infinity beside it), `1/(−0)` is `−∞`
//! (`Complex` gives `+∞`), the reciprocal of a quaternion with an infinite part and no NaN is 0,
//! an overflowing product or power of finite operands has no NaN part, `powi` of zero and of
//! infinities differs (`0⁻¹` is `∞ + NaN·i` here, the reciprocal, and `NaN + NaN·i` in
//! `Complex`; `∞¹` is `∞` here and `∞ + NaN·i` there, where the power starts from 1), and `0^t`
//! is `+∞` for `t < 0`.
//!
//! A unit quaternion `q` is a rotation of three-dimensional space, `v ↦ q·v·q⁻¹` for a vector `v`
//! read as the quaternion `0 + v`: the active rotation by the angle `θ` about the unit axis `u`
//! for `q = cos(θ/2) + u·sin(θ/2)`, counterclockwise when the axis points at the viewer, and `q`
//! and `−q` are the same rotation. The product composes rotations, the right one first:
//! `(a·b).rotate(v) = a.rotate(b.rotate(v))`. The methods that read a quaternion as a rotation
//! read it as `q/|q|`, whatever its norm, and give NaN for the zero quaternion and for one with a
//! part that is not finite. A rotation converts to and from an axis and an angle, a rotation
//! vector, a rotation matrix (Shepperd's method), and Euler angles in all twelve sequences,
//! intrinsic or extrinsic, with the conventions of SciPy's `Rotation` (the method of Bernardes
//! and Viollet); it rotates vectors, interpolates by slerp (Shoemake) and nlerp, and measures the
//! angle to another.
//!
//! Nothing here is constant-time.

mod euler;
mod functions;
mod ops;
mod rotation;
mod scale;

pub use euler::{EulerFrame, EulerSequence};

use crate::math::complex::{Complex, Float};
use core::fmt;
use scale::in_range;

/// The quaternion `w + x·i + y·j + z·k`.
///
/// # Examples
///
/// ```
/// use a::math::quaternion::Quaternion;
///
/// let i = Quaternion::<f64>::I;
/// let j = Quaternion::<f64>::J;
/// assert_eq!(i * j, Quaternion::K);
/// assert_eq!(j * i, -Quaternion::K);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Quaternion<T> {
    /// The real (scalar) part.
    pub w: T,
    /// The coefficient of `i`.
    pub x: T,
    /// The coefficient of `j`.
    pub y: T,
    /// The coefficient of `k`.
    pub z: T,
}

impl<T> Quaternion<T> {
    /// The quaternion `w + x·i + y·j + z·k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!((q.w, q.x, q.y, q.z), (1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Quaternion { w, x, y, z }
    }
}

impl<T: Float> Quaternion<T> {
    /// Zero.
    pub const ZERO: Self = Quaternion::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    /// One.
    pub const ONE: Self = Quaternion::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);
    /// The unit `i`.
    pub const I: Self = Quaternion::new(T::ZERO, T::ONE, T::ZERO, T::ZERO);
    /// The unit `j`.
    pub const J: Self = Quaternion::new(T::ZERO, T::ZERO, T::ONE, T::ZERO);
    /// The unit `k`.
    pub const K: Self = Quaternion::new(T::ZERO, T::ZERO, T::ZERO, T::ONE);

    /// The quaternion `scalar + vector[0]·i + vector[1]·j + vector[2]·k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::from_parts(1.0, [2.0, 3.0, 4.0]);
    /// assert_eq!(q, Quaternion::new(1.0, 2.0, 3.0, 4.0));
    /// assert_eq!((q.scalar(), q.vector()), (1.0, [2.0, 3.0, 4.0]));
    /// ```
    #[inline]
    pub const fn from_parts(scalar: T, vector: [T; 3]) -> Self {
        Quaternion::new(scalar, vector[0], vector[1], vector[2])
    }

    /// The scalar (real) part, `w`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(Quaternion::new(1.0, 2.0, 3.0, 4.0).scalar(), 1.0);
    /// ```
    #[inline]
    pub fn scalar(self) -> T {
        self.w
    }

    /// The vector part, `[x, y, z]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(Quaternion::new(1.0, 2.0, 3.0, 4.0).vector(), [2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn vector(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }

    /// The complex number `w + x·i` when the parts of `j` and `k` are zeros (of either sign),
    /// else `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(
    ///     Quaternion::new(1.0, 2.0, 0.0, -0.0).to_complex(),
    ///     Some(Complex::new(1.0, 2.0))
    /// );
    /// assert_eq!(Quaternion::new(1.0, 2.0, 3.0, 0.0).to_complex(), None);
    /// ```
    #[inline]
    pub fn to_complex(self) -> Option<Complex<T>> {
        if self.y == T::ZERO && self.z == T::ZERO {
            Some(Complex::new(self.w, self.x))
        } else {
            None
        }
    }

    /// The conjugate, `w − x·i − y·j − z·k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.conj(), Quaternion::new(1.0, -2.0, -3.0, -4.0));
    /// ```
    #[inline]
    pub fn conj(self) -> Self {
        Quaternion::new(self.w, -self.x, -self.y, -self.z)
    }

    /// The square of the norm, `w² + x² + y² + z²`: cheaper than [`norm`](Self::norm), but it
    /// overflows and underflows where the squares do.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(Quaternion::new(1.0, 2.0, 3.0, 4.0).norm_sqr(), 30.0);
    /// ```
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// The norm `√(w² + x² + y² + z²)`, without an overflow or underflow in the squares: where the
    /// squared norm is outside `[MIN/ε, ε/MIN]` the parts are first scaled by a power of two,
    /// exactly, that brings the largest into `[1, 2)`. An infinite part gives `∞`, even beside a
    /// NaN, as `hypot` does; otherwise a NaN part gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(Quaternion::new(1.0, 2.0, 2.0, 4.0).norm(), 5.0);
    /// // The squares would overflow, and underflow.
    /// let (big, tiny) = (2f64.powi(1000), f64::MIN_POSITIVE / 2f64.powi(38)); // 2¹⁰⁰⁰, 2⁻¹⁰⁶⁰
    /// assert_eq!(Quaternion::new(3.0 * big, 0.0, 4.0 * big, 0.0).norm(), 5.0 * big);
    /// assert_eq!(Quaternion::new(0.0, 3.0 * tiny, 0.0, 4.0 * tiny).norm(), 5.0 * tiny);
    /// ```
    #[inline]
    pub fn norm(self) -> T {
        let s = self.norm_sqr();
        if in_range(s) {
            s.sqrt()
        } else {
            self.norm_cold()
        }
    }

    #[cold]
    #[inline(never)]
    fn norm_cold(self) -> T {
        if !self.is_finite() {
            return self.w.hypot(self.x).hypot(self.y.hypot(self.z));
        }
        if self.is_zero() {
            return T::ZERO;
        }
        let k = scale::exponent(self.max_abs());
        scale::ldexp(self.ldexp(-k).norm_sqr().sqrt(), k)
    }

    /// The unit quaternion `self / |self|`, its digits kept at the ends of the range by the
    /// scaling of [`norm`](Self::norm). Zero, and a quaternion with a part that is infinite or
    /// NaN, give NaN in every part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(0.0, 3.0, 0.0, 4.0);
    /// assert_eq!(q.normalize(), Quaternion::new(0.0, 0.6, 0.0, 0.8));
    /// // Subnormal parts, whose squares vanish.
    /// let tiny = f64::MIN_POSITIVE / 2f64.powi(38); // 2⁻¹⁰⁶⁰
    /// assert_eq!((q * tiny).normalize(), Quaternion::new(0.0, 0.6, 0.0, 0.8));
    /// assert!(Quaternion::<f64>::ZERO.normalize().w.is_nan());
    /// ```
    #[inline]
    pub fn normalize(self) -> Self {
        let s = self.norm_sqr();
        if in_range(s) {
            let n = s.sqrt();
            Quaternion::new(self.w / n, self.x / n, self.y / n, self.z / n)
        } else {
            self.normalize_cold()
        }
    }

    #[cold]
    #[inline(never)]
    fn normalize_cold(self) -> Self {
        if !self.is_finite() || self.is_zero() {
            return Self::NAN;
        }
        let q = self.ldexp(-scale::exponent(self.max_abs()));
        let n = q.norm_sqr().sqrt();
        Quaternion::new(q.w / n, q.x / n, q.y / n, q.z / n)
    }

    /// `self` divided by its norm, as [`normalize`](Self::normalize) gives it, with a unit
    /// quaternion returned as it is: dividing by a norm of exactly 1 changes nothing. (Even with
    /// half of random unit quaternions of norm 1 exactly, the test pays for itself.)
    #[inline]
    pub(super) fn unit(self) -> Self {
        if self.norm_sqr() == T::ONE {
            self
        } else {
            self.normalize()
        }
    }

    /// The inverse `self⁻¹ = conj(self) / |self|²`, with `self·self⁻¹ = self⁻¹·self = 1`; it is
    /// `1 / self`, with the scaling and the special values of the quotient: the reciprocal of a
    /// zero has the real part `1/w` (an infinity of the sign of the zero `w`) and NaN for the
    /// others, and that of a quaternion with an infinite part, and no NaN, is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
    /// assert_eq!(q.recip(), Quaternion::new(0.25, -0.25, -0.25, -0.25));
    /// assert_eq!(q * q.recip(), Quaternion::ONE);
    /// // |q|² = 2^1202 would overflow.
    /// let big = 2f64.powi(600);
    /// assert_eq!((q * big).recip(), q.recip() / big);
    /// ```
    #[inline]
    pub fn recip(self) -> Self {
        let s = self.norm_sqr();
        if in_range(s) {
            Quaternion::new(self.w / s, -self.x / s, -self.y / s, -self.z / s)
        } else {
            Self::over_cold(Self::ONE, self)
        }
    }

    /// The dot product of the parts, `w·w' + x·x' + y·y' + z·z'`; for unit quaternions, the cosine
    /// of half the angle between the rotations, up to sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let p = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(p.dot(Quaternion::new(1.0, 0.0, -1.0, 1.0)), 2.0);
    /// assert_eq!(p.dot(p), p.norm_sqr());
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Whether every part is finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert!(Quaternion::new(1.0, 2.0, 3.0, 4.0).is_finite());
    /// assert!(!Quaternion::new(1.0, f64::INFINITY, 3.0, 4.0).is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Whether a part is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert!(Quaternion::new(1.0, 2.0, f64::NAN, 4.0).is_nan());
    /// assert!(!Quaternion::new(1.0, 2.0, 3.0, 4.0).is_nan());
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        self.w.is_nan() || self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl<T: Float> From<T> for Quaternion<T> {
    /// The real quaternion `w + 0·i + 0·j + 0·k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(Quaternion::from(2.5), Quaternion::new(2.5, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    fn from(w: T) -> Self {
        Quaternion::new(w, T::ZERO, T::ZERO, T::ZERO)
    }
}

impl<T: Float> From<Complex<T>> for Quaternion<T> {
    /// The quaternion `re + im·i + 0·j + 0·k`: the complex numbers are the subalgebra of the
    /// quaternions spanned by `1` and `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::from(Complex::new(1.0, 2.0));
    /// assert_eq!(q, Quaternion::new(1.0, 2.0, 0.0, 0.0));
    /// ```
    #[inline]
    fn from(z: Complex<T>) -> Self {
        Quaternion::new(z.re, z.im, T::ZERO, T::ZERO)
    }
}

impl<T: Float> From<[T; 4]> for Quaternion<T> {
    /// The quaternion of the parts `[w, x, y, z]`, the scalar first, as the fields are. SciPy,
    /// glam and nalgebra's coordinates put it last, `[x, y, z, w]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(q, Quaternion::new(1.0, 2.0, 3.0, 4.0));
    /// assert_eq!(<[f64; 4]>::from(q), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    fn from(p: [T; 4]) -> Self {
        Quaternion::new(p[0], p[1], p[2], p[3])
    }
}

impl<T: Float> From<Quaternion<T>> for [T; 4] {
    /// The parts `[w, x, y, z]`, the scalar first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// assert_eq!(<[f64; 4]>::from(Quaternion::new(1.0, 2.0, 3.0, 4.0)), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        [q.w, q.x, q.y, q.z]
    }
}

impl<T: Float> fmt::Display for Quaternion<T> {
    /// Writes `w+xi+yj+zk`, each sign `-` for a part with its sign bit set, each part as
    /// `Display` writes the float, with the precision given: `1-2i+3j+4.5k`, or
    /// `1.00-2.00i+3.00j+4.50k` with `{:.2}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, -2.0, 3.0, 4.5);
    /// assert_eq!(format!("{q}"), "1-2i+3j+4.5k");
    /// assert_eq!(format!("{q:.2}"), "1.00-2.00i+3.00j+4.50k");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = |t: T| if t.is_sign_negative() { '-' } else { '+' };
        let (sx, sy, sz) = (sign(self.x), sign(self.y), sign(self.z));
        let (x, y, z) = (self.x.abs(), self.y.abs(), self.z.abs());
        match f.precision() {
            Some(p) => write!(
                f,
                "{:.*}{sx}{:.*}i{sy}{:.*}j{sz}{:.*}k",
                p, self.w, p, x, p, y, p, z
            ),
            None => write!(f, "{}{sx}{x}i{sy}{y}j{sz}{z}k", self.w),
        }
    }
}
