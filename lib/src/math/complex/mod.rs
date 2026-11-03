//! Complex numbers: a [`Complex<T>`] is `re + im·i` with parts of the [`Float`] type `T`, `f32` or
//! `f64`.
//!
//! They add with `+` and `+=`, subtract with `-` and `-=`, multiply with `*` and `*=` and divide
//! with `/` and `/=`, each operand owned or borrowed, and take a float of their own type as an
//! operand on either side. They negate with `-`, compare with `==`, add up with `Sum` and multiply
//! with `Product`, and are written as `1+2i` by `Display`.
//!
//! [`Complex::abs`] and [`Complex::arg`] give the polar form, [`Complex::from_polar`] builds a
//! number from it, and [`Complex::conj`], [`Complex::norm_sqr`] and [`Complex::recip`] the rest
//! of the arithmetic. [`Complex::exp`], [`Complex::ln`], [`Complex::sqrt`], [`Complex::powi`],
//! [`Complex::powf`] and [`Complex::powc`] are the principal values, with the branch cut of the
//! logarithm and the square root along the negative real axis, where the sign of a zero imaginary
//! part picks the side, as in C99's `<complex.h>`.

mod float;
mod ops;

pub use float::Float;

use core::fmt;

/// The complex number `re + im·i`.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
///
/// let z = Complex::new(3.0, 4.0);
/// assert_eq!(z.abs(), 5.0);
/// assert_eq!(z * z.conj(), Complex::new(25.0, 0.0));
/// assert_eq!(Complex::new(-4.0, 0.0).sqrt(), Complex::new(0.0, 2.0));
/// assert_eq!(format!("{}", z - 2.0 * Complex::I), "3+2i");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T> Complex<T> {
    /// The number `re + im·i`.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }
}

impl<T: Float> Complex<T> {
    /// Zero.
    pub const ZERO: Self = Complex::new(T::ZERO, T::ZERO);
    /// One.
    pub const ONE: Self = Complex::new(T::ONE, T::ZERO);
    /// The imaginary unit `i`.
    pub const I: Self = Complex::new(T::ZERO, T::ONE);

    /// The number with the absolute value `r` and the argument `theta`: `r·e^(i·theta)`.
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        let (sin, cos) = theta.sin_cos();
        Complex::new(r * cos, r * sin)
    }

    /// The absolute value and the argument, as [`abs`](Self::abs) and [`arg`](Self::arg) give
    /// them.
    #[inline]
    pub fn to_polar(self) -> (T, T) {
        (self.abs(), self.arg())
    }

    /// The complex conjugate, `re - im·i`.
    #[inline]
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    /// The absolute value `√(re² + im²)`, without an overflow or underflow in the squares.
    #[inline]
    pub fn abs(self) -> T {
        self.re.hypot(self.im)
    }

    /// The square of the absolute value, `re² + im²`: cheaper than `abs`, but it overflows and
    /// underflows where the squares do.
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// The argument, the angle from the positive real axis, in `(-π, π]`; `-π` for a negative
    /// real number with a negative zero imaginary part.
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// `1 / self`.
    #[inline]
    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Whether both parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Whether either part is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// `e^self`: `e^re·(cos im + i·sin im)`. A real number stays real, with the sign of its zero
    /// imaginary part.
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        if self.im == T::ZERO {
            return Complex::new(r, self.im);
        }
        let (sin, cos) = self.im.sin_cos();
        Complex::new(r * cos, r * sin)
    }

    /// The principal natural logarithm, `ln |self| + i·arg(self)`, whose imaginary part is in
    /// `(-π, π]`. The logarithm of zero is `-∞`. Near the unit circle the real part is
    /// `ln(1 + (|self|² - 1)) / 2`, so that it keeps its digits where `ln |self|` is small.
    pub fn ln(self) -> Self {
        let (a, b) = (self.re.abs(), self.im.abs());
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        let re = if big > T::HALF && big < T::TWO {
            ((big - T::ONE) * (big + T::ONE) + small * small).ln_1p() * T::HALF
        } else {
            self.abs().ln()
        };
        Complex::new(re, self.arg())
    }

    /// The principal square root, whose real part is not negative, by Kahan's method, which
    /// neither overflows nor cancels: for `re ≥ 0`, `t = √((re + |self|) / 2)` and the root
    /// `t + i·im / (2t)`, and the other way round for `re < 0`. Infinities and NaNs give what
    /// C99's `csqrt` gives.
    pub fn sqrt(self) -> Self {
        let (re, im) = (self.re, self.im);
        if re == T::ZERO && im == T::ZERO {
            return Complex::new(T::ZERO, im);
        }
        if im.is_infinite() {
            return Complex::new(T::INFINITY, im);
        }
        if re.is_infinite() {
            let zero_or_nan = if im.is_nan() {
                im
            } else {
                T::ZERO.copysign(im)
            };
            return if re > T::ZERO {
                Complex::new(re, zero_or_nan)
            } else {
                Complex::new(zero_or_nan.abs(), T::INFINITY.copysign(im))
            };
        }
        if re.is_nan() || im.is_nan() {
            return Complex::new(T::NAN, T::NAN);
        }
        let big = re.abs().max(im.abs());
        if big < T::from_f64(4.0) * T::MIN_POSITIVE {
            // Near the subnormals: scaled up by 2^64, exactly, and the root back by 2^32.
            let up = T::from_f64(18_446_744_073_709_551_616.0);
            return (self * up).sqrt() / T::from_f64(4_294_967_296.0);
        }
        if big > T::MAX / T::from_f64(4.0) {
            return (self * T::from_f64(0.25)).sqrt() * T::TWO;
        }
        let t = ((re.abs() + self.abs()) * T::HALF).sqrt();
        if re >= T::ZERO {
            Complex::new(t, im / (T::TWO * t))
        } else {
            Complex::new(im.abs() / (T::TWO * t), t.copysign(im))
        }
    }

    /// `self` to the integer power `n`, by repeated squaring: exact as long as the products are.
    /// A negative power is the power of the reciprocal, and `0^0` is 1.
    pub fn powi(self, n: i32) -> Self {
        let mut base = if n < 0 { self.recip() } else { self };
        let mut n = n.unsigned_abs();
        let mut r = Self::ONE;
        while n > 0 {
            if n & 1 == 1 {
                r *= base;
            }
            n >>= 1;
            if n > 0 {
                base *= base;
            }
        }
        r
    }

    /// The principal value of `self` to the real power `x`: `e^(x·ln self)`, with `0^x = 0` for
    /// `x ≠ 0` and `0^0 = 1`.
    pub fn powf(self, x: T) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return if x == T::ZERO { Self::ONE } else { Self::ZERO };
        }
        (self.ln() * x).exp()
    }

    /// The principal value of `self` to the complex power `w`: `e^(w·ln self)`, with `0^w = 0`
    /// for `w ≠ 0` and `0^0 = 1`.
    pub fn powc(self, w: Self) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return if w == Self::ZERO {
                Self::ONE
            } else {
                Self::ZERO
            };
        }
        (self.ln() * w).exp()
    }
}

impl<T: Float> From<T> for Complex<T> {
    /// The real number `re + 0·i`.
    #[inline]
    fn from(re: T) -> Self {
        Complex::new(re, T::ZERO)
    }
}

impl<T: Float> fmt::Display for Complex<T> {
    /// Writes `re+imi` or `re-imi`, each part as `Display` writes the float, with the precision
    /// given: `1.5-2i`, or `1.50-2.00i` with `{:.2}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.im.is_sign_negative() { '-' } else { '+' };
        match f.precision() {
            Some(p) => write!(f, "{:.*}{}{:.*}i", p, self.re, sign, p, self.im.abs()),
            None => write!(f, "{}{}{}i", self.re, sign, self.im.abs()),
        }
    }
}
