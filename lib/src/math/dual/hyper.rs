//! Hyper-dual numbers (Fike and Alonso, AIAA 2011-886): `a + b·ε₁ + c·ε₂ + d·ε₁ε₂` with
//! `ε₁² = ε₂² = 0` and `ε₁ε₂ ≠ 0`, so that `f(x + ε₁ + ε₂)` carries `f″(x)` in its `ε₁ε₂` part,
//! exactly up to rounding, with no step to choose and nothing to cancel (but for the rule at
//! `0·∞` that [`DualNum`](super::DualNum) states).

use super::functions::{
    atan_slope, coefficient, exponents, huge, inverse_sqrt_one_minus_square,
    inverse_sqrt_square_minus_one, less_one, log_slope, one_minus_square, power, power_f,
    power_from_value, powi_times, sech, times_power, unsigned_zero,
};
use super::num::{DualNum, float, scale, sealed};
use crate::math::complex::Float;
use core::f64::consts::{LN_2, LN_10, LOG2_E};
use core::fmt;

/// The hyper-dual number `re + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂`, with `ε₁² = ε₂² = 0`.
///
/// A function of one variable evaluated at `x + ε₁ + ε₂` gives `f(x) + f′(x)·ε₁ + f′(x)·ε₂ +
/// f″(x)·ε₁ε₂`; one of several variables, with `ε₁` on `xᵢ` and `ε₂` on `xⱼ`, gives
/// `∂²f/∂xᵢ∂xⱼ` in the `ε₁ε₂` part, which is how [`hessian`](super::hessian) forms each entry.
/// A function `f` of the number is `f(a) + f′(a)·(b·ε₁ + c·ε₂) + (f′(a)·d + f″(a)·b·c)·ε₁ε₂`
/// (Fike and Alonso, "The development of hyper-dual numbers for exact second-derivative
/// calculations", AIAA 2011-886, from the Taylor series of `f` about `a`, which ends at the
/// `ε₁ε₂` term), with `f′` and `f″` formed as the methods of
/// [`DualNum`] state; the functions of two numbers, [`atan2`](DualNum::atan2),
/// [`hypot`](DualNum::hypot), [`pow`](DualNum::pow) and [`mul_add`](DualNum::mul_add), take the
/// second partial derivatives in both. The parts are of any [`DualNum`] type, so that
/// `Dual<HyperDual<f64>>` gives third derivatives. `==` and the order compare the values
/// alone. Nothing is constant-time.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, HyperDual};
///
/// // f(x) = x³ at x = 2: f = 8, f′ = 12, f″ = 12; and e^x, whose derivatives are all e^x.
/// let x = HyperDual::new(2.0, 1.0, 1.0, 0.0);
/// let f = x * x * x;
/// assert_eq!((f.re, f.eps1, f.eps2, f.eps1eps2), (8.0, 12.0, 12.0, 12.0));
/// let e = x.exp();
/// assert_eq!((e.eps1, e.eps1eps2), (2.0f64.exp(), 2.0f64.exp()));
/// assert_eq!(format!("{}", HyperDual::new(1.0, -2.0, 0.5, 3.0)), "1-2ε₁+0.5ε₂+3ε₁ε₂");
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct HyperDual<T> {
    /// The value.
    pub re: T,
    /// The coefficient of `ε₁`: the derivative along the first direction.
    pub eps1: T,
    /// The coefficient of `ε₂`: the derivative along the second direction.
    pub eps2: T,
    /// The coefficient of `ε₁ε₂`: the second derivative along the two directions.
    pub eps1eps2: T,
}

impl<T> HyperDual<T> {
    /// The hyper-dual number `re + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::dual::HyperDual;
    ///
    /// let h = HyperDual::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!((h.re, h.eps1, h.eps2, h.eps1eps2), (1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub const fn new(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        HyperDual {
            re,
            eps1,
            eps2,
            eps1eps2,
        }
    }
}

impl<T: DualNum> HyperDual<T> {
    /// The number of the value `re` and zero derivative parts.
    #[inline]
    pub(super) fn from_value(re: T) -> Self {
        HyperDual::new(re, T::ZERO, T::ZERO, T::ZERO)
    }

    /// `f(self)` from the value `f0` and the derivatives `f1` and `f2` of `f` at `self.re`:
    /// `f0 + f1·(b·ε₁ + c·ε₂) + (f1·d + f2·b·c)·ε₁ε₂`.
    #[inline]
    fn chain(self, f0: T, f1: T, f2: T) -> Self {
        let cross = scale(f1, self.eps1eps2) + scale(f2, self.eps1 * self.eps2);
        HyperDual::new(f0, scale(f1, self.eps1), scale(f1, self.eps2), cross)
    }

    /// `f(self, y)` from the value `f0`, the first partial derivatives `fx` and `fy` and the
    /// second `fxx`, `fxy` and `fyy` of `f` at `(self.re, y.re)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn chain2(self, y: Self, f0: T, fx: T, fy: T, fxx: T, fxy: T, fyy: T) -> Self {
        let x = self;
        let eps1 = scale(fx, x.eps1) + scale(fy, y.eps1);
        let eps2 = scale(fx, x.eps2) + scale(fy, y.eps2);
        let cross = scale(fx, x.eps1eps2)
            + scale(fy, y.eps1eps2)
            + scale(fxx, x.eps1 * x.eps2)
            + scale(fxy, x.eps1 * y.eps2 + x.eps2 * y.eps1)
            + scale(fyy, y.eps1 * y.eps2);
        HyperDual::new(f0, eps1, eps2, cross)
    }
}

impl<T: DualNum> sealed::Sealed for HyperDual<T> {
    #[inline]
    fn all_parts_zero(&self) -> bool {
        self.re.all_parts_zero()
            && self.eps1.all_parts_zero()
            && self.eps2.all_parts_zero()
            && self.eps1eps2.all_parts_zero()
    }
}

impl<T: DualNum> DualNum for HyperDual<T> {
    type Float = T::Float;

    const ZERO: Self = HyperDual::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    const ONE: Self = HyperDual::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);

    #[inline]
    fn constant(x: T::Float) -> Self {
        HyperDual::from_value(T::constant(x))
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        HyperDual::from_value(T::from_f64(x))
    }

    #[inline]
    fn re(self) -> T::Float {
        self.re.re()
    }

    #[inline]
    fn recip(self) -> Self {
        let r = self.re.recip();
        let r2 = r * r;
        self.chain(r, -r2, r2 * r * <T::Float as Float>::TWO)
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        let s0 = unsigned_zero(s);
        let f1 = (s0 * <T::Float as Float>::TWO).recip();
        // f″ = −1/(4·s³) = −f′·(f′/s), formed so that it overflows only where it should.
        self.chain(s, f1, -(f1 * (f1 / s0)))
    }

    #[inline]
    fn cbrt(self) -> Self {
        let a = self.re;
        let c = a.cbrt();
        let f1 = (c * c * float::<T::Float>(3.0)).recip();
        // f″ = −2/(9·c⁵) = −(2/3)·f′/a, the factor first, so that f′/a does not overflow
        // where f″ does not (at 5.1e−186).
        self.chain(c, f1, f1 * float::<T::Float>(-2.0 / 3.0) / a)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let a = self.re;
        let m = f64::from(n);
        let v = a.powi(n);
        let (c1, c2) = (float::<T::Float>(m), float::<T::Float>(m * (m - 1.0)));
        let f1 = coefficient(c1, || powi_times(c1, a, n, 1));
        let f2 = coefficient(c2, || powi_times(c2, a, n, 2));
        self.chain(v, f1, f2)
    }

    #[inline]
    fn powf(self, e: T::Float) -> Self {
        let a = self.re;
        let v = a.powf(e);
        let x = exponents(e);
        let f1 = coefficient(e, || {
            if x.huge {
                power_from_value(T::constant(e), v, a, e, 1)
            } else {
                times_power(T::constant(e), power_f(a, x.e1, x.d1), v, a, 1)
            }
        });
        let f2 = coefficient(x.c2, || {
            if x.huge {
                power_from_value(T::constant(x.c2), v, a, e, 2)
            } else {
                times_power(T::constant(x.c2), power_f(a, x.e2, x.d2), v, a, 2)
            }
        });
        self.chain(v, f1, f2)
    }

    #[inline]
    fn pow(self, y: Self) -> Self {
        let (a, b) = (self.re, y.re);
        let v = a.pow(b);
        let ln = a.ln();
        // The exponents y − 1 and y − 2 with the remainders of their rounding (see `powf`).
        // From 2ᵖ − 2 on (see `powf`) the powers are taken from the value, with their parity.
        let b1 = b - T::ONE;
        let big = huge(b.re());
        let zero = <T::Float as Float>::ZERO;
        let d1 = if big { zero } else { less_one(b.re()).1 };
        let r2 = if big { zero } else { less_one(b1.re()).1 };
        let p1 = if big {
            power_from_value(T::ONE, v, a, b.re(), 1)
        } else {
            power(a, b1, d1)
        };
        let b_zero = b.all_parts_zero();
        let fx = if b_zero {
            T::ZERO
        } else {
            times_power(b, p1, v, a, 1)
        };
        let fxx = if b_zero || b1.all_parts_zero() {
            T::ZERO
        } else {
            if big {
                power_from_value(b * b1, v, a, b.re(), 2)
            } else {
                let p2 = power(a, b1 - T::ONE, r2 + d1);
                times_power(b * b1 + b * d1, p2, v, a, 2)
            }
        };
        // At x = 0, where xʸ is 0 (y > 0), the derivatives in y are taken as 0: the limits of
        // xʸ·ln x and xʸ·ln²x, but a choice for the mixed xʸ⁻¹·(1 + y ln x), whose limit is 0 only
        // for y > 1 and −∞ for 0 < y ≤ 1. Where xʸ only underflows they are formed.
        let (fy, fyy, fxy) = if a.re() == zero && v.re() == zero {
            (T::ZERO, T::ZERO, T::ZERO)
        } else {
            let fy = v * ln;
            // ∂²/∂x∂y xʸ = xʸ⁻¹·(1 + y·ln x).
            // xʸ⁻¹ alone may be subnormal where the product is not (at x = 1.6e33, y = −8.35).
            (fy, fy * ln, times_power(b * ln + T::ONE, p1, v, a, 1))
        };
        self.chain2(y, v, fx, fy, fxx, fxy, fyy)
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e, e)
    }

    #[inline]
    fn exp2(self) -> Self {
        let v = self.re.exp2();
        let f1 = v * float::<T::Float>(LN_2);
        self.chain(v, f1, f1 * float::<T::Float>(LN_2))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let e = self.re.exp();
        self.chain(self.re.exp_m1(), e, e)
    }

    #[inline]
    fn ln(self) -> Self {
        let r = unsigned_zero(self.re).recip();
        self.chain(self.re.ln(), r, -(r * r))
    }

    #[inline]
    fn log2(self) -> Self {
        let r = unsigned_zero(self.re).recip();
        let c = float::<T::Float>(LOG2_E);
        self.chain(self.re.log2(), r * c, -(r * c) * r)
    }

    #[inline]
    fn log10(self) -> Self {
        let l = float::<T::Float>(LN_10);
        let f1 = log_slope(self.re, l);
        // f″ = −1/(x²·l) = −f′·(f′·l): the intermediates are below the result, where −r²/l would
        // overflow in r² first (f32 at 4.4e−20).
        self.chain(self.re.log10(), f1, -(f1 * (f1 * l)))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        let r = (self.re + T::ONE).recip();
        self.chain(self.re.ln_1p(), r, -(r * r))
    }

    #[inline]
    fn log(self, base: T::Float) -> Self {
        let l = base.ln();
        let f1 = log_slope(self.re, l);
        self.chain(self.re.log(base), f1, -(f1 * (f1 * l)))
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c, -s)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s, -c)
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.re.tan();
        let f1 = t * t + T::ONE;
        self.chain(t, f1, t * f1 * <T::Float as Float>::TWO)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain(s, c, -s), self.chain(c, -s, -c))
    }

    #[inline]
    fn asin(self) -> Self {
        let a = self.re;
        let g = inverse_sqrt_one_minus_square(a);
        self.chain(a.asin(), g, a * g * g * g)
    }

    #[inline]
    fn acos(self) -> Self {
        let a = self.re;
        let g = inverse_sqrt_one_minus_square(a);
        self.chain(a.acos(), -g, -(a * g * g * g))
    }

    #[inline]
    fn atan(self) -> Self {
        let a = self.re;
        let d = atan_slope(a);
        self.chain(a.atan(), d, -(a * d * d * <T::Float as Float>::TWO))
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        let (y0, x0) = (self.re, x.re);
        let r = y0.hypot(x0);
        let (cy, cx) = (y0 / r, x0 / r);
        let two = <T::Float as Float>::TWO;
        // With c = x/r and s = y/r: ∂/∂y = c/r, ∂/∂x = −s/r, ∂²/∂y² = −2cs/r², ∂²/∂x² = 2cs/r²
        // and ∂²/∂y∂x = (y − x)(y + x)/r⁴. The difference is of y and x themselves, exact near
        // the diagonals, where that of the rounded quotients s − c would cancel (by 41 % at
        // y = 1 + 2⁻⁵², x = 1); each factor is divided by r before it meets the next, so that
        // none underflows ahead of the result.
        let cs = cx * (cy / r) / r * two;
        // Beyond half the largest float y ± x could overflow (to a NaN against a zero factor):
        // there both are halved first, exactly, and the product of the halved factors is taken
        // four times.
        let half = <T::Float as Float>::HALF;
        let large = <T::Float as Float>::MAX * half;
        let mixed = if y0.re().abs() > large || x0.re().abs() > large {
            let (yh, xh) = (y0 * half, x0 * half);
            (yh - xh) / r / r * ((yh + xh) / r) / r * float::<T::Float>(4.0)
        } else {
            (y0 - x0) / r / r * ((y0 + x0) / r) / r
        };
        self.chain2(x, y0.atan2(x0), cx / r, -(cy / r), -cs, mixed, cs)
    }

    #[inline]
    fn sinh(self) -> Self {
        let (s, c) = (self.re.sinh(), self.re.cosh());
        self.chain(s, c, s)
    }

    #[inline]
    fn cosh(self) -> Self {
        let (s, c) = (self.re.sinh(), self.re.cosh());
        self.chain(c, s, c)
    }

    #[inline]
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        let s = sech(self.re);
        let f1 = s * s;
        self.chain(t, f1, -(t * f1 * <T::Float as Float>::TWO))
    }

    #[inline]
    fn asinh(self) -> Self {
        let a = self.re;
        let f1 = T::ONE.hypot(a).recip();
        // f″ = −a/(1 + a²)^(3/2) = −(a·f′)·f′², in an order that does not underflow early.
        self.chain(a.asinh(), f1, -((a * f1) * (f1 * f1)))
    }

    #[inline]
    fn acosh(self) -> Self {
        let a = self.re;
        let f1 = inverse_sqrt_square_minus_one(a);
        self.chain(a.acosh(), f1, -((a * f1) * (f1 * f1)))
    }

    #[inline]
    fn atanh(self) -> Self {
        let a = self.re;
        let f1 = one_minus_square(a).recip();
        self.chain(a.atanh(), f1, a * f1 * f1 * <T::Float as Float>::TWO)
    }

    #[inline]
    fn abs(self) -> Self {
        let v = self.re();
        if v > <T::Float as Float>::ZERO {
            self
        } else if v < <T::Float as Float>::ZERO {
            -self
        } else if v == <T::Float as Float>::ZERO {
            Self::ZERO
        } else {
            HyperDual::new(self.re.abs(), self.eps1, self.eps2, self.eps1eps2)
        }
    }

    #[inline]
    fn hypot(self, y: Self) -> Self {
        let (x0, y0) = (self.re, y.re);
        let h = x0.hypot(y0);
        if h.re() == <T::Float as Float>::ZERO {
            return HyperDual::from_value(h);
        }
        let (cx, cy) = (x0 / h, y0 / h);
        // ∂²/∂x² = y²/h³, ∂²/∂y² = x²/h³, ∂²/∂x∂y = −xy/h³, each as a ratio times a ratio, so
        // that the square of a small ratio does not underflow where the result does not.
        let (fxx, fyy, fxy) = (cy * (cy / h), cx * (cx / h), -(cx * (cy / h)));
        self.chain2(y, h, cx, cy, fxx, fxy, fyy)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let x = self;
        let eps1 = x.re.mul_add(a.eps1, x.eps1.mul_add(a.re, b.eps1));
        let eps2 = x.re.mul_add(a.eps2, x.eps2.mul_add(a.re, b.eps2));
        let cross = x.re.mul_add(
            a.eps1eps2,
            x.eps1eps2.mul_add(
                a.re,
                x.eps1.mul_add(a.eps2, x.eps2.mul_add(a.eps1, b.eps1eps2)),
            ),
        );
        HyperDual::new(x.re.mul_add(a.re, b.re), eps1, eps2, cross)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let (u, v) = (self.re(), other.re());
        if v < u || u.is_nan() { other } else { self }
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let (u, v) = (self.re(), other.re());
        if v > u || u.is_nan() { other } else { self }
    }

    #[inline]
    fn signum(self) -> Self {
        HyperDual::from_value(self.re.signum())
    }
}

impl<T: DualNum> fmt::Display for HyperDual<T> {
    /// Writes `re+eps1ε₁+eps2ε₂+eps1eps2ε₁ε₂`, each part as `Display` writes it, with the
    /// precision given, and a part that is itself a dual number in parentheses: `1-2ε₁+0.5ε₂+3ε₁ε₂`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision();
        super::write_part(f, p, &self.re, "")?;
        super::write_part(f, p, &self.eps1, "ε₁")?;
        super::write_part(f, p, &self.eps2, "ε₂")?;
        super::write_part(f, p, &self.eps1eps2, "ε₁ε₂")
    }
}
