//! The elementary functions of dual numbers: `f(a + bε) = f(a) + f′(a)·b·ε`, with `f′` formed
//! so that it keeps its digits where the obvious formula would cancel or overflow.

use super::Dual;
use super::num::{DualNum, float, scale, sealed};
use crate::math::complex::Float;
use core::f64::consts::{LN_2, LN_10, LOG2_E};

impl<T: DualNum> Dual<T> {
    /// `f(self)` from the value `f0` and the derivative `f1` of `f` at `self.re`.
    #[inline]
    fn chain(self, f0: T, f1: T) -> Self {
        Dual::new(f0, scale(f1, self.eps))
    }

    /// `f(self, y)` from the value `f0` and the partial derivatives `fx` and `fy` of `f` at
    /// `(self.re, y.re)`.
    #[inline]
    fn chain2(self, y: Self, f0: T, fx: T, fy: T) -> Self {
        Dual::new(f0, scale(fx, self.eps) + scale(fy, y.eps))
    }
}

/// `x()`, or zero where the coefficient `c` that `x()` multiplies is zero, so that `0·xⁿ` is 0
/// where `xⁿ` is infinite: the derivatives of `x⁰` and `x¹` are 0 at 0.
#[inline]
pub(super) fn coefficient<T: DualNum>(c: T::Float, x: impl FnOnce() -> T) -> T {
    if c == <T::Float as Float>::ZERO {
        T::ZERO
    } else {
        x()
    }
}

/// `a + 0`, which is `a` but for `−0`, which it makes `+0`: the derivatives of the square root
/// and the logarithms at 0 are one-sided, from the right, and `+∞` for either zero, where
/// `1/(2·√−0)` and `1/−0` would make them `−∞`.
#[inline]
pub(super) fn unsigned_zero<T: DualNum>(a: T) -> T {
    a + <T::Float as Float>::ZERO
}

/// `a − 1` rounded and the exact remainder `(a − 1) − fl(a − 1)`, by Knuth's two-sum.
#[inline]
pub(super) fn less_one<F: Float>(a: F) -> (F, F) {
    let b = -F::ONE;
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Whether `|e| ≥ 2ᵖ − 2`, `p` the precision of the float, or `e` is infinite: every such float is
/// an integer, and `e − 1` or `e − 2` need not be a float (`−(2⁵³ − 1) − 2` is not), so that the
/// powers of the derivatives of `xᵉ` are taken from the value instead, as `xᵉ/x` and `xᵉ/x²`
/// (see [`power_from_value`]), which have the parity of `e − 1` and `e − 2`.
#[inline]
pub(super) fn huge<F: Float>(e: F) -> bool {
    let limit = F::from_f64(2f64.powi(F::MANTISSA_DIGITS as i32) - 2.0);
    e.abs() >= limit
}

/// `c·xᵉ⁻ᵏ` for `k` = 1 or 2 and an exponent [`huge`] beyond the floats' integers, the power from
/// the value `v = xᵉ`, as `v/xᵏ`: it has the parity of `e − k` that `x^fl(e − k)` would not (the
/// derivative of `x^(2⁵³ + 2)` at −1 is −(2⁵³ + 2)), and the sign of an overflow. At `x = ±0` and
/// `±∞`, where `v/xᵏ` would be `0/0` or `∞/∞`, the power is `|x|^fl(e − k)` (0 or `∞`), negative
/// for a negative `x` where `e − k` is odd (an infinite `e` counts as even). A power of 0 gives 0,
/// also where the coefficient `e·(e − 1)` has overflowed.
#[inline]
pub(super) fn power_from_value<T: DualNum>(c: T, v: T, x: T, e: T::Float, k: u32) -> T {
    let r = x.re();
    let p = if r != <T::Float as Float>::ZERO && r.is_finite() {
        if k == 1 { v / x } else { v / x / x }
    } else {
        let two = <T::Float as Float>::TWO;
        let e_odd = e.is_finite() && e % two != <T::Float as Float>::ZERO;
        let odd = e_odd != (k == 1);
        let q = x.abs().powf(e - float::<T::Float>(f64::from(k)));
        if odd && r.is_sign_negative() { -q } else { q }
    };
    if p.all_parts_zero() { T::ZERO } else { p * c }
}

/// The exponents of the first and second derivatives of `xᵉ`, `e − 1` and `e − 2`, each rounded
/// and with the remainder its rounding left out, and the coefficient `e·(e − 1)` of the second
/// (exactly 0 for `e` = 0 and 1). For a [`huge`] exponent the remainders are 0 and not used.
pub(super) struct Exponents<F> {
    pub(super) e1: F,
    pub(super) d1: F,
    pub(super) e2: F,
    pub(super) d2: F,
    pub(super) c2: F,
    pub(super) huge: bool,
}

#[inline]
pub(super) fn exponents<F: Float>(e: F) -> Exponents<F> {
    if huge(e) {
        // e·(e − 1) as e·fl(e − 1), within 2⁻ᵖ of it; +∞ for either infinity.
        let e1 = e - F::ONE;
        return Exponents {
            e1,
            d1: F::ZERO,
            e2: e1 - F::ONE,
            d2: F::ZERO,
            c2: e * e1,
            huge: true,
        };
    }
    let (e1, d1) = less_one(e);
    let (e2, r2) = less_one(e1);
    let c2 = if e == F::ZERO || e == F::ONE {
        F::ZERO
    } else {
        e * e1 + e * d1
    };
    Exponents {
        e1,
        d1,
        e2,
        d2: r2 + d1,
        c2,
        huge: false,
    }
}

/// `x^(e + d)` for an exponent `e` and the small remainder `d` that its rounding left out, as
/// `xᵉ·(1 + d·ln x)`. `xᵉ` alone would carry the rounding `d` multiplied by `ln x` into the
/// result — 145 ulps of `0.1·x^(−0.9)` at `x = 10³⁰⁰`, some 50 at ordinary `x` for an exponent of
/// full precision — where `d·ln x`, below `10⁻¹³` wherever `xᵉ` is a finite nonzero number, is
/// all the correction there is to make. The exponent is a dual number for `pow`. Nothing is
/// corrected where `d` is 0 (an exact `e − 1`); for `x ≤ 0`, where only an integer exponent has a
/// real power, and an integer below `2ᵖ` has an exact `e − 1` (beyond, the exponent is [`huge`]
/// and the power is taken by [`power_from_value`]); nor where `xᵉ` is 0 or infinite, where the
/// correction would not be small and could turn the sign of an overflow.
#[inline]
pub(super) fn power<T: DualNum>(x: T, e: T, d: T::Float) -> T {
    corrected(x.pow(e), x, d)
}

/// [`power`] for an exponent that is a float, by `powf`.
#[inline]
pub(super) fn power_f<T: DualNum>(x: T, e: T::Float, d: T::Float) -> T {
    corrected(x.powf(e), x, d)
}

/// `p·(1 + d·ln x)` where [`power`] says the correction is made, and `p` elsewhere.
#[inline]
fn corrected<T: DualNum>(p: T, x: T, d: T::Float) -> T {
    let zero = <T::Float as Float>::ZERO;
    let (r, pr) = (x.re(), p.re());
    if d == zero || r <= zero || r.is_nan() || pr == zero || !pr.is_finite() {
        p
    } else {
        p * (x.ln() * d + T::ONE)
    }
}

/// `1/(x·l)`, the derivative of a logarithm to a base of natural logarithm `l`, at `x + 0` (see
/// [`unsigned_zero`]): as `1/(x·l)` for `|l| ≥ 1` and as `(1/x)/l` below, so that the derivative of
/// the slope itself, which the nested types take, overflows only where the second derivative
/// `−1/(x²·l)` does (`(1/x)′ = −1/x²` alone overflows first for `|l| > 1`: at `4.4e−155` to the
/// base 0.027, and in `f32` for `log10` at `4.4e−20`).
#[inline]
pub(super) fn log_slope<T: DualNum>(x: T, l: T::Float) -> T {
    let x = unsigned_zero(x);
    if l.abs() >= <T::Float as Float>::ONE {
        (x * l).recip()
    } else {
        x.recip() / l
    }
}

/// `x` to the power `n − k`, which for `n` near `i32::MIN` is `xⁿ / xᵏ`, as `n − k` does not fit.
#[inline]
pub(super) fn powi_less<T: DualNum>(x: T, n: i32, k: i32) -> T {
    match n.checked_sub(k) {
        Some(m) => x.powi(m),
        None => x.powi(n) / x.powi(k),
    }
}

/// `c·xⁿ⁻ᵏ`, a derivative of `xⁿ`. Where `xⁿ⁻ᵏ` alone falls below the normal numbers at `x ≠ 0`
/// — subnormal, or 0 where the standard library's `powi` forms a negative power as `1/x^|m|` and
/// `x^|m|` overflows — though `c·xⁿ⁻ᵏ` need not, it is taken in two halves of the power with the
/// coefficient between them, `(x^(m − ⌊m/2⌋)·c)·x^⌊m/2⌋`, `m = n − k`, so that no factor leaves
/// the range before the product does: `−1000·x⁻¹⁰⁰¹` at 2.03, `−12·x⁻¹³` at 978.7 in `f32`.
/// (`|c| ≥ 1` for an integer power, so an overflow of the power is one of the product too.)
#[inline]
pub(super) fn powi_times<T: DualNum>(c: T::Float, x: T, n: i32, k: i32) -> T {
    let p = powi_less(x, n, k);
    let (pr, xr) = (p.re(), x.re());
    let min = <T::Float as Float>::MIN_POSITIVE;
    if xr == <T::Float as Float>::ZERO || !xr.is_finite() || pr.abs() >= min || pr.is_nan() {
        return p * c;
    }
    let m = i64::from(n) - i64::from(k);
    let h = m / 2;
    (x.powi((m - h) as i32) * c) * x.powi(h as i32)
}

/// `c·p` for the power `p = x^(e−k)`, `k` = 1 or 2, of a derivative of `xᵉ`; but where `p`
/// alone has left the range — infinite, or below the normal numbers, for `x ≠ 0` — while the value
/// `v = xᵉ` has not, `(c·v)/xᵏ`, the coefficient first. In those places only the coefficient can
/// bring the product back: `|c| < 1` where `p` overflows (`−0.6·x^(−1.6)` at `8.5e−194`), `|c| > 1`
/// where it underflows. The test reads the values alone: in the nested types a derivative part of
/// `p` can overflow where its value does not, and is not brought back (a limit the page states).
#[inline]
pub(super) fn times_power<T: DualNum>(c: T, p: T, v: T, x: T, k: u32) -> T {
    let (pr, vr, xr) = (p.re(), v.re(), x.re());
    let min = <T::Float as Float>::MIN_POSITIVE;
    let lost = xr != <T::Float as Float>::ZERO
        && xr.is_finite()
        && vr.is_finite()
        && vr.abs() >= min
        && !(pr.is_finite() && pr.abs() >= min);
    if !lost {
        return p * c;
    }
    let t = v * c;
    if k == 1 { t / x } else { t / x / x }
}

impl<T: DualNum> sealed::Sealed for Dual<T> {
    #[inline]
    fn all_parts_zero(&self) -> bool {
        self.re.all_parts_zero() && self.eps.all_parts_zero()
    }
}

impl<T: DualNum> DualNum for Dual<T> {
    type Float = T::Float;

    const ZERO: Self = Dual::new(T::ZERO, T::ZERO);
    const ONE: Self = Dual::new(T::ONE, T::ZERO);

    #[inline]
    fn constant(x: T::Float) -> Self {
        Dual::new(T::constant(x), T::ZERO)
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        Dual::new(T::from_f64(x), T::ZERO)
    }

    #[inline]
    fn re(self) -> T::Float {
        self.re.re()
    }

    #[inline]
    fn recip(self) -> Self {
        let r = self.re.recip();
        self.chain(r, -(r * r))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        self.chain(s, (unsigned_zero(s) * <T::Float as Float>::TWO).recip())
    }

    #[inline]
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        self.chain(c, (c * c * float::<T::Float>(3.0)).recip())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let a = self.re;
        let v = a.powi(n);
        let c = float::<T::Float>(f64::from(n));
        let f1 = coefficient(c, || powi_times(c, a, n, 1));
        self.chain(v, f1)
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
        self.chain(v, f1)
    }

    #[inline]
    fn pow(self, y: Self) -> Self {
        let (a, b) = (self.re, y.re);
        let v = a.pow(b);
        let fx = if b.all_parts_zero() {
            T::ZERO
        } else if huge(b.re()) {
            power_from_value(b, v, a, b.re(), 1)
        } else {
            times_power(b, power(a, b - T::ONE, less_one(b.re()).1), v, a, 1)
        };
        // xʸ·ln x is 0 in the limit at x = 0 for y > 0, where xʸ is 0; where xʸ only underflows
        // the product is formed.
        let zero = <T::Float as Float>::ZERO;
        let fy = if a.re() == zero && v.re() == zero {
            T::ZERO
        } else {
            v * a.ln()
        };
        self.chain2(y, v, fx, fy)
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    #[inline]
    fn exp2(self) -> Self {
        let v = self.re.exp2();
        self.chain(v, v * float::<T::Float>(LN_2))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }

    #[inline]
    fn ln(self) -> Self {
        self.chain(self.re.ln(), unsigned_zero(self.re).recip())
    }

    #[inline]
    fn log2(self) -> Self {
        let f1 = unsigned_zero(self.re).recip() * float::<T::Float>(LOG2_E);
        self.chain(self.re.log2(), f1)
    }

    #[inline]
    fn log10(self) -> Self {
        let f1 = log_slope(self.re, float::<T::Float>(LN_10));
        self.chain(self.re.log10(), f1)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), (self.re + T::ONE).recip())
    }

    #[inline]
    fn log(self, base: T::Float) -> Self {
        let f1 = log_slope(self.re, base.ln());
        self.chain(self.re.log(base), f1)
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s)
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.re.tan();
        self.chain(t, t * t + T::ONE)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }

    #[inline]
    fn asin(self) -> Self {
        self.chain(self.re.asin(), inverse_sqrt_one_minus_square(self.re))
    }

    #[inline]
    fn acos(self) -> Self {
        self.chain(self.re.acos(), -inverse_sqrt_one_minus_square(self.re))
    }

    #[inline]
    fn atan(self) -> Self {
        let a = self.re;
        self.chain(a.atan(), atan_slope(a))
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        let (y0, x0) = (self.re, x.re);
        let r = y0.hypot(x0);
        let (cy, cx) = (y0 / r, x0 / r);
        // ∂/∂y = x/r², ∂/∂x = −y/r², with r = hypot(x, y): no square to overflow.
        self.chain2(x, y0.atan2(x0), cx / r, -(cy / r))
    }

    #[inline]
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }

    #[inline]
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }

    #[inline]
    fn tanh(self) -> Self {
        let s = sech(self.re);
        self.chain(self.re.tanh(), s * s)
    }

    #[inline]
    fn asinh(self) -> Self {
        let a = self.re;
        self.chain(a.asinh(), T::ONE.hypot(a).recip())
    }

    #[inline]
    fn acosh(self) -> Self {
        let a = self.re;
        self.chain(a.acosh(), inverse_sqrt_square_minus_one(a))
    }

    #[inline]
    fn atanh(self) -> Self {
        let a = self.re;
        self.chain(a.atanh(), one_minus_square(a).recip())
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
            Dual::new(self.re.abs(), self.eps)
        }
    }

    #[inline]
    fn hypot(self, y: Self) -> Self {
        let (x0, y0) = (self.re, y.re);
        let h = x0.hypot(y0);
        if h.re() == <T::Float as Float>::ZERO {
            return Dual::new(h, T::ZERO);
        }
        self.chain2(y, h, x0 / h, y0 / h)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let eps = self.re.mul_add(a.eps, self.eps.mul_add(a.re, b.eps));
        Dual::new(self.re.mul_add(a.re, b.re), eps)
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
        Dual::new(self.re.signum(), T::ZERO)
    }
}

/// `1 − a²`: as `(1 − a)(1 + a)` for `|a| ≥ 1/2`, where `1 − a` is exact and `1 − a²` would lose
/// the last digits of `a²` near ±1; as `1 − a²` below, where the derivative of the product,
/// `(1 − a) − (1 + a)`, would cancel (which the nested types, `Dual<Dual<_>>`, would see).
#[inline]
pub(super) fn one_minus_square<T: DualNum>(a: T) -> T {
    if a.re().abs() >= <T::Float as Float>::HALF {
        (T::ONE - a) * (T::ONE + a)
    } else {
        T::ONE - a * a
    }
}

/// `1/√(1 − a²)`, the derivative of the arcsine, from [`one_minus_square`].
#[inline]
pub(super) fn inverse_sqrt_one_minus_square<T: DualNum>(a: T) -> T {
    one_minus_square(a).sqrt().recip()
}

/// `1/(1 + a²)`, the derivative of the arctangent: as `r²/(1 + r²)` with `r = 1/a` for
/// `|a| > 1`, so that no square overflows, nor its derivatives in the nested types.
#[inline]
pub(super) fn atan_slope<T: DualNum>(a: T) -> T {
    if a.re().abs() > <T::Float as Float>::ONE {
        let r = a.recip();
        let r2 = r * r;
        r2 / (r2 + T::ONE)
    } else {
        (a * a + T::ONE).recip()
    }
}

/// `1/cosh a`: as `2e^(−|a|)` for `|a| > 20`, where `e^(−2|a|)` is below the rounding of 1 in
/// either type and `cosh a` would overflow (at 89 in `f32`) and take its derivatives with it.
#[inline]
pub(super) fn sech<T: DualNum>(a: T) -> T {
    if a.re().abs() > float::<T::Float>(20.0) {
        (-a.abs()).exp() * <T::Float as Float>::TWO
    } else {
        a.cosh().recip()
    }
}

/// `1/(√(a − 1)·√(a + 1))`, the derivative of the inverse hyperbolic cosine: `a − 1` is exact
/// near 1, where `a² − 1` would lose the digits of `a²`, and the product of the roots does not
/// overflow where `a²` would.
#[inline]
pub(super) fn inverse_sqrt_square_minus_one<T: DualNum>(a: T) -> T {
    ((a - T::ONE).sqrt() * (a + T::ONE).sqrt()).recip()
}
