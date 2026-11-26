//! The Riemann and Hurwitz zeta functions, the dilogarithm and the polylogarithm.

mod complex;
mod fd_tables;
mod hurwitz;
mod polylog;
mod real;
pub(super) mod util;

pub(crate) use complex::zeta_c64;
pub(crate) use real::zeta_f64;

use super::{Real, via};

/// The Hurwitz zeta function `ζ(s, a) = Σₙ (n + a)^(−s)`, for `s > 1` and `a > 0` (`+∞` at
/// `s = 1`, NaN outside).
///
/// # Examples
///
/// ```
/// use a::math::special::{hurwitz_zeta, zeta};
///
/// // ζ(s, 1) = ζ(s), ζ(2, 1/2) = 3 ζ(2) = π²/2
/// assert_eq!(hurwitz_zeta(3.0f64, 1.0), zeta(3.0f64));
/// let pi2 = core::f64::consts::PI.powi(2);
/// assert!((hurwitz_zeta(2.0f64, 0.5) - pi2 / 2.0).abs() < 1e-15);
/// // ζ(s, a) ≈ a^(1−s)/(s − 1) for a huge a
/// assert!((hurwitz_zeta(2.0f64, 1e170) / 1e-170 - 1.0).abs() < 1e-15);
/// ```
pub fn hurwitz_zeta<T: Real>(s: T, a: T) -> T {
    T::from_f64(hurwitz::hurwitz_f64(s.to_f64(), a.to_f64()))
}

/// The dilogarithm `Li₂(x) = −∫₀^x ln(1 − t)/t dt`; for `x > 1`, its real part.
///
/// # Examples
///
/// ```
/// use a::math::special::dilog;
///
/// let pi2 = core::f64::consts::PI.powi(2);
/// assert_eq!(dilog(1.0f64), pi2 / 6.0);
/// assert!((dilog(-1.0f64) + pi2 / 12.0).abs() < 1e-15);
/// ```
pub fn dilog<T: Real>(x: T) -> T {
    via(x, polylog::dilog_f64)
}

/// The polylogarithm `Li_s(x) = Σₙ xⁿ / nˢ` of a real order `s`, for `x ≤ 1` (`ζ(s)` at
/// `x = 1`, for `s > 1`); for `x < −1` the real value of the continuation.
///
/// # Examples
///
/// ```
/// use a::math::special::{polylog, zeta};
///
/// assert_eq!(polylog(3.0f64, 1.0), zeta(3.0f64));
/// // Li₋₁(x) = x/(1 − x)², and the zeros Li₋₂ₖ(−1) = 0
/// assert_eq!(polylog(-1.0f64, 0.5), 2.0);
/// assert_eq!(polylog(-2.0f64, -1.0), 0.0);
/// // Li_s(x) → x as s → ∞
/// assert_eq!(polylog(1e300f64, 0.5), 0.5);
/// ```
pub fn polylog<T: Real>(s: T, x: T) -> T {
    T::from_f64(polylog::polylog_real(s.to_f64(), x.to_f64()))
}
