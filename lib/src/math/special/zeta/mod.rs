//! The Riemann and Hurwitz zeta functions, the dilogarithm and the polylogarithm.

mod complex;
mod hurwitz;
mod polylog;
mod real;
pub(super) mod util;

pub(crate) use complex::zeta_c64;
pub(crate) use real::zeta_f64;

use super::{Real, via};

/// The Hurwitz zeta function `ζ(s, a) = Σₙ (n + a)^(−s)`, for `s > 1` and `a > 0`.
pub fn hurwitz_zeta<T: Real>(s: T, a: T) -> T {
    T::from_f64(hurwitz::hurwitz_f64(s.to_f64(), a.to_f64()))
}

/// The dilogarithm `Li₂(x) = −∫₀^x ln(1 − t)/t dt`; for `x > 1`, its real part.
pub fn dilog<T: Real>(x: T) -> T {
    via(x, polylog::dilog_f64)
}

/// The polylogarithm `Li_s(x) = Σₙ xⁿ / nˢ` of a real order `s`, for `x ≤ 1` (`ζ(s)` at
/// `x = 1`, for `s > 1`).
pub fn polylog<T: Real>(s: T, x: T) -> T {
    T::from_f64(polylog::polylog_real(s.to_f64(), x.to_f64()))
}
