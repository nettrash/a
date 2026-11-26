//! Special functions of `f32` and `f64` arguments, and of [`Complex`] ones for the gamma function,
//! its logarithm, the error functions and the Riemann zeta function.
//!
//! * The gamma family: [`gamma`], [`lgamma`] and [`lgamma_sign`], [`digamma`] and [`polygamma`],
//!   [`beta`] and [`lbeta`], the regularised incomplete gamma functions [`gamma_p`] and
//!   [`gamma_q`] and the regularised incomplete beta function [`beta_inc`] with their
//!   complements and inverses, and [`factorial`], [`binomial`] and [`pochhammer`] of floats.
//! * The error functions: [`erf`], [`erfc`], the scaled [`erfcx`], their inverses [`erf_inv`] and
//!   [`erfc_inv`], the Faddeeva function [`faddeeva`], the Fresnel integrals [`fresnel`] and
//!   Dawson's integral [`dawson`].
//! * The Bessel functions [`bessel_j`], [`bessel_y`], [`bessel_i`] and [`bessel_k`] of any real
//!   order, the scaled [`bessel_i_scaled`] and [`bessel_k_scaled`], the spherical
//!   [`sph_bessel_j`], [`sph_bessel_y`], [`sph_bessel_i`] and [`sph_bessel_k`], and the Airy
//!   functions [`airy_ai`] and [`airy_bi`] with their derivatives.
//! * The Riemann zeta function [`zeta`], the Hurwitz zeta function [`hurwitz_zeta`], the
//!   dilogarithm [`dilog`] and the polylogarithm [`polylog`]; both real branches of the Lambert W
//!   function, [`lambert_w0`] and [`lambert_wm1`]; and the exponential integrals [`expint_e1`],
//!   [`expint_ei`] and [`expint_en`].
//! * The complete and incomplete elliptic integrals [`ellip_k`], [`ellip_e`], [`ellip_pi`],
//!   [`ellip_f`], [`ellip_e_inc`] and [`ellip_pi_inc`] of the parameter `m = k²`, Carlson's
//!   symmetric integrals [`carlson_rf`], [`carlson_rd`], [`carlson_rj`] and [`carlson_rc`], and
//!   the Jacobi elliptic functions [`jacobi_elliptic`].
//! * The orthogonal polynomials of Legendre ([`legendre_p`], [`assoc_legendre_p`]), Hermite
//!   ([`hermite_h`], [`hermite_he`]), Laguerre ([`laguerre_l`], [`assoc_laguerre_l`]), Chebyshev
//!   ([`chebyshev_t`], [`chebyshev_u`]), Jacobi ([`jacobi_p`]) and Gegenbauer ([`gegenbauer_c`]).
//!
//! Every function is generic over the type of its arguments: [`Real`] (`f32` and `f64`), or
//! [`Scalar`] (also `Complex<f32>` and `Complex<f64>`) for the functions with complex arguments.
//! An `f32` argument is computed in `f64` and the result rounded once. Outside its domain a
//! function returns NaN, at a pole an infinity, and a NaN argument gives NaN.
//!
//! # Examples
//!
//! ```
//! use a::math::complex::Complex;
//! use a::math::special::{gamma, lgamma};
//!
//! assert!((gamma(5.0f64) - 24.0).abs() < 1e-13);
//! assert!((gamma(0.5f32) - core::f32::consts::PI.sqrt()).abs() < 1e-6);
//! assert!((lgamma(100.0f64) - 359.134_205_369_575_4).abs() < 1e-11);
//! let z = gamma(Complex::new(1.0f64, 1.0));
//! assert!((z.re - 0.498_015_668_118_356).abs() < 1e-14);
//! assert!((z.im + 0.154_949_828_301_810_7).abs() < 1e-14);
//! ```

mod airy;
mod bessel;
mod dd;
mod elliptic;
mod erf;
mod expint;
mod gamma;
mod lambert;
mod orthopoly;
mod zeta;

use crate::math::complex::{Complex, Float};

pub use airy::{airy_ai, airy_ai_prime, airy_bi, airy_bi_prime};
pub use bessel::{
    bessel_i, bessel_i_scaled, bessel_j, bessel_k, bessel_k_scaled, bessel_y, sph_bessel_i,
    sph_bessel_j, sph_bessel_k, sph_bessel_y,
};
pub use elliptic::{
    carlson_rc, carlson_rd, carlson_rf, carlson_rj, ellip_e, ellip_e_inc, ellip_f, ellip_k,
    ellip_pi, ellip_pi_inc, jacobi_elliptic,
};
pub use erf::{dawson, erf_inv, erfc_inv, faddeeva, fresnel};
pub use expint::{expint_e1, expint_ei, expint_en};
pub use gamma::{
    beta, beta_inc, beta_inc_c, beta_inc_c_inv, beta_inc_inv, binomial, digamma, factorial,
    gamma_p, gamma_p_inv, gamma_q, gamma_q_inv, lbeta, lgamma_sign, pochhammer, polygamma,
};
pub use lambert::{lambert_w0, lambert_wm1};
pub use orthopoly::{
    assoc_laguerre_l, assoc_legendre_p, chebyshev_t, chebyshev_u, gegenbauer_c, hermite_h,
    hermite_he, jacobi_p, laguerre_l, legendre_p,
};
pub use zeta::{dilog, hurwitz_zeta, polylog};

pub(crate) mod sealed {
    use crate::math::complex::Complex;

    pub trait Real {
        /// Whether the type is `f32` (whose forms may take kernels of `f32` grade, computed in
        /// `f64` to well past an `f32`'s precision and rounded once).
        const F32: bool;
    }
    impl Real for f32 {
        const F32: bool = true;
    }
    impl Real for f64 {
        const F32: bool = false;
    }

    /// The functions of [`super::Scalar`], each type's own.
    pub trait Scalar: Copy {
        fn gamma(self) -> Self;
        fn lgamma(self) -> Self;
        fn erf(self) -> Self;
        fn erfc(self) -> Self;
        fn erfcx(self) -> Self;
        fn zeta(self) -> Self;
    }

    // `f32` through `f64`, `Complex<f32>` through `Complex<f64>`.
    macro_rules! through {
        ($($f:ident),*) => {
            impl Scalar for f64 {
                $(#[inline] fn $f(self) -> f64 { super::ep::$f::real(self) })*
            }
            impl Scalar for f32 {
                $(#[inline] fn $f(self) -> f32 { super::ep::$f::real32(self as f64) as f32 })*
            }
            impl Scalar for Complex<f64> {
                $(#[inline] fn $f(self) -> Complex<f64> { super::ep::$f::complex(self) })*
            }
            impl Scalar for Complex<f32> {
                $(#[inline] fn $f(self) -> Complex<f32> {
                    let z = super::ep::$f::complex(Complex::new(self.re as f64, self.im as f64));
                    Complex::new(z.re as f32, z.im as f32)
                })*
            }
        };
    }
    through!(gamma, lgamma, erf, erfc, erfcx, zeta);
}

/// `f32` or `f64`: the type of the arguments of every special function.
pub trait Real: Float + sealed::Real {}
impl Real for f32 {}
impl Real for f64 {}

/// `f32`, `f64`, `Complex<f32>` or `Complex<f64>`: the arguments of [`gamma`], [`lgamma`],
/// [`erf`], [`erfc`], [`erfcx`] and [`zeta`].
pub trait Scalar: sealed::Scalar {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex<f32> {}
impl Scalar for Complex<f64> {}

/// A real function of `f64` and its `f32` form through it.
#[inline]
pub(crate) fn via<T: Real>(x: T, f: impl FnOnce(f64) -> f64) -> T {
    T::from_f64(f(x.to_f64()))
}

/// A real function of `f64` and its `f32` form through `g`, a kernel of `f32` grade (in `f64`,
/// rounded once to the `f32`).
#[inline]
pub(crate) fn via_f32<T: Real>(x: T, f: impl FnOnce(f64) -> f64, g: impl FnOnce(f64) -> f64) -> T {
    let x = x.to_f64();
    T::from_f64(if <T as sealed::Real>::F32 { g(x) } else { f(x) })
}

// The internal entry points of the functions of `Scalar`, by area: `ep::<name>::real` of an
// `f64` and `ep::<name>::complex` of a `Complex<f64>`.
mod ep {
    macro_rules! entry {
        ($($name:ident: $area:ident::$real:ident, $real32:ident, $complex:ident;)*) => {$(
            pub(crate) mod $name {
                pub(crate) use crate::math::special::$area::{
                    $complex as complex, $real as real, $real32 as real32,
                };
            }
        )*};
    }
    // `real32` is the kernel of the `f32` form (of `f32` grade, or the `f64` one).
    entry! {
        gamma: gamma::gamma_f64, gamma_f64, gamma_c64;
        lgamma: gamma::lgamma_f64, lgamma_f64, lgamma_c64;
        erf: erf::erf_f64, erf_f32, erf_c64;
        erfc: erf::erfc_f64, erfc_f64, erfc_c64;
        erfcx: erf::erfcx_f64, erfcx_f64, erfcx_c64;
        zeta: zeta::zeta_f64, zeta_f64, zeta_c64;
    }
}

/// The gamma function `Γ(x) = ∫₀^∞ t^(x−1) e^(−t) dt`, continued to the whole plane but the poles
/// `0, −1, −2, …` (±∞ at `±0`, NaN at the negative integers).
#[inline]
pub fn gamma<T: Scalar>(x: T) -> T {
    x.gamma()
}

/// The logarithm of the gamma function: of a real `x`, `ln |Γ(x)|` (its sign is
/// [`lgamma_sign`]'s); of a complex `z`, the principal branch `ln Γ(z)`, analytic in the plane cut
/// along the negative real axis and real on the positive one (SciPy's `loggamma`).
#[inline]
pub fn lgamma<T: Scalar>(x: T) -> T {
    x.lgamma()
}

/// The error function `erf(x) = 2/√π ∫₀^x e^(−t²) dt`.
#[inline]
pub fn erf<T: Scalar>(x: T) -> T {
    x.erf()
}

/// The complementary error function `erfc(x) = 1 − erf(x)`, without the cancellation.
#[inline]
pub fn erfc<T: Scalar>(x: T) -> T {
    x.erfc()
}

/// The scaled complementary error function `erfcx(x) = e^(x²) erfc(x)`.
#[inline]
pub fn erfcx<T: Scalar>(x: T) -> T {
    x.erfcx()
}

/// The Riemann zeta function `ζ(s) = Σ n^(−s)`, continued to the plane but its pole at `s = 1`.
#[inline]
pub fn zeta<T: Scalar>(s: T) -> T {
    s.zeta()
}
