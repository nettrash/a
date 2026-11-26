//! Numerical methods on `f32` and `f64`: roots, integration, ordinary differential equations,
//! optimisation, interpolation and differentiation.
//!
//! * Roots: the zeros of a function of one real variable — [`bisect`], [`brent`], [`toms748`],
//!   [`secant`], [`newton`] and [`halley`] — and of a system of equations, by Newton's method
//!   with a Jacobian ([`newton_system`]) and by Broyden's method ([`broyden`]).
//! * Quadrature: the integral of a function of one real variable, by adaptive Gauss–Kronrod
//!   quadrature as QUADPACK's QAGS and QAGI do it ([`quad`] on G10–K21, [`quad_g10k21`],
//!   [`quad_g7k15`]), by Clenshaw–Curtis ([`clenshaw_curtis`]), and by the tanh-sinh double
//!   exponential method with its exp-sinh and sinh-sinh forms ([`tanh_sinh`]); over infinite
//!   intervals too ([`quad_half_infinite`], [`quad_infinite`], or an infinite limit to any of
//!   them but Clenshaw–Curtis); in several dimensions by Genz and Malik's [`cubature`], by plain
//!   Monte Carlo ([`monte_carlo`]) and by quasi-Monte Carlo on the [`Sobol`] sequence
//!   ([`quasi_monte_carlo`]). A result the rule could not bring within its tolerances comes back
//!   with `converged` false and its error estimate; `None` is for a NaN limit or options that
//!   ask for nothing.
//! * Gaussian rules: the nodes and weights of Gauss–Legendre ([`legendre`]), Gauss–Hermite
//!   ([`hermite`], [`hermite_scaled`]), Gauss–Laguerre ([`laguerre`], [`laguerre_scaled`]) and
//!   Gauss–Jacobi ([`jacobi`]) quadrature, by Golub and Welsch's method up to
//!   [`GOLUB_WELSCH_LARGEST`] points and by Newton's method from asymptotic first guesses above.
//! * Differential equations: the initial value problem `y′ = f(t, y)` by [`solve`],
//!   [`integrate`], [`solve_at`] and [`solve_ivp`], the last with dense output ([`DenseOutput`])
//!   and event location ([`Event`]); by the explicit Runge–Kutta methods [`Rk4`],
//!   [`DormandPrince5`] and [`DormandPrince853`] and the stiff [`RadauIia`] and [`Bdf`], each
//!   controlling its own step size.
//! * Optimisation: the minimum of a function — of one variable by the golden section
//!   ([`golden_section`]) and by Brent's method ([`minimize`]); of several by Nelder and Mead's
//!   simplex ([`nelder_mead`]), by the conjugate gradient ([`conjugate_gradient`]) and by
//!   [`bfgs`]; of several within bounds by L-BFGS-B ([`lbfgsb`]); of a sum of squares by
//!   [`gauss_newton`] and by [`levenberg_marquardt`], both on the QR decomposition.
//! * Linear programming: the optimum of a linear objective under linear inequalities, equations
//!   and bounds, by the simplex method in two phases ([`LinearProgram`]).
//! * Interpolation and approximation: the cubic splines, Akima's and the shape-preserving PCHIP
//!   ([`Spline`]), Chebyshev approximation ([`chebyshev`]), polynomial interpolation in
//!   barycentric form ([`barycentric_interpolate`]), and rational approximation by the AAA
//!   algorithm ([`aaa`], [`aaa_complex`]).
//! * Differentiation: [`derivative`] and [`second_derivative`] by Richardson extrapolation; exact
//!   derivatives, gradients, Jacobians and Hessians by dual numbers are in
//!   [`a::math::dual`](crate::math::dual).
//!
//! Everything is generic over [`Real`], `f32` and `f64`. Where `f32` would lose too much, an
//! `f32` problem is computed in `f64`: the Gaussian rules, the nodes of the quadrature rules and
//! the sums of the Monte Carlo rules, and the poles and zeros of AAA.
//!
//! A method never returns an answer it has not verified. One that cannot reach its tolerance says
//! so: by `None` (the methods for one equation, [`solve`], [`integrate`] and [`solve_at`], the
//! derivatives), by a result with `converged` false and the best point or estimate it reached
//! (the systems, the quadrature rules, the optimisers and the fits), by [`OdeStatus::Failed`]
//! ([`solve_ivp`]), by [`LpStatus::IterationLimit`] or `None` (the simplex method), or by
//! [`Aaa::converged`] false. A call with inputs of the wrong shape panics. A method that takes
//! options comes with the defaults as `name` and with an options struct, built by `new()` and
//! `with_*` calls, as `name_with`; the quadrature rules other than [`quad`], [`monte_carlo`] and
//! [`quasi_monte_carlo`] take their [`QuadOptions`] as an argument.
//!
//! # Examples
//!
//! ```
//! use a::math::numerics::{brent, quad};
//!
//! // √2 as the root of x² − 2, and π as four times the integral of 1/(1 + x²).
//! let r = brent(1.0, 2.0, |x: f64| x * x - 2.0).unwrap();
//! assert!((r - core::f64::consts::SQRT_2).abs() < 1e-15);
//!
//! let i = quad(|x: f64| 4.0 / (1.0 + x * x), 0.0, 1.0).unwrap();
//! assert!((i.value - core::f64::consts::PI).abs() < 1e-15);
//! assert!(i.converged);
//! ```

pub(crate) mod sealed {
    /// The float types of the numerical methods; implemented for `f32` and `f64` only.
    pub trait Real {}
    impl Real for f32 {}
    impl Real for f64 {}
}

use crate::math::complex::Float;

/// `f32` or `f64`: the float type of a numerical method's arguments, unknowns and results. On top
/// of [`Float`] it adds the elementary functions the methods need, and it is a float the
/// transforms of [`a::math::fft`](crate::math::fft) are planned for. The trait is sealed.
pub trait Real: Float + crate::math::fft::FftFloat + sealed::Real {
    /// The sine.
    fn sin(self) -> Self;
    /// The cosine.
    fn cos(self) -> Self;
    /// The tangent.
    fn tan(self) -> Self;
    /// The arcsine, in `[-π/2, π/2]`.
    fn asin(self) -> Self;
    /// The arccosine, in `[0, π]`.
    fn acos(self) -> Self;
    /// The arctangent, in `(-π/2, π/2)`.
    fn atan(self) -> Self;
    /// The hyperbolic sine.
    fn sinh(self) -> Self;
    /// The hyperbolic cosine.
    fn cosh(self) -> Self;
    /// The hyperbolic tangent.
    fn tanh(self) -> Self;
    /// The inverse hyperbolic sine.
    fn asinh(self) -> Self;
    /// The inverse hyperbolic cosine.
    fn acosh(self) -> Self;
    /// The inverse hyperbolic tangent.
    fn atanh(self) -> Self;
    /// The cube root, of the same sign as the argument.
    fn cbrt(self) -> Self;
    /// `self` to the real power `e`.
    fn powf(self, e: Self) -> Self;
    /// The largest integer not greater than it.
    fn floor(self) -> Self;
    /// The smallest integer not less than it.
    fn ceil(self) -> Self;
    /// The nearest integer, halves away from zero.
    fn round(self) -> Self;
    /// The integer part.
    fn trunc(self) -> Self;
    /// The fractional part.
    fn fract(self) -> Self;
    /// `1` or `−1`, or NaN for NaN and `±1` for `±0`.
    fn signum(self) -> Self;
    /// `1/self`.
    fn recip(self) -> Self;
    /// The base-2 logarithm.
    fn log2(self) -> Self;
    /// The base-10 logarithm.
    fn log10(self) -> Self;
    /// `e^self − 1`, accurate near zero.
    fn expm1(self) -> Self;
    /// The nearest integer as a float, halves to even.
    fn round_ties_even(self) -> Self;
}

macro_rules! real {
    ($t:ident) => {
        impl Real for $t {
            #[inline]
            fn sin(self) -> Self {
                $t::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                $t::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                $t::tan(self)
            }
            #[inline]
            fn asin(self) -> Self {
                $t::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                $t::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                $t::atan(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                $t::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                $t::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                $t::tanh(self)
            }
            #[inline]
            fn asinh(self) -> Self {
                $t::asinh(self)
            }
            #[inline]
            fn acosh(self) -> Self {
                $t::acosh(self)
            }
            #[inline]
            fn atanh(self) -> Self {
                $t::atanh(self)
            }
            #[inline]
            fn cbrt(self) -> Self {
                $t::cbrt(self)
            }
            #[inline]
            fn powf(self, e: Self) -> Self {
                $t::powf(self, e)
            }
            #[inline]
            fn floor(self) -> Self {
                $t::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                $t::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                $t::round(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                $t::trunc(self)
            }
            #[inline]
            fn fract(self) -> Self {
                $t::fract(self)
            }
            #[inline]
            fn signum(self) -> Self {
                $t::signum(self)
            }
            #[inline]
            fn recip(self) -> Self {
                1.0 / self
            }
            #[inline]
            fn log2(self) -> Self {
                $t::log2(self)
            }
            #[inline]
            fn log10(self) -> Self {
                $t::log10(self)
            }
            #[inline]
            fn expm1(self) -> Self {
                // 2·e^(x/2)·sinh(x/2), accurate near zero where e^x − 1 is not.
                let h = self * $t::HALF;
                $t::TWO * h.exp() * h.sinh()
            }
            #[inline]
            fn round_ties_even(self) -> Self {
                self.round_ties_even()
            }
        }
    };
}

real!(f32);
real!(f64);

mod aaa;
mod diff;
mod gauss;
mod interp;
mod lbfgsb;
mod lp;
mod ode;
mod opt;
mod quad;
mod root;
mod system;

pub use aaa::{Aaa, AaaOptions, aaa, aaa_complex, aaa_complex_with, aaa_with};
pub use diff::{
    DiffOptions, derivative, derivative_with, second_derivative, second_derivative_with,
};
pub use gauss::{
    GOLUB_WELSCH_LARGEST, NodesWeights, hermite, hermite_scaled, jacobi, laguerre, laguerre_scaled,
    legendre,
};
pub use interp::{
    Chebyshev, Spline, barycentric_interpolate, barycentric_weights, chebyshev, chebyshev_nodes,
};
pub use lbfgsb::{Bound, LbfgsbOptions, LbfgsbResult, LbfgsbStop, lbfgsb, lbfgsb_with};
pub use lp::{LinearProgram, LpOptions, LpResult, LpStatus};

pub use ode::{
    Bdf, ButcherTableau, DenseOutput, DenseStep, DormandPrince5, DormandPrince853, Event,
    EventDirection, EventPoint, NoJacobian, OdeFailure, OdeMethod, OdeOptions, OdeResult,
    OdeSolution, OdeStatus, RadauIia, Rk4, integrate, integrate_with, solve, solve_at,
    solve_at_with, solve_ivp, solve_ivp_with, solve_with,
};
pub use opt::{
    FitResult, MinResult, OneDimResult, OptOptions, bfgs, bfgs_with, conjugate_gradient,
    conjugate_gradient_with, gauss_newton, gauss_newton_with, golden_section, golden_section_with,
    levenberg_marquardt, levenberg_marquardt_with, minimize, minimize_with, nelder_mead,
    nelder_mead_with,
};
pub use quad::{
    MonteCarloOptions, QuadOptions, QuadResult, SOBOL_MAX_DIMENSION, Sobol, clenshaw_curtis,
    cubature, monte_carlo, monte_carlo_with, quad, quad_g7k15, quad_g10k21, quad_half_infinite,
    quad_infinite, quasi_monte_carlo, quasi_monte_carlo_with, tanh_sinh,
};
pub use root::{
    RootOptions, bisect, bisect_with, brent, brent_with, halley, halley_with, newton, newton_with,
    secant, secant_with, toms748, toms748_with,
};
pub use system::{
    SystemOptions, SystemResult, SystemStop, broyden, broyden_with, broyden_with_jacobian,
    newton as newton_system, newton_with as newton_system_with,
};

/// The largest number of iterations a method runs before it gives up, when it is not told
/// otherwise.
pub(crate) const DEFAULT_MAX_ITER: usize = 100;

/// Whether `a` is truly below `b`, with the answer every method wants when it decides whether it
/// was given something it can work with: a NaN on either side is not below anything, and so is
/// refused rather than believed.
#[inline]
pub(crate) fn ordered<T: PartialOrd>(a: T, b: T) -> bool {
    matches!(a.partial_cmp(&b), Some(core::cmp::Ordering::Less))
}
