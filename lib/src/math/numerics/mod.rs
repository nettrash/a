//! Numerical methods on `f32` and `f64`: roots, integration, ordinary differential equations,
//! optimisation, interpolation and differentiation.
//!
//! * [`root`]: the zeros of a function of one real variable — [`bisect`](root::bisect),
//!   [`brent`](root::brent), [`secant`](root::secant), [`newton`](root::newton) and
//!   [`halley`](root::halley) — and of a system of equations, by Newton's method with a Jacobian
//!   and by Broyden's method.
//! * [`quad`]: the integral of a function of one real variable, by adaptive Gauss–Kronrod
//!   quadrature (G7–K15 and G10–K21), by Clenshaw–Curtis, and by the tanh-sinh double exponential
//!   method; over infinite intervals too, and in several dimensions by cubature and by Monte
//!   Carlo.
//! * [`gauss`]: the nodes and weights of Gauss–Legendre, Gauss–Hermite, Gauss–Laguerre and
//!   Gauss–Jacobi quadrature, by Golub and Welsch's method on
//!   [`hermitian_eigen`](crate::math::matrix::Matrix::hermitian_eigen) and by Newton's method for
//!   the largest orders.
//! * [`ode`]: the initial value problem `y′ = f(t, y)` — explicit Runge–Kutta methods (the
//!   classical fourth order and Dormand and Prince's 5(4)) and a stiff one (Radau IIA of three
//!   stages, whose corrector is solved on [`lu`](crate::math::matrix::Matrix::lu)) — all of them
//!   with step-size control.
//! * [`opt`]: the minimum of a function — of one variable by the golden section and by Brent's
//!   method; of several by Nelder and Mead's simplex, by the conjugate gradient, and by BFGS; of a
//!   sum of squares by Gauss–Newton and by Levenberg and Marquardt, the latter's damped corrector
//!   and the former's least squares step solved on
//!   [`lu`](crate::math::matrix::Matrix::lu) and
//!   [`qr`](crate::math::matrix::Matrix::qr).
//! * [`interp`]: interpolation and approximation — the cubic splines (natural, clamped and
//!   not-a-knot), Akima's and the shape-preserving PCHIP, Chebyshev approximation on Chebyshev
//!   nodes, and barycentric interpolation with its weights.
//! * [`diff`]: numerical differentiation by Richardson extrapolation.
//!
//! Everything is generic over [`Real`], `f32` and `f64`; an `f32` problem is usually solved
//! through `f64` where that is cheaper than the loss of accuracy. A method that cannot reach its
//! tolerance returns `None` rather than a wrong answer, and a call with inputs of the wrong shape
//! panics.
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
/// of [`Float`] it adds the elementary functions the methods need. The trait is sealed.
pub trait Real: Float + sealed::Real {
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

mod diff;
mod gauss;
mod interp;
mod ode;
mod opt;
mod quad;
mod root;
mod system;

pub use diff::{
    DiffOptions, derivative, derivative_with, second_derivative, second_derivative_with,
};
pub use gauss::{NodesWeights, hermite, jacobi, laguerre, legendre};
pub use interp::{
    Chebyshev, Spline, barycentric_interpolate, barycentric_weights, chebyshev, chebyshev_nodes,
};

pub use ode::{
    DormandPrince5, OdeMethod, OdeResult, RadauIia, Rk4, RkOptions, integrate, integrate_with,
    solve, solve_with,
};
pub use opt::{
    FitResult, MinResult, OneDimResult, OptOptions, bfgs, bfgs_with, conjugate_gradient,
    conjugate_gradient_with, gauss_newton, gauss_newton_with, golden_section, golden_section_with,
    levenberg_marquardt, levenberg_marquardt_with, minimize, minimize_with, nelder_mead,
    nelder_mead_with,
};
pub use quad::{
    QuadOptions, QuadResult, clenshaw_curtis, cubature, monte_carlo, quad, quad_g7k15, quad_g10k21,
    quad_half_infinite, quad_infinite, tanh_sinh,
};
pub use root::{
    RootOptions, bisect, bisect_with, brent, brent_with, halley, halley_with, newton, newton_with,
    secant, secant_with,
};
pub use system::{
    SystemOptions, broyden, broyden_with, newton as newton_system,
    newton_with as newton_system_with,
};

/// The largest number of iterations a method runs before it gives up, when it is not told
/// otherwise.
pub(crate) const DEFAULT_MAX_ITER: usize = 100;

/// Whether `x` is within `tol` of `y`, relatively and absolutely, the test every convergence check
/// here makes: `|x − y| ≤ tol · max(|x|, |y|, 1)`.
#[inline]
pub(crate) fn close<T: Real>(x: T, y: T, tol: T) -> bool {
    let d = (x - y).abs();
    d <= tol * x.abs().max(y.abs().max(T::ONE))
}

/// Whether `a` is truly below `b`, with the answer every method wants when it decides whether it
/// was given something it can work with: a NaN on either side is not below anything, and so is
/// refused rather than believed.
#[inline]
pub(crate) fn ordered<T: PartialOrd>(a: T, b: T) -> bool {
    matches!(a.partial_cmp(&b), Some(core::cmp::Ordering::Less))
}
