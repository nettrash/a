//! The problems the ODE tests share, in `f32` and `f64`, and the measures of error they use.

use a::math::matrix::Matrix;
use a::math::numerics::{OdeOptions, Real};

/// The non-stiff problems of the comparisons with SciPy and with Hairer's codes, set up as in
/// `data/gen_reference.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Problem {
    /// `y′ = −y` from 1 on `[0, 10]`.
    Decay,
    /// The harmonic oscillator `y₀′ = y₁`, `y₁′ = −y₀` from `(1, 0)` through ten periods.
    Oscillator,
    /// Kepler's problem from the perihelion of an orbit of eccentricity 0.9, three periods.
    Kepler,
    /// Lotka and Volterra's predators and prey from `(10, 5)` on `[0, 15]`.
    Lotka,
    /// Arenstorf's periodic orbit of the restricted three-body problem, one period.
    Arenstorf,
    /// The Brusselator `y₀′ = 1 + y₀²y₁ − 4y₀`, `y₁′ = 3y₀ − y₀²y₁` from `(1.5, 3)` on `[0, 20]`.
    Brusselator,
}

/// The mass ratio of Arenstorf's problem, and its initial velocity and period.
const MU: f64 = 0.012277471;
#[allow(clippy::excessive_precision)]
const ARENSTORF_V0: f64 = -2.00158510637908252240537862224;
#[allow(clippy::excessive_precision)]
const ARENSTORF_T: f64 = 17.0652165601579625588917206249;

impl Problem {
    /// The initial state.
    pub fn y0(self) -> Vec<f64> {
        match self {
            Problem::Decay => vec![1.0],
            Problem::Oscillator => vec![1.0, 0.0],
            Problem::Kepler => vec![1.0 - 0.9, 0.0, 0.0, ((1.0f64 + 0.9) / (1.0 - 0.9)).sqrt()],
            Problem::Lotka => vec![10.0, 5.0],
            Problem::Arenstorf => vec![0.994, 0.0, 0.0, ARENSTORF_V0],
            Problem::Brusselator => vec![1.5, 3.0],
        }
    }

    /// The end of the interval, which starts at 0.
    pub fn t1(self) -> f64 {
        match self {
            Problem::Decay => 10.0,
            Problem::Oscillator => 20.0 * core::f64::consts::PI,
            Problem::Kepler => 6.0 * core::f64::consts::PI,
            Problem::Lotka => 15.0,
            Problem::Arenstorf => ARENSTORF_T,
            Problem::Brusselator => 20.0,
        }
    }

    /// The right-hand side.
    pub fn rhs<T: Real>(self, _t: T, y: &Matrix<T>, dy: &mut Matrix<T>) {
        let c = T::from_f64;
        match self {
            Problem::Decay => dy[(0, 0)] = -y[(0, 0)],
            Problem::Oscillator => {
                dy[(0, 0)] = y[(1, 0)];
                dy[(1, 0)] = -y[(0, 0)];
            }
            Problem::Kepler => {
                let r2 = y[(0, 0)] * y[(0, 0)] + y[(1, 0)] * y[(1, 0)];
                let r3 = r2 * r2.sqrt();
                dy[(0, 0)] = y[(2, 0)];
                dy[(1, 0)] = y[(3, 0)];
                dy[(2, 0)] = -y[(0, 0)] / r3;
                dy[(3, 0)] = -y[(1, 0)] / r3;
            }
            Problem::Lotka => {
                dy[(0, 0)] = c(1.5) * y[(0, 0)] - y[(0, 0)] * y[(1, 0)];
                dy[(1, 0)] = c(-3.0) * y[(1, 0)] + y[(0, 0)] * y[(1, 0)];
            }
            Problem::Arenstorf => {
                let (mu, mup) = (c(MU), T::ONE - c(MU));
                let (x, z) = (y[(0, 0)], y[(1, 0)]);
                let r1 = (x + mu) * (x + mu) + z * z;
                let r1 = r1 * r1.sqrt();
                let r2 = (x - mup) * (x - mup) + z * z;
                let r2 = r2 * r2.sqrt();
                dy[(0, 0)] = y[(2, 0)];
                dy[(1, 0)] = y[(3, 0)];
                dy[(2, 0)] = x + c(2.0) * y[(3, 0)] - mup * (x + mu) / r1 - mu * (x - mup) / r2;
                dy[(3, 0)] = z - c(2.0) * y[(2, 0)] - mup * z / r1 - mu * z / r2;
            }
            Problem::Brusselator => {
                let (u, v) = (y[(0, 0)], y[(1, 0)]);
                dy[(0, 0)] = T::ONE + u * u * v - c(4.0) * u;
                dy[(1, 0)] = c(3.0) * u - u * u * v;
            }
        }
    }

    /// The right-hand side as a closure, for the entry points.
    pub fn f<T: Real>(self) -> impl FnMut(T, &Matrix<T>, &mut Matrix<T>) + Copy {
        move |t, y, dy| self.rhs(t, y, dy)
    }
}

/// The column of `v`, converted to `T`.
pub fn col<T: Real>(v: &[f64]) -> Matrix<T> {
    Matrix::from_column_major(v.len(), 1, v.iter().map(|&x| T::from_f64(x)).collect())
}

/// The error of `y` against `exact`, componentwise in the mixed absolute and relative scale
/// `1 + |exact|` that `rtol = atol` stands for, and the largest of them.
pub fn error<T: Real>(y: &[T], exact: &[f64]) -> f64 {
    assert_eq!(y.len(), exact.len());
    y.iter()
        .zip(exact)
        .map(|(&v, &e)| (v.to_f64() - e).abs() / (1.0 + e.abs()))
        .fold(0.0, f64::max)
}

/// `miri` under Miri, which interprets every floating-point operation and so affords tens of
/// steps where a test natively takes thousands, and `native` otherwise.
pub fn miri_or<T>(miri: T, native: T) -> T {
    if cfg!(miri) { miri } else { native }
}

/// Whether two integrations that take the same steps reached the same state: bit for bit — but
/// under Miri, which perturbs the last bits of `powf` and the other functions IEEE 754 leaves
/// open, and with them the step sizes, from one run to the next, only to a hair.
pub fn same_state<T: Real>(a: &[T], b: &[T]) -> bool {
    if cfg!(miri) {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| (x.to_f64() - y.to_f64()).abs() <= 1e-9 * (1.0 + x.to_f64().abs()))
    } else {
        a == b
    }
}

/// The options with `rtol = atol = tol`.
pub fn tol<T: Real>(tol: f64) -> OdeOptions<T> {
    OdeOptions::new().with_tolerances(T::from_f64(tol), T::from_f64(tol))
}

/// The options of a method held to the fixed step `h`: tolerances no step can fail, the first
/// step `h` and no step longer than `h` (and a hair over it, so that the last step of an interval
/// of `n·h` lands on its end rather than being split for the rounding of the time).
pub fn fixed<T: Real>(h: f64) -> OdeOptions<T> {
    OdeOptions::new()
        .with_tolerances(T::from_f64(1e10), T::from_f64(1e10))
        .with_h0(T::from_f64(h))
        .with_hmax(T::from_f64(h * (1.0 + 1e-9)))
}

/// The slope of the least-squares line through `(ln x, ln y)`.
pub fn log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x.ln(), b + y.ln()));
    let (mx, my) = (sx / n, sy / n);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &(x, y) in points {
        sxy += (x.ln() - mx) * (y.ln() - my);
        sxx += (x.ln() - mx) * (x.ln() - mx);
    }
    sxy / sxx
}
