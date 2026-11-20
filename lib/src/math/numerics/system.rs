//! The zeros of a system of equations, by Newton's method with a Jacobian and by Broyden's
//! method.
//!
//! A system of `n` equations in `n` unknowns is a function `f` from `Rⁿ` to `Rⁿ`, which is held
//! as a [`Matrix`] with one column. Newton's method needs the Jacobian `J` of `f` too, and solves
//! the correction equation `J·δ = −f` on [`lu`]; Broyden's method needs only `f` and keeps a
//! finite-difference approximation of `J` that a rank-one update improves at every step.
//!
//! Both return `None` when they cannot reach the tolerance.

use super::{DEFAULT_MAX_ITER, Real};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// The options of [`newton`] and [`broyden`]: the tolerances, how many steps are run, and how the
/// finite differences of [`broyden`] are formed.
#[derive(Debug, Clone, Copy)]
pub struct SystemOptions<T> {
    /// How close two successive iterates must be for the answer to be accepted.
    pub xtol: T,
    /// How small the infinity norm of the residuals must be for the answer to be accepted.
    pub ftol: T,
    /// The largest number of steps.
    pub max_iter: usize,
    /// The relative step of the finite differences, `√ε` by default.
    pub h: T,
}

impl<T: Real> Default for SystemOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> SystemOptions<T> {
    /// The default options: `xtol` 4·`ε`, `ftol` `ε`, `max_iter` [`DEFAULT_MAX_ITER`] and the
    /// finite-difference step `√ε`, all in `T`.
    pub fn new() -> Self {
        let eps = T::EPSILON;
        Self {
            xtol: T::from_f64(4.0) * eps,
            ftol: eps,
            max_iter: DEFAULT_MAX_ITER,
            h: eps.sqrt(),
        }
    }

    /// Sets [`xtol`](Self::xtol).
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets [`ftol`](Self::ftol).
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets [`max_iter`](Self::max_iter).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets [`h`](Self::h).
    pub fn with_finite_difference_step(mut self, h: T) -> Self {
        self.h = h;
        self
    }
}

/// The infinity norm of a one-column matrix.
fn norm_inf<T: Real + Scalar>(a: &Matrix<T>) -> T {
    let mut m = T::ZERO;
    for x in a.as_slice() {
        m = m.max(Float::abs(*x));
    }
    m
}

/// The solution of `f(x) = 0` from the starting point `x0`, by Newton's method: at every step the
/// Jacobian `j(x)` is formed and the correction `δ` solving `j(x)·δ = −f(x)` is subtracted from
/// the iterate. `f` and `j` take the iterate as a one-column matrix and return the values and the
/// Jacobian as one- and `n × n`-column matrices of the same number of rows.
///
/// Returns `None` when the Jacobian is singular or the iteration does not converge.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::newton_system;
///
/// // The intersection of the circle of radius 2 centred on the origin with the line y = x − 1
/// // that lies nearest the starting point, at ((1 + √7)/2, (√7 − 1)/2).
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0, x[(0, 0)] - x[(1, 0)] - 1.0]);
/// let j = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], [1.0, -1.0]]);
/// let x = newton_system(&Matrix::from_column_major(2, 1, vec![1.5, 0.5]), f, j).unwrap();
/// assert!((x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-12);
/// assert!((x[(1, 0)] - 0.822_875_655_532_295_2).abs() < 1e-12);
/// ```
pub fn newton<T>(
    x0: &Matrix<T>,
    f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    j: impl FnMut(&Matrix<T>) -> Matrix<T>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
{
    newton_with(x0, f, j, &SystemOptions::new())
}

/// [`newton`] with the options `opts`.
pub fn newton_with<T>(
    x0: &Matrix<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    mut j: impl FnMut(&Matrix<T>) -> Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
{
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 {
        return None;
    }
    let mut x = x0.clone();
    let mut fx = f(&x);
    if fx.rows() != n || fx.cols() != 1 {
        return None;
    }
    for _ in 0..opts.max_iter {
        if norm_inf(&fx) <= opts.ftol {
            return Some(x);
        }
        let jx = j(&x);
        if jx.shape() != (n, n) {
            return None;
        }
        let delta = jx.lu().solve(&-&fx)?;
        for i in 0..n {
            x[(i, 0)] += delta[(i, 0)];
        }
        let step = norm_inf(&delta);
        fx = f(&x);
        if !fx
            .as_slice()
            .iter()
            .all(|v| Float::is_finite(Float::abs(*v)))
        {
            return None;
        }
        if step <= opts.xtol * norm_inf(&x).max(T::ONE) {
            return Some(x);
        }
    }
    if norm_inf(&fx) <= opts.ftol.max(opts.xtol) {
        Some(x)
    } else {
        None
    }
}

/// The solution of `f(x) = 0` from the starting point `x0`, by Broyden's method: the Jacobian is
/// approximated by finite differences once and then improved by a rank-one update at every step,
/// so that `f` alone is needed.
///
/// Returns `None` when the approximation becomes singular or the iteration does not converge.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::broyden;
///
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0, x[(0, 0)] - x[(1, 0)] - 1.0]);
/// let x = broyden(&Matrix::from_column_major(2, 1, vec![1.5, 0.5]), f).unwrap();
/// assert!((x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-12);
/// ```
pub fn broyden<T>(x0: &Matrix<T>, f: impl FnMut(&Matrix<T>) -> Matrix<T>) -> Option<Matrix<T>>
where
    T: Real + Scalar,
{
    broyden_with(x0, f, &SystemOptions::new())
}

/// [`broyden`] with the options `opts`.
pub fn broyden_with<T>(
    x0: &Matrix<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
{
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 {
        return None;
    }
    let mut x = x0.clone();
    let mut fx = f(&x);
    if fx.rows() != n || fx.cols() != 1 {
        return None;
    }
    // The starting Jacobian, by central differences of the columns.
    let mut g = Matrix::<T>::zeros(n, n);
    for k in 0..n {
        let xk = x[(k, 0)];
        let h = opts.h * T::ONE.max(Float::abs(xk));
        x[(k, 0)] = xk + h;
        let up = f(&x);
        x[(k, 0)] = xk - h;
        let down = f(&x);
        x[(k, 0)] = xk;
        for i in 0..n {
            g[(i, k)] = (up[(i, 0)] - down[(i, 0)]) / (T::TWO * h);
        }
    }
    let mut ginv = g.lu().inverse()?;

    for _ in 0..opts.max_iter {
        if norm_inf(&fx) <= opts.ftol {
            return Some(x);
        }
        let delta = &ginv * T::from_f64(-1.0) * &fx;
        for i in 0..n {
            x[(i, 0)] += delta[(i, 0)];
        }
        let step = norm_inf(&delta);
        let fx_old = fx.clone();
        fx = f(&x);
        if !fx
            .as_slice()
            .iter()
            .all(|v| Float::is_finite(Float::abs(*v)))
        {
            return None;
        }
        // The rank-one (Broyden's second) update of the inverse Jacobian: `G ← G + ((δ − G·Δf)·δ†G)
        // / (δ†G·Δf)`, where δ is the step taken and Δf the change in the values.
        let mut df = Matrix::<T>::zeros(n, 1);
        for i in 0..n {
            df[(i, 0)] = fx[(i, 0)] - fx_old[(i, 0)];
        }
        let gd = &ginv * &df;
        let mut y = Matrix::<T>::zeros(n, 1);
        let mut denom = T::ZERO;
        for i in 0..n {
            let mut s = T::ZERO;
            for k in 0..n {
                s += delta[(k, 0)] * ginv[(k, i)];
            }
            y[(i, 0)] = delta[(i, 0)] - gd[(i, 0)];
            denom += s * df[(i, 0)];
        }
        if Float::abs(denom) > T::EPSILON * norm_inf(&delta).max(T::ONE) {
            let mut w = Matrix::<T>::zeros(1, n);
            for j in 0..n {
                let mut s = T::ZERO;
                for k in 0..n {
                    s += delta[(k, 0)] * ginv[(k, j)];
                }
                w[(0, j)] = s;
            }
            let c = &y * &w * (T::from_f64(1.0) / denom);
            for i in 0..n {
                for j in 0..n {
                    ginv[(i, j)] += c[(i, j)];
                }
            }
        }
        if step <= opts.xtol * norm_inf(&x).max(T::ONE) && norm_inf(&fx) <= opts.ftol.max(opts.xtol)
        {
            return Some(x);
        }
    }
    if norm_inf(&fx) <= opts.ftol.max(opts.xtol) {
        Some(x)
    } else {
        None
    }
}
