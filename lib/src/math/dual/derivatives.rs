//! Derivatives, gradients, Jacobians and Hessians by passes of dual and hyper-dual numbers, in the
//! shapes the solvers of [`numerics`](crate::math::numerics) take.
//!
//! Forward mode carries one direction a pass: a derivative is one pass of [`Dual`] numbers, a
//! second derivative one of [`HyperDual`] numbers, a gradient or a Jacobian of `n` variables `n`
//! passes, one for each variable (a column of the Jacobian each), and a Hessian `n(n + 1)/2`
//! passes, one for each entry on and above the diagonal, with `ε₁` on `xᵢ` and `ε₂` on `xⱼ`. Each
//! result is exact up to the rounding of the arithmetic of the pass, but for one rule: a tangent
//! part that is zero contributes nothing, also against an infinite derivative, which keeps a
//! gradient usable when a variable that does not move sits at a singular point (`√y` at `y = 0`)
//! and gives 0 where an intermediate whose derivative is 0 (or underflows) meets an infinite
//! derivative and the true limit of `0·∞` is not 0 (see [`DualNum`]). In a pass a product takes
//! three products of floats (dual) or nine (hyper-dual), and an elementary function its value and
//! its derivatives, so a pass costs a small multiple of an evaluation of the function.

use super::{Dual, DualNum, HyperDual};
use crate::math::matrix::Matrix;

/// The number of rows of `x`, which must be a column.
///
/// # Panics
///
/// Panics if `x` has other than one column.
#[track_caller]
fn column_length<T>(x: &Matrix<T>) -> usize {
    assert!(
        x.cols() == 1,
        "the point is {}×{}, not a column",
        x.rows(),
        x.cols()
    );
    x.rows()
}

/// The derivative of `f` at `x`, by one pass of dual numbers: the `ε` part of `f(x + ε)`.
///
/// `f` is written over dual numbers, as a closure or as a function generic over [`DualNum`]
/// passed by name. `T` is `f32` or `f64`, or itself a dual number, so that derivatives nest.
///
/// The result is exact up to rounding but for the rule that a zero tangent part contributes
/// nothing (see [`DualNum`]): an intermediate whose derivative is 0 at `x` (or underflows), met by
/// an infinite derivative, gives 0 where the true limit of `0·∞` is not 0 — the derivative of
/// `cbrt(x³)` at 0 comes back as 0, not 1, and so does that of `(x³)^(1/3)`, of `cbrt(sin³x)` and,
/// where `x³` underflows, of `cbrt(x³)` at `10⁻³⁰⁰`. num-dual gives NaN there.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, DualNum, derivative};
/// use a::math::numerics::newton;
///
/// // The root of e^x − 3x² nearest 1 by Newton's method, with the exact derivative.
/// fn f<D: DualNum<Float = f64>>(x: D) -> D {
///     x.exp() - x * x * 3.0
/// }
/// let x = newton(1.0, f, |x| derivative(f, x)).unwrap();
/// assert!((x - 0.910_007_572_488_709).abs() < 1e-15);
///
/// // A closure over dual numbers, and the derivative of a derivative.
/// assert_eq!(derivative(|x: Dual<f64>| x * x * x, 2.0), 12.0);
/// let f2 = derivative(|x: Dual<f64>| derivative(|y: Dual<Dual<f64>>| y * y * y, x), 2.0);
/// assert_eq!(f2, 12.0);
/// ```
pub fn derivative<T: DualNum>(mut f: impl FnMut(Dual<T>) -> Dual<T>, x: T) -> T {
    f(Dual::new(x, T::ONE)).eps
}

/// The value and the derivative of `f` at `x`, from one pass of dual numbers, as
/// [`derivative`] forms it.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, DualNum, value_and_derivative};
///
/// let (v, d) = value_and_derivative(|x: Dual<f64>| x.ln() * x, 1.0);
/// assert_eq!((v, d), (0.0, 1.0));
/// ```
pub fn value_and_derivative<T: DualNum>(mut f: impl FnMut(Dual<T>) -> Dual<T>, x: T) -> (T, T) {
    let r = f(Dual::new(x, T::ONE));
    (r.re, r.eps)
}

/// The second derivative of `f` at `x`, by one pass of hyper-dual numbers: the `ε₁ε₂` part of
/// `f(x + ε₁ + ε₂)`.
///
/// Exact up to rounding but for the rule of [`derivative`] at `0·∞`: the second derivative of
/// `cbrt(x⁶)` at 0 comes back as 0, not 2.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, derivative, second_derivative};
/// use a::math::numerics::halley;
///
/// // The root of cos x = x by Halley's method, with both derivatives exact.
/// fn f<D: DualNum<Float = f64>>(x: D) -> D {
///     x.cos() - x
/// }
/// let x = halley(0.5, f, |x| derivative(f, x), |x| second_derivative(f, x)).unwrap();
/// assert!((x - 0.739_085_133_215_160_7).abs() < 1e-15);
/// assert_eq!(second_derivative(f, 0.0), -1.0);
/// ```
pub fn second_derivative<T: DualNum>(mut f: impl FnMut(HyperDual<T>) -> HyperDual<T>, x: T) -> T {
    f(HyperDual::new(x, T::ONE, T::ONE, T::ZERO)).eps1eps2
}

/// The value, the derivative and the second derivative of `f` at `x`, from one pass of
/// hyper-dual numbers, as [`second_derivative`] forms it.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, HyperDual, value_and_derivatives};
///
/// let (v, d1, d2) = value_and_derivatives(|x: HyperDual<f64>| x.powi(4), 2.0);
/// assert_eq!((v, d1, d2), (16.0, 32.0, 48.0));
/// ```
pub fn value_and_derivatives<T: DualNum>(
    mut f: impl FnMut(HyperDual<T>) -> HyperDual<T>,
    x: T,
) -> (T, T, T) {
    let r = f(HyperDual::new(x, T::ONE, T::ONE, T::ZERO));
    (r.re, r.eps1, r.eps1eps2)
}

/// The point `x` as constants of the type `D`, made by `make`.
fn constants<T: DualNum, D>(x: &Matrix<T>, make: impl Fn(T) -> D) -> Vec<D> {
    x.as_slice().iter().map(|&v| make(v)).collect()
}

/// The gradient of `f` at the point `x`, an `n × 1` column, by `n` passes of dual numbers: in
/// pass `i` the variable `xᵢ` is `xᵢ + ε` and the others are constants, and the `ε` part of the
/// result is `∂f/∂xᵢ`. An empty point has an empty gradient, without a call of `f`.
///
/// `f` takes the variables as a slice, as [`Matrix::as_slice`] gives a column, so that one
/// function generic over [`DualNum`] serves for the value, as `f(x.as_slice())`, and for the
/// gradient, as `gradient(f, x)`: what [`bfgs`](crate::math::numerics::bfgs),
/// [`conjugate_gradient`](crate::math::numerics::conjugate_gradient) and
/// [`lbfgsb`](crate::math::numerics::lbfgsb()) take.
///
/// The rule that a zero tangent part contributes nothing (see [`DualNum`]) holds here too: it
/// keeps a gradient usable where another variable sits at a point of infinite derivative (as `√y`
/// at `y = 0`), and it gives 0 where an intermediate whose derivative is 0 meets an infinite one
/// and the true limit of `0·∞` is not 0 — `cbrt(x³)` has the derivative 0 at 0 here, not 1. Likewise the
/// gradient of `cbrt(x³ + y³)` at the origin comes back as `(0, 0)`, not `(1, 1)`.
///
/// # Panics
///
/// Panics if `x` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, gradient};
/// use a::math::matrix::Matrix;
/// use a::math::numerics::bfgs;
///
/// // Rosenbrock's function, once, for its values and its gradient.
/// fn rosen<D: DualNum<Float = f64>>(x: &[D]) -> D {
///     let a = D::constant(1.0) - x[0];
///     let b = x[1] - x[0] * x[0];
///     a * a + b * b * 100.0
/// }
/// let x0 = Matrix::from_column_major(2, 1, vec![-1.2, 1.0]);
/// let g = gradient(rosen, &x0);
/// assert!((g[(0, 0)] + 215.6).abs() < 1e-12 && (g[(1, 0)] + 88.0).abs() < 1e-12);
/// let r = bfgs(|x| rosen(x.as_slice()), |x| gradient(rosen, x), &x0).unwrap();
/// assert!(r.converged);
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 1.0).abs() < 1e-7);
/// ```
#[track_caller]
pub fn gradient<T: DualNum>(f: impl FnMut(&[Dual<T>]) -> Dual<T>, x: &Matrix<T>) -> Matrix<T> {
    let n = column_length(x);
    if n == 0 {
        return Matrix::from_column_major(0, 1, Vec::new());
    }
    value_and_gradient(f, x).1
}

/// The value and the gradient of `f` at the point `x`, from the `n` passes of [`gradient`];
/// for an empty point, the value from one call of `f`.
///
/// # Panics
///
/// Panics if `x` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, value_and_gradient};
/// use a::math::matrix::Matrix;
///
/// // f(x, y) = x²·y at (3, 2): 18, with the gradient (2xy, x²) = (12, 9).
/// let x = Matrix::from_column_major(2, 1, vec![3.0, 2.0]);
/// let (v, g) = value_and_gradient(|x: &[Dual<f64>]| x[0] * x[0] * x[1], &x);
/// assert_eq!(v, 18.0);
/// assert_eq!(g.as_slice(), &[12.0, 9.0]);
/// ```
#[track_caller]
pub fn value_and_gradient<T: DualNum>(
    mut f: impl FnMut(&[Dual<T>]) -> Dual<T>,
    x: &Matrix<T>,
) -> (T, Matrix<T>) {
    let n = column_length(x);
    let mut point = constants(x, |v| Dual::new(v, T::ZERO));
    if n == 0 {
        return (f(&point).re, Matrix::from_column_major(0, 1, Vec::new()));
    }
    let mut value = T::ZERO;
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        point[i].eps = T::ONE;
        let r = f(&point);
        point[i].eps = T::ZERO;
        if i == 0 {
            value = r.re;
        }
        g.push(r.eps);
    }
    (value, Matrix::from_column_major(n, 1, g))
}

/// The `m × n` Jacobian of `f` at the point `x`, an `n × 1` column, by `n` passes of dual
/// numbers: in pass `j` the variable `xⱼ` is `xⱼ + ε`, and the `ε` parts of the `m` results are
/// column `j`, `∂fᵢ/∂xⱼ`. `f` returns its `m` values as anything that is a slice of dual
/// numbers — a `Vec`, an array — of the same length in every pass. An empty point gives an
/// `m × 0` Jacobian, `m` from one call of `f`.
///
/// The Jacobian is what [`newton_system`](crate::math::numerics::newton_system),
/// [`broyden_with_jacobian`](crate::math::numerics::broyden_with_jacobian),
/// [`gauss_newton`](crate::math::numerics::gauss_newton) and
/// [`levenberg_marquardt`](crate::math::numerics::levenberg_marquardt) take.
///
/// The rule that a zero tangent part contributes nothing (see [`DualNum`]) holds here too: it
/// keeps a Jacobian usable where another variable sits at a point of infinite derivative (as `√y`
/// at `y = 0`), and it gives 0 where an intermediate whose derivative is 0 meets an infinite one
/// and the true limit of `0·∞` is not 0 — `cbrt(x³)` has the derivative 0 at 0 here, not 1.
///
/// # Panics
///
/// Panics if `x` is not a column, or if `f` returns values of different lengths in two passes.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, jacobian};
/// use a::math::matrix::Matrix;
/// use a::math::numerics::newton_system;
///
/// // The circle of radius 2 and the line y = x − 1, once, for the residual and its Jacobian.
/// fn f<D: DualNum<Float = f64>>(x: &[D]) -> [D; 2] {
///     [x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1] - 1.0]
/// }
/// let column = |v: [f64; 2]| Matrix::from_column_major(2, 1, v.to_vec());
/// let x0 = column([1.5, 0.5]);
/// assert_eq!(jacobian(f, &x0), Matrix::from_rows(&[[3.0, 1.0], [1.0, -1.0]]));
/// let r = newton_system(&x0, |x| column(f(x.as_slice())), |x| jacobian(f, x)).unwrap();
/// assert!(r.converged);
/// assert!((r.x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-15);
/// ```
#[track_caller]
pub fn jacobian<T: DualNum, R: AsRef<[Dual<T>]>>(
    f: impl FnMut(&[Dual<T>]) -> R,
    x: &Matrix<T>,
) -> Matrix<T> {
    value_and_jacobian(f, x).1
}

/// The values of `f` at the point `x`, an `m × 1` column, and its `m × n` Jacobian there, from
/// the `n` passes of [`jacobian`].
///
/// # Panics
///
/// Panics if `x` is not a column, or if `f` returns values of different lengths in two passes.
///
/// # Examples
///
/// ```
/// use a::math::dual::{Dual, DualNum, value_and_jacobian};
/// use a::math::matrix::Matrix;
///
/// // Polar to Cartesian coordinates at r = 2, θ = 0.
/// let p = Matrix::from_column_major(2, 1, vec![2.0, 0.0]);
/// let (v, j) = value_and_jacobian(|p: &[Dual<f64>]| vec![p[0] * p[1].cos(), p[0] * p[1].sin()], &p);
/// assert_eq!(v.as_slice(), &[2.0, 0.0]);
/// assert_eq!(j, Matrix::from_rows(&[[1.0, 0.0], [0.0, 2.0]]));
/// ```
#[track_caller]
pub fn value_and_jacobian<T: DualNum, R: AsRef<[Dual<T>]>>(
    mut f: impl FnMut(&[Dual<T>]) -> R,
    x: &Matrix<T>,
) -> (Matrix<T>, Matrix<T>) {
    let n = column_length(x);
    let mut point = constants(x, |v| Dual::new(v, T::ZERO));
    if n == 0 {
        let r = f(&point);
        let values: Vec<T> = r.as_ref().iter().map(|d| d.re).collect();
        let m = values.len();
        return (
            Matrix::from_column_major(m, 1, values),
            Matrix::from_column_major(m, 0, Vec::new()),
        );
    }
    let mut values = Vec::new();
    let mut data = Vec::new();
    let mut m = 0;
    for j in 0..n {
        point[j].eps = T::ONE;
        let r = f(&point);
        point[j].eps = T::ZERO;
        let r = r.as_ref();
        if j == 0 {
            m = r.len();
            values = r.iter().map(|d| d.re).collect();
            data.reserve(m.checked_mul(n).unwrap_or(0));
        } else {
            assert!(
                r.len() == m,
                "the function gave {} values in pass {j}, and {m} in the first",
                r.len()
            );
        }
        data.extend(r.iter().map(|d| d.eps));
    }
    (
        Matrix::from_column_major(m, 1, values),
        Matrix::from_column_major(m, n, data),
    )
}

/// The `n × n` Hessian of `f` at the point `x`, an `n × 1` column, by `n(n + 1)/2` passes of
/// hyper-dual numbers: in the pass for `i ≤ j` the variable `xᵢ` carries `ε₁` and `xⱼ` carries
/// `ε₂` (`xᵢ` both when `i = j`), and the `ε₁ε₂` part of the result is `∂²f/∂xᵢ∂xⱼ`, written to
/// both `(i, j)` and `(j, i)`, so the matrix is symmetric. An empty point has an empty Hessian,
/// without a call of `f`.
///
/// The rule that a zero tangent part contributes nothing (see [`DualNum`]) holds here too: it
/// keeps a Hessian usable where another variable sits at a point of infinite derivative (as `√y`
/// at `y = 0`), and it gives 0 where an intermediate whose derivative is 0 meets an infinite one
/// and the true limit of `0·∞` is not 0 — `cbrt(x³)` has the derivative 0 at 0 here, not 1.
///
/// # Panics
///
/// Panics if `x` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, hessian};
/// use a::math::matrix::Matrix;
///
/// // f(x, y) = x²y + y³ at (1, 2): the Hessian [[2y, 2x], [2x, 6y]] = [[4, 2], [2, 12]].
/// fn f<D: DualNum<Float = f64>>(x: &[D]) -> D {
///     x[0] * x[0] * x[1] + x[1].powi(3)
/// }
/// let h = hessian(f, &Matrix::from_column_major(2, 1, vec![1.0, 2.0]));
/// assert_eq!(h, Matrix::from_rows(&[[4.0, 2.0], [2.0, 12.0]]));
/// ```
#[track_caller]
pub fn hessian<T: DualNum>(
    f: impl FnMut(&[HyperDual<T>]) -> HyperDual<T>,
    x: &Matrix<T>,
) -> Matrix<T> {
    let n = column_length(x);
    if n == 0 {
        return Matrix::from_column_major(0, 0, Vec::new());
    }
    value_gradient_and_hessian(f, x).2
}

/// The value, the gradient and the Hessian of `f` at the point `x`, from the `n(n + 1)/2` passes
/// of [`hessian`]: the passes on the diagonal carry the gradient in their `ε₁` parts. For an
/// empty point, the value from one call of `f`.
///
/// # Panics
///
/// Panics if `x` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::dual::{DualNum, value_gradient_and_hessian};
/// use a::math::matrix::Matrix;
///
/// // Newton's method, x ← x − H⁻¹·∇f, for the minimum of the convex e^(x−1) + e^(−y) + (x − y)²,
/// // which lies on the line x + y = 1.
/// fn f<D: DualNum<Float = f64>>(x: &[D]) -> D {
///     (x[0] - 1.0).exp() + (-x[1]).exp() + (x[0] - x[1]).powi(2)
/// }
/// let mut x = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
/// for _ in 0..6 {
///     let (_, g, h) = value_gradient_and_hessian(f, &x);
///     x = &x - &h.solve(&g).unwrap();
/// }
/// let (_, g, _) = value_gradient_and_hessian(f, &x);
/// assert!(g.as_slice().iter().all(|v| v.abs() < 1e-15));
/// assert!((x[(0, 0)] + x[(1, 0)] - 1.0).abs() < 1e-15);
/// ```
#[track_caller]
pub fn value_gradient_and_hessian<T: DualNum>(
    mut f: impl FnMut(&[HyperDual<T>]) -> HyperDual<T>,
    x: &Matrix<T>,
) -> (T, Matrix<T>, Matrix<T>) {
    let n = column_length(x);
    let mut point = constants(x, |v| HyperDual::new(v, T::ZERO, T::ZERO, T::ZERO));
    if n == 0 {
        return (
            f(&point).re,
            Matrix::from_column_major(0, 1, Vec::new()),
            Matrix::from_column_major(0, 0, Vec::new()),
        );
    }
    let size = n
        .checked_mul(n)
        .expect("the Hessian has more entries than a vector can hold");
    let mut h = vec![T::ZERO; size];
    let mut g = vec![T::ZERO; n];
    let mut value = T::ZERO;
    for i in 0..n {
        point[i].eps1 = T::ONE;
        for j in i..n {
            point[j].eps2 = T::ONE;
            let r = f(&point);
            point[j].eps2 = T::ZERO;
            if j == i {
                g[i] = r.eps1;
                if i == 0 {
                    value = r.re;
                }
            }
            h[i + j * n] = r.eps1eps2;
            h[j + i * n] = r.eps1eps2;
        }
        point[i].eps1 = T::ZERO;
    }
    (
        value,
        Matrix::from_column_major(n, 1, g),
        Matrix::from_column_major(n, n, h),
    )
}
