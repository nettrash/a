//! The nodes and weights of Gaussian quadrature.
//!
//! An `n`-point Gaussian rule integrates every polynomial of degree `2n − 1` exactly against its
//! weight function: the nodes are the eigenvalues of the `n × n` Jacobi matrix of the three-term
//! recurrence of the orthogonal polynomials, and the weights `μ₀` times the squares of the first
//! components of the eigenvectors — Golub and Welsch's method, which this runs on
//! [`hermitian_eigen`](crate::math::matrix::Matrix::hermitian_eigen). The nodes of the largest
//! orders are refined by Newton's method on the polynomial itself, which the eigensolver's
//! accuracy does not reach.

use super::Real;
use crate::math::complex::Float;
use crate::math::matrix::Matrix;
use crate::math::matrix::Scalar;
use crate::math::special::{beta as beta_fn, legendre_p};

/// The `n` nodes `xᵢ` and the `n` weights `wᵢ` of a Gaussian rule, ascending in `xᵢ`. The rule
/// approximates `∫ f(x) ω(x) dx` by `Σᵢ wᵢ f(xᵢ)`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodesWeights<T> {
    /// The nodes, ascending.
    pub nodes: Vec<T>,
    /// The weights, in the same order as the nodes.
    pub weights: Vec<T>,
}

impl<T: Real> NodesWeights<T> {
    /// The nodes and weights as a pair of slices.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        (&self.nodes, &self.weights)
    }

    /// The rule applied to `f`: `Σᵢ wᵢ f(xᵢ)`.
    pub fn integrate(&self, mut f: impl FnMut(T) -> T) -> T {
        let mut s = T::ZERO;
        for (x, w) in self.nodes.iter().zip(&self.weights) {
            s = w.mul_add(f(*x), s);
        }
        s
    }
}

/// The nodes and weights of `n`-point Gauss–Legendre quadrature, which integrates against the
/// weight 1 on `[-1, 1]`; the nodes are the zeros of `Pₙ` and the total weight is 2.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use a::math::numerics::legendre;
///
/// let rule = legendre::<f64>(5);
/// // Every polynomial of degree 9 or less: ∫₋₁¹ x⁶ dx = 2/7.
/// assert!((rule.integrate(|x: f64| x.powi(6)) - 2.0 / 7.0).abs() < 1e-15);
/// ```
pub fn legendre<T: Real + Scalar<Real = T> + crate::math::special::Real>(
    n: usize,
) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let off = |k: usize| {
        let k = T::from_f64(k as f64);
        k / Float::sqrt(T::from_f64(4.0) * k * k - T::ONE)
    };
    let mut rule = golub_welsch(n, |_| T::ZERO, off, T::TWO);
    refine_legendre(&mut rule);
    rule
}

/// The nodes and weights of `n`-point Gauss–Laguerre quadrature, which integrates against the
/// weight `e⁻ˣ` on `[0, ∞)`; the nodes are the zeros of `Lₙ` and the total weight is 1.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use a::math::numerics::laguerre;
///
/// let rule = laguerre::<f64>(8);
/// // ∫₀^∞ x³ e⁻ˣ dx = 3! = 6.
/// assert!((rule.integrate(|x: f64| x * x * x) - 6.0).abs() < 1e-12);
/// ```
pub fn laguerre<T: Real + Scalar<Real = T>>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let diag = |k: usize| T::from_f64(2.0 * k as f64 + 1.0);
    // The recurrence of the monic Laguerre polynomials has γₖ = k², so the off-diagonal of the
    // symmetric Jacobi matrix is k itself and not its square root.
    let off = |k: usize| T::from_f64(k as f64);
    golub_welsch(n, diag, off, T::ONE)
}

/// The nodes and weights of `n`-point Gauss–Hermite quadrature, which integrates against the
/// weight `e^(−x²)` on `(-∞, ∞)`; the nodes are the zeros of the physicists' `Hₙ` and the total
/// weight is `√π`.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use a::math::numerics::hermite;
///
/// let rule = hermite::<f64>(10);
/// // ∫₋∞^∞ x⁴ e^(−x²) dx = 3√π/4.
/// let want = 3.0 * core::f64::consts::PI.sqrt() / 4.0;
/// assert!((rule.integrate(|x: f64| x * x * x * x) - want).abs() < 1e-12);
/// ```
pub fn hermite<T: Real + Scalar<Real = T>>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let off = |k: usize| Float::sqrt(T::from_f64(k as f64) * T::HALF);
    let mu0 = Float::sqrt(T::PI);
    golub_welsch(n, |_| T::ZERO, off, mu0)
}

/// The nodes and weights of `n`-point Gauss–Jacobi quadrature, which integrates against the
/// weight `(1 − x)^α (1 + x)^β` on `[-1, 1]`; the nodes are the zeros of `Pₙ^(α,β)` and the total
/// weight is `2^(α+β+1) B(α+1, β+1)`.
///
/// # Panics
///
/// Panics if `n` is zero, or if `α` or `β` is at most `−1`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::jacobi;
///
/// // α = β = −1/2 is Gauss–Chebyshev of the first kind, whose weight is 1/√(1 − x²).
/// let rule = jacobi::<f64>(6, -0.5, -0.5);
/// let want = core::f64::consts::PI;
/// assert!((rule.integrate(|_: f64| 1.0) - want).abs() < 1e-13);
/// ```
pub fn jacobi<T: Real + Scalar<Real = T> + crate::math::special::Real>(
    n: usize,
    alpha: T,
    beta: T,
) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    assert!(
        alpha > -T::ONE && beta > -T::ONE,
        "the exponents must be greater than −1"
    );
    let ab = alpha + beta;
    let diag = |k: usize| {
        if k == 0 {
            (beta - alpha) / (ab + T::TWO)
        } else {
            let k = T::from_f64(k as f64);
            (beta * beta - alpha * alpha) / ((T::TWO * k + ab) * (T::TWO * k + ab + T::TWO))
        }
    };
    let off = |k: usize| {
        let k = T::from_f64(k as f64);
        let s = T::TWO * k + ab;
        if k == T::ONE {
            // γ₁ in full is 4(1+α)(1+β)(1+α+β) / [(2+α+β)²(3+α+β)(1+α+β)]: the factor that
            // vanishes at α+β = −1 above and below cancels before either is divided by, which
            // is what makes the Chebyshev case α = β = −1/2 come out at 1/2 rather than 0/0.
            let g = T::from_f64(4.0) * (T::ONE + alpha) * (T::ONE + beta) / (s * s * (s + T::ONE));
            return Float::sqrt(g);
        }
        let num = T::from_f64(4.0) * k * (k + alpha) * (k + beta) * (k + ab);
        let den = s * s * (s + T::ONE) * (s - T::ONE);
        Float::sqrt(num / den)
    };
    let mu0 = T::from_f64(2.0).powf(ab + T::ONE) * beta_fn(alpha + T::ONE, beta + T::ONE);
    golub_welsch(n, diag, off, mu0)
}

/// Golub and Welsch's method: the eigenvalues of the symmetric tridiagonal matrix with the
/// diagonal `a` and the off-diagonal `b`, and `mu0` times the squares of the first components of
/// the eigenvectors.
fn golub_welsch<T: Real + Scalar<Real = T>>(
    n: usize,
    mut a: impl FnMut(usize) -> T,
    mut b: impl FnMut(usize) -> T,
    mu0: T,
) -> NodesWeights<T> {
    let mut j = Matrix::<T>::zeros(n, n);
    for k in 0..n {
        j[(k, k)] = a(k);
        if k + 1 < n {
            let v = b(k + 1);
            j[(k, k + 1)] = v;
            j[(k + 1, k)] = v;
        }
    }
    let eigen = j.hermitian_eigen();
    let nodes = eigen.values().to_vec();
    let vectors = eigen.vectors();
    let weights = (0..n)
        .map(|k| mu0 * vectors[(0, k)] * vectors[(0, k)])
        .collect();
    NodesWeights { nodes, weights }
}

/// Refines the Legendre nodes by Newton's method on `Pₙ`, which recovers the last digits the
/// eigensolver leaves out. The zeros are symmetric, so half of them are refined and the rest
/// mirrored.
fn refine_legendre<T: Real + crate::math::special::Real>(rule: &mut NodesWeights<T>) {
    let n = rule.nodes.len();
    if n < 2 {
        return;
    }
    let n32 = n as u32;
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut x = rule.nodes[i];
        for _ in 0..8 {
            let p = legendre_p(n32, x);
            let pm = legendre_p(n32 - 1, x);
            // (x² − 1) Pₙ′ = n (x Pₙ − Pₙ₋₁)
            let dp = (T::from_f64(n as f64) * (x * p - pm)) / (x * x - T::ONE);
            let d = p / dp;
            x -= d;
            if d.abs() <= T::EPSILON * x.abs() {
                break;
            }
        }
        rule.nodes[i] = x;
        rule.nodes[n - 1 - i] = -x;
    }
    // The weights follow from the nodes: wᵢ = 2 / ((1 − xᵢ²) Pₙ′(xᵢ)²).
    for i in 0..n {
        let x = rule.nodes[i];
        let pm = legendre_p(n32 - 1, x);
        let dp = (T::from_f64(n as f64) * (x * legendre_p(n32, x) - pm)) / (x * x - T::ONE);
        rule.weights[i] = T::TWO / ((T::ONE - x * x) * dp * dp);
    }
}
