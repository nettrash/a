//! The nodes and weights of Gaussian quadrature.
//!
//! An `n`-point Gaussian rule integrates every polynomial of degree `2n − 1` exactly against its
//! weight function. Its nodes are the zeros of the `n`-th orthogonal polynomial — the eigenvalues
//! of the `n × n` Jacobi matrix of the three-term recurrence — and its weights are `μ₀` times the
//! squares of the first components of the unit eigenvectors, `μ₀` the total weight.
//!
//! The rules are computed by two methods, as the order decides:
//!
//! * up to order [`GOLUB_WELSCH_LARGEST`], Golub and Welsch's (1969): the nodes are the
//!   eigenvalues of the Jacobi matrix, from the Hermitian eigensolver of `a::math::matrix`
//!   ([`hermitian_eigenvalues`](crate::math::matrix::Matrix::hermitian_eigenvalues)), which costs
//!   `O(n³)` time and `O(n²)` memory;
//! * above it, Newton's method on the polynomial itself, from first guesses given by the
//!   asymptotic expansions of the zeros — Olver's in terms of the zeros of `J₀` and Tricomi's for
//!   Legendre, Langer's about the turning point for Hermite, both of them for Laguerre — in
//!   `O(n²)` time and `O(n)` memory: each of the `n` Newton iterations runs the three-term
//!   recurrence once, in `O(n)`, and nothing of size `n²` is formed.
//!
//! Either way the nodes are then finished the same way, by Newton's method on the orthonormal
//! recurrence, first in `f64` and then in double-double, so that each node is known to some 100
//! bits before it is rounded; and the weights are Golub and Welsch's `μ₀ v₀²` with the eigenvector
//! `v ∝ (p₀(x), …, p_{n−1}(x))` taken from the recurrence at that node, the Christoffel number
//! `μ₀ / Σₖ pₖ(x)²`. The eigensolver's own eigenvectors are not used for the weights: their small
//! components are accurate only absolutely, which leaves the small weights of the large rules as
//! noise. Gauss–Jacobi has no asymptotic first guesses here and takes the first method at every
//! order.
//!
//! The polynomials are rescaled by powers of two as the recurrence runs, so that the Hermite and
//! Laguerre rules of any order neither overflow nor underflow on the way: a weight too small for
//! the type comes out as a subnormal number or as zero, smoothly. Where the weights are too small
//! to be of use — the outer weights of the Hermite and Laguerre rules of a few hundred points —
//! [`hermite_scaled`] and [`laguerre_scaled`] give the same nodes with the weights `wᵢ e^(xᵢ²)`
//! and `wᵢ e^(xᵢ)`, which stay near 1, for integrals written without the weight function.
//!
//! Every rule is computed in `f64` and rounded to the type at the end.

mod dd;
mod guess;
mod newton;

use super::Real;
use dd::Dd;
use newton::{Recurrence, Zero, at, newton};

/// The largest order computed by Golub and Welsch's method; above it the first guesses are the
/// asymptotic ones. The two agree to the last bit around it (both end in the same Newton
/// iterations), and the choice is one of accuracy and cost: from order 100 on the asymptotic
/// guesses are within `1.4·10⁻³` of the spacing of the zeros (Laguerre's; Hermite's `1.4·10⁻⁴`,
/// Legendre's `10⁻⁸`, all falling with the order), so that Newton's method converges in a few
/// iterations and cannot slip to a neighbouring zero — which is checked all the same: the zeros
/// found must be distinct and in order, or the rule is Golub and Welsch's after all — while the
/// dense eigenproblem is already the larger part of the cost and grows as `n³`.
pub const GOLUB_WELSCH_LARGEST: usize = 100;

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
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::legendre;
    ///
    /// let rule = legendre::<f64>(3);
    /// let (x, w) = rule.as_slices();
    /// assert_eq!((x.len(), w.len()), (3, 3));
    /// assert_eq!(x[1], 0.0);
    /// ```
    pub fn as_slices(&self) -> (&[T], &[T]) {
        (&self.nodes, &self.weights)
    }

    /// The rule applied to `f`: `Σᵢ wᵢ f(xᵢ)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::hermite;
    ///
    /// // ∫ x² e^(−x²) dx over the line is √π/2.
    /// let got = hermite::<f64>(4).integrate(|x: f64| x * x);
    /// assert!((got - core::f64::consts::PI.sqrt() / 2.0).abs() < 1e-15);
    /// ```
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
/// The nodes are symmetric about 0 to the last bit, and the middle node of an odd order is 0.
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
///
/// // A rule of ten thousand points, by Newton's method in O(n²).
/// let big = legendre::<f64>(10_000);
/// assert!((big.weights.iter().sum::<f64>() - 2.0).abs() < 1e-13);
/// ```
#[track_caller]
pub fn legendre<T: Real>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let rec = Recurrence::legendre(n);
    let zeros = symmetric_zeros(&rec, guess::legendre);
    rule(&zeros, |z| z.weight(Dd::new(2.0), 0))
}

/// The nodes and weights of `n`-point Gauss–Hermite quadrature, which integrates against the
/// weight `e^(−x²)` on `(-∞, ∞)`; the nodes are the zeros of the physicists' `Hₙ` and the total
/// weight is `√π`.
///
/// The weight of the outermost nodes falls like `e^(−2n)`: below the smallest normal `f64` from
/// `n = 371` on (`f32`: 52), where it comes out as a subnormal number, and as zero from `n = 389`
/// (`f32`: 61). A sum of the rule's terms is then still right for an integrand that does not
/// grow like `e^(x²)`; for one that does, use [`hermite_scaled`] on the integrand times
/// `e^(−x²)`.
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
/// assert!((rule.integrate(|x: f64| x * x * x * x) - want).abs() < 1e-14);
/// ```
#[track_caller]
pub fn hermite<T: Real>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let rec = Recurrence::hermite(n);
    let zeros = symmetric_zeros(&rec, guess::hermite);
    rule(&zeros, |z| z.weight(Dd::SQRT_PI, 0))
}

/// The nodes of [`hermite`] and the weights `wᵢ e^(xᵢ²)`, which approximate `∫ g(x) dx` over the
/// line by `Σᵢ wᵢ e^(xᵢ²) g(xᵢ)`: these weights stay near `π/√(2n)` and never leave the range of
/// the type, where Hermite's own fall out of it from 371 points on.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use a::math::numerics::hermite_scaled;
///
/// // ∫ sech x dx over the line is π: the integrand decays like e^(−|x|), far more slowly than
/// // the weight e^(−x²), and the rule of 800 points needs its outermost weights.
/// let rule = hermite_scaled::<f64>(800);
/// let got = rule.integrate(|x: f64| 1.0 / x.cosh());
/// assert!((got - core::f64::consts::PI).abs() < 1e-6);
/// ```
#[track_caller]
pub fn hermite_scaled<T: Real>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let rec = Recurrence::hermite(n);
    let zeros = symmetric_zeros(&rec, guess::hermite);
    rule(&zeros, |z| z.scaled_weight(Dd::LN_SQRT_PI, |x| x.mul(x)))
}

/// The nodes and weights of `n`-point Gauss–Laguerre quadrature, which integrates against the
/// weight `e⁻ˣ` on `[0, ∞)`; the nodes are the zeros of `Lₙ` and the total weight is 1.
///
/// The weight of the largest node falls like `e^(−4n)`: below the smallest normal `f64` from
/// `n = 186` on (`f32`: 27), where it comes out as a subnormal number, and as zero from `n = 196`
/// (`f32`: 31); for an integrand that grows like `eˣ`, use [`laguerre_scaled`] on the integrand
/// times `e⁻ˣ`.
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
/// assert!((rule.integrate(|x: f64| x * x * x) - 6.0).abs() < 1e-13);
/// ```
#[track_caller]
pub fn laguerre<T: Real>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let zeros = laguerre_zeros(n);
    rule(&zeros, |z| z.weight(Dd::ONE, 0))
}

/// The nodes of [`laguerre`] and the weights `wᵢ eˣⁱ`, which approximate `∫₀^∞ g(x) dx` by
/// `Σᵢ wᵢ eˣⁱ g(xᵢ)`, and stay representable at every order.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use a::math::numerics::laguerre_scaled;
///
/// // ∫₀^∞ dx/(1 + x)² = 1, an integrand of algebraic decay.
/// let rule = laguerre_scaled::<f64>(400);
/// let got = rule.integrate(|x: f64| 1.0 / ((1.0 + x) * (1.0 + x)));
/// assert!((got - 1.0).abs() < 1e-3);
/// ```
#[track_caller]
pub fn laguerre_scaled<T: Real>(n: usize) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    let zeros = laguerre_zeros(n);
    rule(&zeros, |z| z.scaled_weight(Dd::ZERO, |x| x))
}

/// The nodes and weights of `n`-point Gauss–Jacobi quadrature, which integrates against the
/// weight `(1 − x)^α (1 + x)^β` on `[-1, 1]`; the nodes are the zeros of `Pₙ^(α,β)` and the total
/// weight is `2^(α+β+1) B(α+1, β+1)`.
///
/// The nodes are Golub and Welsch's at every order (`O(n³)` time and `O(n²)` memory), finished
/// by Newton's method like the other rules. Where `2^(α+β+1)` or `B(α+1, β+1)` leaves the range
/// of an `f64` — `α + β` above about 1000 — the total weight is formed from their logarithms,
/// which cancel and cost about `(α + β)/4` units in the last place of every weight (measured on
/// `α = β`: 260 ulps at `10³`, `10⁻⁸` relatively at `10⁸`, `10⁻⁴` at `10¹²`, 11% at `10¹⁵`). From
/// `α + β = 2⁵³` on, where that is half the weight itself, no digit of it is known, and the
/// weights are NaN. A total weight beyond the range of the type makes the weights infinite.
///
/// # Panics
///
/// Panics if `n` is zero, or if `α` or `β` is not a finite number greater than `−1`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::jacobi;
///
/// // α = β = −1/2 is Gauss–Chebyshev of the first kind, whose weight is 1/√(1 − x²).
/// let rule = jacobi::<f64>(6, -0.5, -0.5);
/// let want = core::f64::consts::PI;
/// assert!((rule.integrate(|_: f64| 1.0) - want).abs() < 1e-14);
/// ```
#[track_caller]
pub fn jacobi<T: Real>(n: usize, alpha: T, beta: T) -> NodesWeights<T> {
    assert!(n > 0, "the number of nodes is zero");
    assert!(
        alpha > -T::ONE && beta > -T::ONE && alpha.is_finite() && beta.is_finite(),
        "the exponents must be finite and greater than −1"
    );
    let (alpha, beta) = (alpha.to_f64(), beta.to_f64());
    let rec = Recurrence::jacobi(n, alpha, beta);
    // With α = β the diagonal is exactly 0 and the rule symmetric: it is mirrored like
    // Legendre's, its middle node exactly 0.
    let zeros = if alpha == beta {
        symmetric_from_eigenvalues(&rec)
    } else {
        general_zeros(&rec, None, -1.0, 1.0)
    };
    let (mu0, mu0_exp) = jacobi_total(alpha, beta);
    rule(&zeros, |z| z.weight(mu0, mu0_exp))
}

/// The rule of the zeros, with the weights `weight` gives, rounded to `T`.
fn rule<T: Real>(zeros: &[Zero], weight: impl Fn(&Zero) -> f64) -> NodesWeights<T> {
    NodesWeights {
        nodes: zeros.iter().map(|z| T::from_f64(z.x.value())).collect(),
        weights: zeros.iter().map(|z| T::from_f64(weight(z))).collect(),
    }
}

/// The zeros of a family symmetric about 0 (Legendre's, Hermite's), ascending: Newton's method
/// on the positive ones, from `asymptotic`'s guesses (largest first) above
/// [`GOLUB_WELSCH_LARGEST`] and from the eigenvalues up to it, mirrored, with the zero 0 exactly
/// in the middle of an odd order.
fn symmetric_zeros(rec: &Recurrence, asymptotic: fn(usize) -> Vec<f64>) -> Vec<Zero> {
    let n = rec.degree();
    if n > GOLUB_WELSCH_LARGEST
        && let Some(positive) = refine_all(rec, &asymptotic(n), Order::Descending, 0.0, f64::MAX)
    {
        return mirror(rec, &positive);
    }
    symmetric_from_eigenvalues(rec)
}

/// The zeros of a symmetric family from the eigenvalues of its Jacobi matrix.
fn symmetric_from_eigenvalues(rec: &Recurrence) -> Vec<Zero> {
    let n = rec.degree();
    // The positive eigenvalues, largest first.
    let positive: Vec<f64> = rec.eigenvalues().into_iter().rev().take(n / 2).collect();
    let refined = refine_all(rec, &positive, Order::Descending, 0.0, f64::MAX)
        .unwrap_or_else(|| kept(rec, &positive));
    mirror(rec, &refined)
}

/// The positive zeros `positive` (largest first) and their mirror images, ascending.
fn mirror(rec: &Recurrence, positive: &[Zero]) -> Vec<Zero> {
    let n = rec.degree();
    let mut all = Vec::with_capacity(n);
    all.extend(positive.iter().map(Zero::mirrored));
    if n % 2 == 1 {
        all.push(at(rec, Dd::ZERO));
    }
    all.extend(positive.iter().rev().copied());
    all
}

/// The zeros of `Lₙ`, ascending.
fn laguerre_zeros(n: usize) -> Vec<Zero> {
    let rec = Recurrence::laguerre(n);
    let asymptotic = (n > GOLUB_WELSCH_LARGEST).then(|| guess::laguerre(n));
    general_zeros(&rec, asymptotic, 0.0, f64::MAX)
}

/// The zeros of a family without symmetry, ascending, inside `(lo, hi)`: from the asymptotic
/// guesses where there are some, else (or should they fail) from the eigenvalues.
fn general_zeros(rec: &Recurrence, asymptotic: Option<Vec<f64>>, lo: f64, hi: f64) -> Vec<Zero> {
    if let Some(g) = asymptotic
        && let Some(z) = refine_all(rec, &g, Order::Ascending, lo, hi)
    {
        return z;
    }
    let eig = rec.eigenvalues();
    refine_all(rec, &eig, Order::Ascending, lo, hi).unwrap_or_else(|| kept(rec, &eig))
}

/// The order a list of zeros must be in.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    Ascending,
    Descending,
}

/// Newton's method from every guess, or `None` should one of the iterations fail or the zeros
/// found not be distinct, in `order`, and inside `(lo, hi)`: `n` distinct zeros of a polynomial
/// of degree `n` are all of them, so a set that passes is the rule.
fn refine_all(
    rec: &Recurrence,
    guesses: &[f64],
    order: Order,
    lo: f64,
    hi: f64,
) -> Option<Vec<Zero>> {
    let zeros = guesses
        .iter()
        .map(|&g| newton(rec, g))
        .collect::<Option<Vec<_>>>()?;
    let inside = zeros.iter().all(|z| z.x.hi > lo && z.x.hi < hi);
    let ordered = zeros.windows(2).all(|w| match order {
        Order::Ascending => w[0].x.hi < w[1].x.hi,
        Order::Descending => w[0].x.hi > w[1].x.hi,
    });
    (inside && ordered).then_some(zeros)
}

/// The eigenvalues as they are, with the weights at them: what is left should Newton's method
/// from the eigenvalues ever fail, which no order tested does.
fn kept(rec: &Recurrence, eig: &[f64]) -> Vec<Zero> {
    eig.iter().map(|&x| at(rec, Dd::new(x))).collect()
}

/// The total weight of the Jacobi weight function, `2^(α+β+1) B(α+1, β+1)`, as a double-double
/// and a power of two: directly where both factors are in range, from their logarithms
/// elsewhere.
fn jacobi_total(alpha: f64, beta: f64) -> (Dd, i64) {
    use crate::math::special::{beta as beta_fn, lbeta};
    let s = alpha + beta + 1.0;
    // 2^α · 2^β · 2 rather than 2^(α+β+1): the sum is rounded where the exponents differ in
    // size (0.3 + 150 by 10⁻¹⁴), and its error would become a relative error of the weights.
    let p = 2.0 * alpha.exp2() * beta.exp2();
    let b = beta_fn(alpha + 1.0, beta + 1.0);
    let v = p * b;
    if p.is_normal() && b.is_normal() && v.is_normal() {
        return (Dd::new(v), 0);
    }
    // The two logarithms below cancel, and the weights lose about (α + β)/4 ulps; from 2⁵³ on
    // that is half of them, and no digit is known.
    if alpha + beta >= 2f64.powi(53) {
        return (Dd::new(f64::NAN), 0);
    }
    // ln μ₀ / ln 2 = s + log₂ B, split into an integer and a fraction.
    let t = s + lbeta(alpha + 1.0, beta + 1.0) / core::f64::consts::LN_2;
    let e = t.floor();
    (Dd::new((t - e).exp2()), e as i64)
}
