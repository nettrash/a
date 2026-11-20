//! The integral of a function of one real variable, and of several.
//!
//! The workhorse is adaptive Gauss–Kronrod quadrature: a Gauss rule of `n` points and its
//! Kronrod extension of `2n + 1` points share their nodes, so the difference of the two estimates
//! is a cheap and reliable error estimate for the interval. This has the pairs G7–K15 and
//! G10–K21; [`quad`] uses the latter, which is about as accurate per evaluation as the former is
//! per unit of work. Where the integrand is smooth, [`clenshaw_curtis`] — the same rule on the
//! Chebyshev nodes, refined by bisection — reuses its evaluations between the levels; where the
//! integrand or one of its derivatives blows up at an end, [`tanh_sinh`] converges faster than
//! any of them.
//!
//! Infinite intervals are mapped onto finite ones and integrated with the same rules; the
//! endpoints of the map are where the difficulty is, and the tanh-sinh rule handles them best.
//!
//! In several dimensions [`cubature`] bisects the tensor-product Kronrod rule where the error is
//! largest, and [`monte_carlo`] estimates the integral from a Sobol' sequence, whose error
//! `O(N^(-1) )` beats the `O(N^(-1/2))` of pseudo-random sampling.

use super::{Real, ordered};

/// The options of the quadrature rules: the tolerances and how much work a rule may do.
#[derive(Debug, Clone, Copy)]
pub struct QuadOptions<T> {
    /// The absolute tolerance of the error estimate.
    pub epsabs: T,
    /// The relative tolerance of the error estimate.
    pub epsrel: T,
    /// The largest number of subintervals a rule keeps, which bounds the work.
    pub max_subintervals: usize,
}

impl<T: Real> Default for QuadOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> QuadOptions<T> {
    /// The default options: `epsabs` and `epsrel` `1e-12`, and `max_subintervals` 200.
    pub fn new() -> Self {
        Self {
            epsabs: T::from_f64(1e-12),
            epsrel: T::from_f64(1e-12),
            max_subintervals: 200,
        }
    }

    /// Sets [`epsabs`](Self::epsabs).
    pub fn with_epsabs(mut self, epsabs: T) -> Self {
        self.epsabs = epsabs;
        self
    }

    /// Sets [`epsrel`](Self::epsrel).
    pub fn with_epsrel(mut self, epsrel: T) -> Self {
        self.epsrel = epsrel;
        self
    }

    /// Sets [`max_subintervals`](Self::max_subintervals).
    pub fn with_max_subintervals(mut self, max_subintervals: usize) -> Self {
        self.max_subintervals = max_subintervals;
        self
    }

    /// Whether an error estimate `err` of an integral `value` is small enough.
    fn accepted(&self, value: T, err: T) -> bool {
        // An estimate that is not a number is never small enough, however large the tolerance.
        value.is_finite() && err.is_finite() && err <= self.epsabs.max(self.epsrel * value.abs())
    }
}

/// What a quadrature rule returns: the estimate, its error estimate, how many times the integrand
/// was called, and whether the error estimate came within the tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadResult<T> {
    /// The estimate of the integral.
    pub value: T,
    /// The estimate of the absolute error of `value`.
    pub error: T,
    /// The number of evaluations of the integrand.
    pub evaluations: usize,
    /// Whether `error` came within the tolerances.
    pub converged: bool,
}

impl<T: Real> QuadResult<T> {
    /// The result as a pair, for tests that only compare the estimate.
    pub fn as_tuple(&self) -> (T, T) {
        (self.value, self.error)
    }
}

/// The 15-point Kronrod rule on `[-1, 1]` and its 7-point Gauss rule, held as the nodes of the
/// symmetric rule with non-negative abscissa: an entry `(u, wk, wg)` is the node `u` of the pair
/// `±u`, with the weight of a *single* node of each rule, and `wg` is `None` where the two rules
/// do not share the node. The last entry is the centre node `0`. A pair therefore contributes
/// `wk · (f(mid + half·u) + f(mid − half·u))`, and the weights of a rule sum to two.
// The published nodes and weights of the two Gauss-Kronrod pairs, given to more digits than
// an f64 carries, so that the values here are the ones the tables give.
#[allow(clippy::excessive_precision)]
const G7K15: [(f64, f64, Option<f64>); 8] = [
    (9.91455371120812612e-01, 2.29353220105292244e-02, None),
    (
        9.49107912342758486e-01,
        6.30920926299785578e-02,
        Some(1.29484966168869703e-01),
    ),
    (8.64864423359769097e-01, 1.04790010322250188e-01, None),
    (
        7.41531185599394460e-01,
        1.40653259715525919e-01,
        Some(2.79705391489276645e-01),
    ),
    (5.86087235467691148e-01, 1.69004726639267910e-01, None),
    (
        4.05845151377397184e-01,
        1.90350578064785420e-01,
        Some(3.81830050505118923e-01),
    ),
    (2.07784955007898481e-01, 2.04432940075298886e-01, None),
    (0.0, 2.09482141084727819e-01, Some(4.17959183673469403e-01)),
];

#[allow(clippy::excessive_precision)]
const G10K21: [(f64, f64, Option<f64>); 11] = [
    (9.95657163025808090e-01, 1.16946388673718742e-02, None),
    (
        9.73906528517171743e-01,
        3.25581623079647248e-02,
        Some(6.66713443086881380e-02),
    ),
    (9.30157491355708244e-01, 5.47558965743519949e-02, None),
    (
        8.65063366688984536e-01,
        7.50396748109199568e-02,
        Some(1.49451349150580587e-01),
    ),
    (7.80817726586416905e-01, 9.31254545836976005e-02, None),
    (
        6.79409568299024436e-01,
        1.09387158802297643e-01,
        Some(2.19086362515982042e-01),
    ),
    (5.62757134668604664e-01, 1.23491976262065845e-01, None),
    (
        4.33395394129247213e-01,
        1.34709217311473339e-01,
        Some(2.69266719309996350e-01),
    ),
    (2.94392862701460201e-01, 1.42775938577060085e-01, None),
    (
        1.48874338981631216e-01,
        1.47739104901338486e-01,
        Some(2.95524224714752870e-01),
    ),
    (0.0, 1.49445554002916897e-01, None),
];
/// One subinterval of an adaptive rule, with both estimates of its integral.
struct Panel<T> {
    a: T,
    b: T,
    mid: T,
    kronrod: T,
    error: T,
    evaluations: usize,
}

impl<T: Real> Panel<T> {
    /// The size of the panel's error, which is what decides which panel to split.
    fn err(&self) -> T {
        self.error
    }
}

/// The Kronrod and Gauss estimates of `f` on `[a, b]` from `table`, whose entries are
/// `(u, wk, wg)`: the abscissa `u` in `(0, 1]`, its Kronrod weight, and the Gauss weight of that
/// node where the two rules share it. The last entry is the centre, whose abscissa is 0.
fn panel<T: Real>(
    f: &mut impl FnMut(T) -> T,
    a: T,
    b: T,
    table: &[(f64, f64, Option<f64>)],
) -> Panel<T> {
    let half = (b - a) * T::HALF;
    let mid = a + half;
    let c = T::from_f64;
    // A node of a panel so narrow that its outermost nodes round onto an end is drawn this far
    // inside it, so that an integrand with an integrable singularity at an end is never asked
    // for its value at the end itself.
    let inset = a.abs().max(b.abs()) * T::EPSILON;
    let mut k = T::ZERO;
    let mut g = T::ZERO;
    for &(x, wk, wg) in table.iter().take(table.len() - 1) {
        let d = half * c(x);
        let s = f((mid + d).min(b - inset)) + f((mid - d).max(a + inset));
        k = c(wk).mul_add(s * half, k);
        if let Some(wg) = wg {
            g = c(wg).mul_add(s * half, g);
        }
    }
    // The centre node, evaluated once.
    let (_, wk, wg) = table[table.len() - 1];
    let fc = f(mid.max(a + inset).min(b - inset));
    k = c(wk).mul_add(fc * half, k);
    if let Some(wg) = wg {
        g = c(wg).mul_add(fc * half, g);
    }
    Panel {
        a,
        b,
        mid,
        kronrod: k,
        error: (k - g).abs(),
        evaluations: 2 * table.len() - 1,
    }
}

/// The adaptive bisection loop shared by the two Kronrod rules: split the panel with the largest
/// error until every estimate is within the tolerances or `max_subintervals` panels are left.
fn adapt<T: Real>(
    f: &mut impl FnMut(T) -> T,
    a: T,
    b: T,
    table: &[(f64, f64, Option<f64>)],
    opts: &QuadOptions<T>,
) -> QuadResult<T> {
    let mut panels = vec![panel(f, a, b, table)];
    let mut evaluations = panels[0].evaluations;
    loop {
        let value: T = panels.iter().map(|p| p.kronrod).fold(T::ZERO, |s, v| s + v);
        let error: T = panels.iter().map(|p| p.err()).fold(T::ZERO, |s, v| s + v);
        if opts.accepted(value, error) || panels.len() >= opts.max_subintervals {
            return QuadResult {
                value,
                error,
                evaluations,
                converged: opts.accepted(value, error),
            };
        }
        // Split the worst panel.
        let mut worst = 0;
        for i in 1..panels.len() {
            if panels[i].err() > panels[worst].err() {
                worst = i;
            }
        }
        let p = panels.swap_remove(worst);
        let left = panel(f, p.a, p.mid, table);
        let right = panel(f, p.mid, p.b, table);
        evaluations += left.evaluations + right.evaluations;
        panels.push(left);
        panels.push(right);
    }
}

/// The integral of `f` on `[a, b]` by adaptive Gauss–Kronrod quadrature with the G10–K21 rule, to
/// the default tolerances of [`QuadOptions`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::quad;
///
/// let r = quad(|x: f64| x.exp(), 0.0, 1.0).unwrap();
/// assert!((r.value - (core::f64::consts::E - 1.0)).abs() < 1e-14);
/// assert!(r.converged);
/// ```
pub fn quad<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<QuadResult<T>> {
    quad_g10k21(f, a, b, &QuadOptions::new())
}

/// The integral of `f` on `[a, b]` by adaptive G7–K15 quadrature.
pub fn quad_g7k15<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    if !ordered(a, b) {
        return None;
    }
    Some(adapt(&mut f, a, b, &G7K15, opts))
}

/// The integral of `f` on `[a, b]` by adaptive G10–K21 quadrature.
pub fn quad_g10k21<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    if !ordered(a, b) {
        return None;
    }
    Some(adapt(&mut f, a, b, &G10K21, opts))
}

/// The integral of `f` on `[a, b]` by Clenshaw–Curtis quadrature: the interpolant through the
/// Chebyshev–Lobatto nodes `cos(kπ/n)`, integrated exactly, refined by bisecting the interval
/// until the estimate of the error comes within the tolerances. The values of one level are
/// reused by the next, so a smooth integrand is called `n + 1` times in total.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, clenshaw_curtis};
///
/// // The integrand is analytic, so the rule converges geometrically.
/// let r = clenshaw_curtis(|x: f64| 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0, &QuadOptions::new()).unwrap();
/// assert!((r.value - 2.0 * 5.0_f64.atan() / 5.0).abs() < 1e-12);
/// ```
pub fn clenshaw_curtis<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    if !ordered(a, b) {
        return None;
    }
    let mut evaluations = 0;
    let mut panels = vec![cc_make(&mut f, a, b, CC_START, &mut evaluations)];
    loop {
        let value: T = panels.iter().fold(T::ZERO, |s, p| s + p.value);
        let error: T = panels.iter().fold(T::ZERO, |s, p| s.max(p.err));
        let converged = opts.accepted(value, error);
        if converged || panels.len() >= opts.max_subintervals {
            return Some(QuadResult {
                value,
                error,
                evaluations,
                converged,
            });
        }
        // Double the level of every panel that can still grow, and bisect the ones that cannot.
        let mut next: Vec<CcPanel<T>> = Vec::with_capacity(panels.len() * 2);
        for p in panels {
            if p.n < CC_MAX {
                next.push(cc_refine(&mut f, &p, &mut evaluations));
            } else {
                let m = p.l + (p.r - p.l) * T::HALF;
                next.push(cc_make(&mut f, p.l, m, CC_START, &mut evaluations));
                next.push(cc_make(&mut f, m, p.r, CC_START, &mut evaluations));
            }
        }
        panels = next;
    }
}

/// The intervals a Clenshaw-Curtis panel starts with, and the most it reaches before the panel is
/// bisected instead of doubled.
const CC_START: usize = 4;
const CC_MAX: usize = 512;

/// One panel of [`clenshaw_curtis`]: the interval, its level, the values at its
/// Chebyshev-Lobatto nodes, the estimate they give, and how far that was from the level below.
struct CcPanel<T> {
    l: T,
    r: T,
    n: usize,
    vals: Vec<T>,
    value: T,
    err: T,
}

/// The values of `f` at the `n + 1` Chebyshev-Lobatto nodes of `[l, r]`, in the order of
/// `cos(kπ/n)` from the right end inwards.
fn cc_values<T: Real>(
    f: &mut impl FnMut(T) -> T,
    l: T,
    r: T,
    n: usize,
    evaluations: &mut usize,
) -> Vec<T> {
    let half = (r - l) * T::HALF;
    let mid = l + half;
    (0..=n)
        .map(|k| {
            let u = core::f64::consts::PI * k as f64 / n as f64;
            *evaluations += 1;
            f(mid + half * T::from_f64(u.cos()))
        })
        .collect()
}

/// The Clenshaw-Curtis estimate of the integral on `[l, r]` from the values at its `n + 1` nodes.
fn cc_estimate<T: Real>(vals: &[T], half: T, n: usize) -> T {
    let mut s = T::ZERO;
    for (k, &v) in vals.iter().enumerate() {
        s += cc_weight::<T>(k, n) * v;
    }
    s * half * T::TWO
}

/// A new panel at `n` intervals. The estimate of the level below is read off the even nodes,
/// which are that level's own grid, so the first error estimate costs no evaluations.
fn cc_make<T: Real>(
    f: &mut impl FnMut(T) -> T,
    l: T,
    r: T,
    n: usize,
    evaluations: &mut usize,
) -> CcPanel<T> {
    let vals = cc_values(f, l, r, n, evaluations);
    let half = (r - l) * T::HALF;
    let fine = cc_estimate(&vals, half, n);
    let coarse: Vec<T> = (0..=n / 2).map(|j| vals[2 * j]).collect();
    CcPanel {
        l,
        r,
        n,
        vals,
        value: fine,
        err: (fine - cc_estimate(&coarse, half, n / 2)).abs(),
    }
}

/// `p` at twice its level. Half of the new nodes are the old ones, so only the `n` odd ones are
/// evaluated.
fn cc_refine<T: Real>(
    f: &mut impl FnMut(T) -> T,
    p: &CcPanel<T>,
    evaluations: &mut usize,
) -> CcPanel<T> {
    let n = p.n;
    let half = (p.r - p.l) * T::HALF;
    let mid = p.l + half;
    let mut vals = vec![T::ZERO; 2 * n + 1];
    for j in 0..=n {
        vals[2 * j] = p.vals[j];
    }
    for k in (1..2 * n).step_by(2) {
        let u = core::f64::consts::PI * k as f64 / (2 * n) as f64;
        vals[k] = f(mid + half * T::from_f64(u.cos()));
        *evaluations += 1;
    }
    let fine = cc_estimate(&vals, half, 2 * n);
    CcPanel {
        l: p.l,
        r: p.r,
        n: 2 * n,
        vals,
        value: fine,
        err: (fine - p.value).abs(),
    }
}

/// The Clenshaw-Curtis weight of the `k`-th of `n + 1` Chebyshev-Lobatto nodes on `[-1, 1]`,
/// halved at the two ends, from the cosine series so that no transform is needed. The rule on an
/// interval of half-width `half` is `2 · half · Σₖ wₖ fₖ`.
fn cc_weight<T: Real>(k: usize, n: usize) -> T {
    let theta = core::f64::consts::PI * k as f64 / n as f64;
    let mut s = 0.0f64;
    for j in 1..=n / 2 {
        let b = if 2 * j == n { 1.0 } else { 2.0 };
        s += b * (2.0 * j as f64 * theta).cos() / ((4.0 * j as f64 * j as f64) - 1.0);
    }
    let end = if k == 0 || k == n { 0.5 } else { 1.0 };
    T::from_f64(end * (1.0 - s) / n as f64)
}

/// The integral of `f` on `[a, b]` by the tanh-sinh (double exponential) rule: the substitution
/// `x = mid + half·tanh((π/2)·sinh t)` pushes the ends of the interval out to infinity, where the
/// trapezoidal rule in `t` converges as quickly as the integrand is smooth at the ends — faster
/// than any algebraic rule where the integrand, or one of its derivatives, becomes infinite at an
/// end. The step of that rule starts at 1 and halves; the estimate is accepted when two levels
/// come within the tolerances.
///
/// Nodes are asked for the value of `f` strictly inside the interval, so an integrable
/// singularity at an end is welcome rather than fatal. One that blows up like `1/√(b − x)` is
/// good to about 1e-8 and no more, because the nodes nearest an end round onto it; a kink in the
/// middle of the interval belongs to [`quad`], which splits where the error is.
///
/// # Examples
///
/// ```
/// use a::math::numerics::tanh_sinh;
/// use a::math::numerics::QuadOptions;
///
/// // The logarithmic singularity at 0, which the adaptive rules reach only slowly:
/// // ∫₀¹ ln x dx = −1.
/// let r = tanh_sinh(|x: f64| x.ln(), 0.0, 1.0, &QuadOptions::new()).unwrap();
/// assert!((r.value + 1.0).abs() < 1e-14);
/// assert!(r.converged);
///
/// // An integrable algebraic singularity at both ends, good to about 1e-8:
/// // ∫₋₁¹ dx/√(1 − x²) = π.
/// let r = tanh_sinh(|x: f64| (1.0 - x * x).sqrt().recip(), -1.0, 1.0, &QuadOptions::new())
///     .unwrap();
/// assert!((r.value - core::f64::consts::PI).abs() < 1e-7);
/// ```
pub fn tanh_sinh<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    if !ordered(a, b) {
        return None;
    }
    let half = (b - a) * T::HALF;
    let mid = a + half;
    let c = T::from_f64;
    let pi = T::PI;
    // The node nearest an end is drawn this far inside it, so that an integrand with an
    // integrable singularity at an end is never asked for its value at the end itself.
    let inset = a.abs().max(b.abs()) * T::EPSILON;
    let mut evaluations = 0;
    let mut prev = T::ZERO;
    let mut value = T::ZERO;
    let mut error = T::ZERO;
    for level in 1..=10 {
        let h = c(0.5f64.powi(level - 1));
        // The weight of a node of the trapezoidal rule in `t`, which is also the factor that
        // turns a term of the sum into its contribution to the integral.
        let step = h * half;
        // A node is dropped only when its contribution is far too small to matter even in the
        // company of all the nodes like it; the weights fall away doubly exponentially, so this
        // stops a few nodes past the point where the sum has really converged.
        let negligible = opts.epsabs * T::EPSILON;
        // The node `t = 0` maps to the midpoint and carries the weight π/2; every other node
        // `t = h·k` comes in the symmetric pair `±t`, which maps onto the two halves of the
        // interval and shares one weight.
        let mut s = T::ZERO;
        let centre = f(mid);
        if centre.is_finite() {
            s = (pi * T::HALF).mul_add(centre, s);
        }
        evaluations += 1;
        let mut k = 1usize;
        loop {
            let t = h * c(k as f64);
            let (sh, ch) = (t.sinh(), t.cosh());
            let e = pi * sh * T::HALF;
            let (se, ce) = (e.sinh(), e.cosh());
            if !ce.is_finite() || ce > c(1e15) {
                break;
            }
            let w = (pi * ch * T::HALF) / (ce * ce);
            let d = half * (se / ce);
            let nodes = [(mid + d).min(b - inset), (mid - d).max(a + inset)];
            // The sum ends when the pair of terms is negligible, which is also where it ends when
            // a node is pinned on a singularity and its terms cannot be had at all.
            let mut worth = false;
            for &node in &nodes {
                let term = w * f(node);
                evaluations += 1;
                if term.is_finite() {
                    s += term;
                    worth |= (step * term).abs() > negligible;
                }
            }
            if !worth {
                break;
            }
            k += 1;
        }
        value = s * step;
        error = (value - prev).abs();
        if opts.accepted(value, error) {
            return Some(QuadResult {
                value,
                error,
                evaluations,
                converged: true,
            });
        }
        prev = value;
    }
    Some(QuadResult {
        value,
        error,
        evaluations,
        converged: opts.accepted(value, error),
    })
}

/// The integral of `f` on `[a, ∞)`, by the map `x = a + t/(1 − t)` of the half-line onto `[0, 1]`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::quad_half_infinite;
/// use a::math::numerics::QuadOptions;
///
/// // ∫₀^∞ e^(−x) dx = 1.
/// let r = quad_half_infinite(|x: f64| (-x).exp(), 0.0, &QuadOptions::new()).unwrap();
/// assert!((r.value - 1.0).abs() < 1e-12);
/// ```
pub fn quad_half_infinite<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    let g = |t: T| {
        let d = T::ONE - t;
        if d <= T::ZERO {
            return T::ZERO;
        }
        f(a + t / d) / (d * d)
    };
    quad_g10k21(&mut { g }, T::ZERO, T::ONE, opts)
}

/// The integral of `f` on `(-∞, ∞)`, by the map `x = t/(1 − t²)` of the half-line `x ≥ 0` onto
/// `[0, 1]`: the two halves of the line are folded together, so that both tails are integrated.
///
/// # Examples
///
/// ```
/// use a::math::numerics::quad_infinite;
/// use a::math::numerics::QuadOptions;
///
/// // ∫₋∞^∞ e^(−x²) dx = √π.
/// let r = quad_infinite(|x: f64| (-x * x).exp(), &QuadOptions::new()).unwrap();
/// assert!((r.value - core::f64::consts::PI.sqrt()).abs() < 1e-12);
/// ```
pub fn quad_infinite<T: Real>(
    mut f: impl FnMut(T) -> T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    let g = |t: T| {
        let d = T::ONE - t * t;
        if d <= T::ZERO {
            return T::ZERO;
        }
        let x = t / d;
        let jac = (T::ONE + t * t) / (d * d);
        (f(x) + f(-x)) * jac
    };
    quad_g10k21(&mut { g }, T::ZERO, T::ONE, opts)
}

/// The integral of `f` over the box `[a₀, b₀] × … × [aₙ₋₁, bₙ₋₁]` by adaptive tensor-product
/// Gauss–Kronrod cubature: the `d`-dimensional G7–K15 product rule estimates the integral of each
/// box against its G7 product rule, and the box whose error is largest is halved along its longest
/// edge.
///
/// The work grows as `15^d`, so this is for two or three dimensions; [`monte_carlo`] scales better.
///
/// # Examples
///
/// ```
/// use a::math::numerics::cubature;
/// use a::math::numerics::QuadOptions;
///
/// // ∫₀¹∫₀¹ xy dx dy = 1/4.
/// let r = cubature(|x: &[f64]| x[0] * x[1], &[0.0, 0.0], &[1.0, 1.0], &QuadOptions::new()).unwrap();
/// assert!((r.value - 0.25).abs() < 1e-12);
/// ```
pub fn cubature<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    let d = a.len();
    if d == 0 || b.len() != d || a.iter().zip(b).any(|(l, r)| !ordered(*l, *r)) {
        return None;
    }
    let mut boxes: Vec<(Vec<T>, Vec<T>)> = vec![(a.to_vec(), b.to_vec())];
    let mut evaluations = 0;
    loop {
        let mut total = T::ZERO;
        let mut err = T::ZERO;
        let mut errs: Vec<T> = Vec::with_capacity(boxes.len());
        for (lo, hi) in &boxes {
            let (v, e, n) = tensor_rule(&mut f, lo, hi);
            evaluations += n;
            total += v;
            err += e;
            errs.push(e);
        }
        if opts.accepted(total, err) || boxes.len() >= opts.max_subintervals {
            return Some(QuadResult {
                value: total,
                error: err,
                evaluations,
                converged: opts.accepted(total, err),
            });
        }
        let mut worst = 0;
        for i in 1..errs.len() {
            if errs[i] > errs[worst] {
                worst = i;
            }
        }
        let (lo, hi) = boxes.swap_remove(worst);
        let mut dim = 0;
        let mut span = T::ZERO;
        for j in 0..d {
            let s = hi[j] - lo[j];
            if s > span {
                span = s;
                dim = j;
            }
        }
        let m = lo[dim] + span * T::HALF;
        let l1 = lo.clone();
        let mut h1 = hi.clone();
        h1[dim] = m;
        let mut l2 = lo.clone();
        let h2 = hi.clone();
        l2[dim] = m;
        boxes.push((l1, h1));
        boxes.push((l2, h2));
    }
}

/// The table of a symmetric rule on `[-1, 1]` in full: every node, with the Kronrod weight of a
/// single node and the Gauss weight of that node where the two rules share it.
fn symmetric(table: &[(f64, f64, Option<f64>)]) -> Vec<(f64, f64, Option<f64>)> {
    let mut out = Vec::with_capacity(2 * table.len() - 1);
    for &(u, wk, wg) in table.iter().take(table.len() - 1) {
        out.push((-u, wk, wg));
        out.push((u, wk, wg));
    }
    out.push(table[table.len() - 1]);
    out
}

/// The `d`-dimensional product G7–K15 rule of the box `[lo, hi]`: the estimate, its error
/// estimate, and the evaluations used. The two rules of the pair share their nodes, so one sweep
/// over the box gives both estimates and their difference is the error.
fn tensor_rule<T: Real>(f: &mut impl FnMut(&[T]) -> T, lo: &[T], hi: &[T]) -> (T, T, usize) {
    let d = lo.len();
    let rule = symmetric(&G7K15);
    let n = rule.len().pow(d as u32);
    let mut points = Vec::with_capacity(n);
    let mut kronrod = Vec::with_capacity(n);
    let mut gauss = Vec::with_capacity(n);
    let mut idx = vec![0usize; d];
    loop {
        let mut x = vec![T::ZERO; d];
        let mut wk = T::ONE;
        let mut wg = T::ONE;
        for j in 0..d {
            let (u, k, g) = rule[idx[j]];
            let half = (hi[j] - lo[j]) * T::HALF;
            x[j] = lo[j] + half + half * T::from_f64(u);
            wk *= T::from_f64(k);
            wg = match g {
                Some(g) => wg * T::from_f64(g),
                None => T::ZERO,
            };
        }
        points.push(x);
        kronrod.push(wk);
        gauss.push(wg);
        let mut j = 0;
        loop {
            idx[j] += 1;
            if idx[j] < rule.len() {
                break;
            }
            idx[j] = 0;
            j += 1;
            if j == d {
                let mut jac = T::ONE;
                for axis in 0..d {
                    jac *= (hi[axis] - lo[axis]) * T::HALF;
                }
                let mut k = T::ZERO;
                let mut g = T::ZERO;
                for i in 0..points.len() {
                    let v = f(&points[i]);
                    k = kronrod[i].mul_add(v, k);
                    g = gauss[i].mul_add(v, g);
                }
                return (k * jac, (jac * (k - g)).abs(), points.len());
            }
        }
    }
}

/// The integral of `f` over the box `[a₀, b₀] × … × [aₙ₋₁, bₙ₋₁]` by quasi-Monte Carlo on the
/// Sobol' sequence: `V · Σᵢ f(xᵢ)/N` with `V` the volume of the box. The error falls as
/// `O(N⁻¹)` for smooth integrands, against `O(N^(-1/2))` of pseudo-random sampling.
///
/// # Examples
///
/// ```
/// use a::math::numerics::monte_carlo;
///
/// // ∫₀¹∫₀¹ (x + y) dx dy = 1.
/// let r = monte_carlo(|x: &[f64]| x[0] + x[1], &[0.0, 0.0], &[1.0, 1.0], 1 << 14);
/// assert!((r.value - 1.0).abs() < 1e-3);
/// ```
pub fn monte_carlo<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    samples: usize,
) -> QuadResult<T> {
    let d = a.len();
    let mut volume = T::ONE;
    for j in 0..d {
        volume *= b[j] - a[j];
    }
    let sobol = Sobol::new(d);
    let mut sum = T::ZERO;
    let mut sum2 = T::ZERO;
    let mut x = vec![T::ZERO; d];
    for i in 0..samples {
        let u = sobol.next(i);
        for j in 0..d {
            x[j] = a[j] + (b[j] - a[j]) * T::from_f64(u[j]);
        }
        let v = f(&x);
        sum += v;
        sum2 += v * v;
    }
    let n = T::from_f64(samples.max(1) as f64);
    let mean = sum / n;
    let variance = (sum2 / n - mean * mean).max(T::ZERO);
    QuadResult {
        value: mean * volume,
        error: (variance / n).sqrt() * volume.abs(),
        evaluations: samples,
        converged: true,
    }
}

/// The Sobol' sequence of `dim` dimensions, whose direction numbers come from the primitive
/// polynomials over `GF(2)`; the first `dim` of them are found at construction.
pub struct Sobol {
    dim: usize,
    /// The direction numbers `vᵢ = mᵢ / 2^i`, `i = 1..32`, of each dimension.
    v: Vec<Vec<u32>>,
}

impl Sobol {
    /// The sequence of `dim` dimensions.
    pub fn new(dim: usize) -> Self {
        let dim = dim.max(1);
        let mut a = Vec::with_capacity(dim);
        let mut v = Vec::with_capacity(dim);
        let mut found = 0;
        let mut candidate = 3u32;
        while found < dim {
            if let Some(degree) = primitive(candidate) {
                if found == 0 {
                    a.push(0);
                } else {
                    a.push(candidate ^ (1 | (1 << degree)));
                }
                let mut vv = vec![0u32; 32];
                for i in 1..=degree {
                    vv[i - 1] = 1u32 << (32 - i);
                }
                for i in (degree + 1)..=32 {
                    let mut x = vv[i - degree - 1];
                    let mut ai = a[found];
                    let mut j = 1;
                    while ai != 0 {
                        if ai & 1 != 0 {
                            x ^= vv[i - j - 1];
                        }
                        ai >>= 1;
                        j += 1;
                    }
                    vv[i - 1] = x ^ (x >> degree);
                }
                v.push(vv);
                found += 1;
            }
            candidate += 2;
        }
        Self { dim, v }
    }

    /// The `i`-th point of the sequence, in `[0, 1)^dim`.
    pub fn next(&self, i: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; self.dim];
        for (d, direction) in self.v.iter().enumerate() {
            let mut x = 0u32;
            let mut n = i as u32;
            let mut k = 0;
            while n != 0 {
                if n & 1 != 0 {
                    x ^= direction[k];
                }
                n >>= 1;
                k += 1;
            }
            out[d] = (x as f64) / (1u64 << 32) as f64;
        }
        out
    }
}

/// The degree of the primitive polynomial over `GF(2)` whose coefficients are the bits of `p`
/// (the leading term included), or `None` when `p` is not primitive.
fn primitive(p: u32) -> Option<usize> {
    let degree = (32 - p.leading_zeros()) as usize - 1;
    if degree == 0 {
        return None;
    }
    // Irreducible and of full period: x^(2^degree) ≡ x modulo p, and no smaller exponent works.
    let mut x = 2u32;
    for _ in 1..degree {
        x = mul_mod(x, x, p, degree);
    }
    if mul_mod(x, x, p, degree) != 2 {
        return None;
    }
    for k in 1..degree {
        let mut y = 2u32;
        for _ in 0..k {
            y = mul_mod(y, y, p, degree);
        }
        if y == 2 {
            return None;
        }
    }
    Some(degree)
}

/// The product `a·b mod p` of two polynomials over `GF(2)`, `p` given by its bits.
fn mul_mod(mut a: u32, mut b: u32, p: u32, degree: usize) -> u32 {
    let mut r = 0u32;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & (1 << degree) != 0 {
            a ^= p;
        }
    }
    r & ((1 << degree) - 1)
}
