[A](../../../README.md) › [Math](../../../README.md#math)

# Numerical methods

`a::math::numerics` solves the standard problems of numerical analysis on `f32` and `f64`: the zeros of a function, its definite integrals, initial value problems for ordinary differential equations, minima and least-squares fits, interpolation and differentiation. Everything is generic over the sealed trait `Real`, implemented for `f32` and `f64` only, and the linear algebra a method needs is done in [`a::math::matrix`](matrix.md) — the LU of the implicit ODE corrector and of the damped fit, the QR of the least-squares step, the Hermitian eigensolver of the Gaussian rules.

- The zeros of a function of one real variable, by bisection, Brent, the secant method, Newton and Halley; of a system of equations, by Newton with a Jacobian and by Broyden.
- The integral of a function of one variable by adaptive Gauss–Kronrod (G7–K15 and G10–K21), Clenshaw–Curtis and tanh-sinh quadrature, over infinite intervals too; of several by cubature and by Sobol' Monte Carlo; and the nodes and weights of the Gauss–Legendre, Hermite, Laguerre and Jacobi rules.
- The initial value problem `y′ = f(t, y)` by the classical explicit Runge–Kutta method of the fourth order, by Dormand and Prince's 5(4) pair and by the stiff Radau IIA method of three stages, all with step-size control.
- The minimum of a function, of one variable by the golden section and by Brent's method, of several by Nelder and Mead's simplex, the conjugate gradient and BFGS, and of a sum of squares by Gauss–Newton and by Levenberg and Marquardt.
- Interpolation and approximation — the cubic, clamped, not-a-knot, Akima and PCHIP splines, Chebyshev approximation and barycentric interpolation — and numerical differentiation by Richardson extrapolation.

Two rules hold everywhere. A method never returns an answer it has not verified: one that cannot reach its tolerance returns `None`, and the optimisers, which always have a best point to report, return it with `converged` false. And every method comes in a pair, `name` with the default options and `name_with` with an options struct built by `with_*` calls:

```rust
use a::math::numerics::{RootOptions, brent, brent_with};

assert!(brent(1.0, 2.0, |x: f64| x * x - 2.0).is_some());
let tight = RootOptions::new().with_max_iter(200);
assert!(brent_with(1.0, 2.0, |x: f64| x * x - 2.0, &tight).is_some());
```

## Examples

```rust
use a::math::numerics::{RadauIia, RkOptions, Spline, bfgs, brent, derivative, quad, solve};
use a::math::matrix::Matrix;

// √2 as the root of x² − 2.
let r = brent(1.0, 2.0, |x: f64| x * x - 2.0).unwrap();
assert!((r - core::f64::consts::SQRT_2).abs() < 1e-15);

// π as an integral, with the error the rule estimated.
let i = quad(|x: f64| 4.0 / (1.0 + x * x), 0.0, 1.0).unwrap();
assert!(i.converged && (i.value - core::f64::consts::PI).abs() < 1e-15);

// The stiff decay y′ = −1000y over fifty time constants: 19 steps.
let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -1000.0 * y[(0, 0)];
let y = solve(f, &Matrix::from_rows(&[[1.0]]), 0.0, 0.05, RadauIia::new(),
              RkOptions::new(1e-6, 1e-8)).unwrap();
assert!((y[(0, 0)] - (-50.0f64).exp()).abs() < 1e-4);

// The slope of the sine at 1, to fourteen digits.
let d = derivative(|x: f64| x.sin(), 1.0).unwrap();
assert!((d - 1.0f64.cos()).abs() < 1e-14);

// Rosenbrock's valley, from (−1.2, 1), on the gradient alone.
let g = |x: &Matrix<f64>| {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
};
let grad = |x: &Matrix<f64>| {
    let (a, b) = (x[(0, 0)], x[(1, 0)]);
    Matrix::from_rows(&[[-2.0 * (1.0 - a) - 400.0 * a * (b - a * a)], [200.0 * (b - a * a)]])
};
let m = bfgs(g, grad, &Matrix::from_rows(&[[-1.2], [1.0]])).unwrap();
assert!(m.converged && (m.x[(0, 0)] - 1.0).abs() < 1e-5);

// And a cubic spline through twenty-one points of a curve.
let x: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
let y: Vec<f64> = x.iter().map(|&t| t.exp()).collect();
let s = Spline::cubic(&x, &y).unwrap();
assert!((s.evaluate(0.5).unwrap() - 0.5f64.exp()).abs() < 1e-5);
```

## Roots

| Function | Needs | Order of convergence | Guaranteed to find the root |
| --- | --- | --- | --- |
| `bisect` | a bracket `[a, b]` with `f(a)·f(b) < 0` | 1 | yes |
| `brent` | a bracket | 2 to 3 in practice | yes |
| `secant` | two starting points | 1.618 | no |
| `newton` | `f` and `f′` | 2 | no |
| `halley` | `f`, `f′` and `f″` | 3 | no |

A bracketing method is told where the root is and cannot miss it; an open method is told only where to start and may wander off, in which case it returns `None`. `RootOptions` holds `xtol`, `rtol`, `ftol` and `max_iter` (100 by default); a root is accepted when `|b − a| ≤ xtol + rtol·|x|` and `|f(x)| ≤ ftol`, with `rtol` defaulting to four epsilons — no tolerance below that can be satisfied by a float.

For a system of `n` equations in `n` unknowns, `newton_system` takes the residual and its Jacobian and solves the correction equation `J·δ = −f` by LU, and `broyden` takes the residual alone and improves a finite-difference Jacobian by a rank-one update at every step. Both are held in `numerics::system`, and both take a `SystemOptions`.

## Quadrature

| Function | Rule | Best for |
| --- | --- | --- |
| `quad` | adaptive G10–K21 | the default: a smooth integrand on a finite interval |
| `quad_g7k15` | adaptive G7–K15 | cheap integrands, where a rule of fewer nodes wins |
| `clenshaw_curtis` | Chebyshev–Lobatto, refined by bisection | an integrand evaluated at a cost, since the levels share nodes |
| `tanh_sinh` | the double-exponential map | an integrand, or one of its derivatives, blowing up at an end |
| `quad_half_infinite`, `quad_infinite` | `quad` on a mapped interval | `[a, ∞)`, `(−∞, ∞)` |
| `cubature` | tensor-product Kronrod, bisected where the error is largest | up to a handful of dimensions |
| `monte_carlo` | Sobol' sequence | many dimensions, where a tensor product is out of the question |

The Gauss and Kronrod rules share their nodes, so the difference of the two estimates is a cheap error estimate for the interval; `QuadOptions` holds `epsabs`, `epsrel` (both `1e-12`) and `max_subintervals` (200), and the result reports the estimate, the error, the number of evaluations and whether the tolerances were met.

`legendre`, `hermite`, `laguerre` and `jacobi` return the nodes and weights of the `n`-point Gaussian rule of each weight function, by Golub and Welsch's method on the Jacobi matrix; the nodes of the largest orders are then refined by Newton's method on the polynomial itself. A rule of `n` points is exact for the polynomials of degree `2n − 1` or less, and `NodesWeights::integrate` applies it to a function.

## Differential equations

`integrate` and `solve` take the right-hand side as `f(t, y, out)`, which writes into a matrix the solver owns — a caller allocates nothing itself — and the state as a column `Matrix`, so a system of any size is written the same way as a scalar equation.

| Method | Order | Right-hand side evaluations a step | Stability |
| --- | --- | --- | --- |
| `Rk4` | 4 | 12 | explicit |
| `DormandPrince5` | 5 | 7 | explicit |
| `RadauIia` | 5 | 9 | L-stable, implicit |

Both explicit methods and the implicit one judge a step the same way: by step doubling, the two half-steps against the whole one, Richardson's divisor behind it. `RadauIia` solves its collocation system by Newton's method on the LU decomposition, by default on a Jacobian differenced from the right-hand side — `integrate_with` takes an analytic one, which is both cheaper and more accurate. A step whose stages are not finite, or whose corrector does not converge, is rejected and the step shrunk; after enough consecutive failures the integration returns `None` rather than the truncated answer it reached. `RkOptions::new(rtol, atol)` carries the two tolerances and the limits `max_steps` (100,000), `newton_max_iter` and `h0`, `hmax`, `jac_step`.

## Optimisation

| Function | Needs | Rate | Misses the minimum |
| --- | --- | --- | --- |
| `golden_section` | a bracket, unimodality | linear, 0.618 of the interval a step | no |
| `minimize` | a bracket, unimodality | superlinear in practice | no |
| `nelder_mead` | `f`, a starting point and a first step | linear | yes |
| `conjugate_gradient` | `f` and ∇`f` | superlinear | yes |
| `bfgs` | `f` and ∇`f` | superlinear | yes |
| `gauss_newton` | the residual and its Jacobian | quadratic | yes |
| `levenberg_marquardt` | the residual and its Jacobian | between 1 and 2 | yes |

`minimize` is Brent's parabolic interpolation inside a golden-section bracket; `conjugate_gradient` is Polak and Ribière's with restarts, `bfgs` the inverse Broyden–Fletcher–Goldfarb–Shanno update, and both drive their step by a line search that satisfies the strong Wolfe conditions. The two fits minimise `½‖r(x)‖²`, the least-squares step by QR and the damped one by LU; `levenberg_marquardt` finishes where `gauss_newton` stalls on a near-singular Jacobian. All of them share `OptOptions`: `xtol`, `rtol`, `ftol`, `gtol` and `max_iter` (1000). `gtol` defaults to `√ε` because at a minimum the function is flat and the gradient cannot be trusted further than the square root of the precision the values are computed in.

## Interpolation and differentiation

The five piecewise cubics share the Hermite form — on every interval the cubic that takes `y[i]` with slope `m[i]` at one knot and `y[i+1]` with slope `m[i+1]` at the next — and differ only in how the slopes are chosen:

| Constructor | Slopes from | Smoothness | Overshoots |
| --- | --- | --- | --- |
| `Spline::cubic` | a tridiagonal solve | `C²` | yes |
| `Spline::cubic_clamped` | the same, the end slopes given | `C²` | yes |
| `Spline::cubic_not_a_knot` | the same, the third derivative continued | `C²` but at two knots | yes |
| `Spline::akima` | Akima's five-point weights | `C¹` | little |
| `Spline::pchip` | the Fritsch and Carlson limiter | `C¹` | never between monotone knots |

A `Spline` evaluates with `evaluate` and `derivative`, and gives `None` outside its knots. Where one polynomial over the whole interval is better, `chebyshev` fits the polynomial that agrees with the function at the Chebyshev nodes and evaluates it by Clenshaw's recurrence in time linear in the degree; `Chebyshev` has `derivative` and `integral` too. `barycentric_weights` and `barycentric_interpolate` evaluate the interpolant from values that are already data, at cost linear in the number of nodes and without the Runge phenomenon of equally spaced ones.

`derivative` and `second_derivative` combine symmetric difference quotients with `h`, `h/2`, `h/4`, … by Neville's algorithm, which cancels the even powers of `h` in turn and reaches about fourteen digits for `f64` from fewer evaluations than a one-sided quotient of the same accuracy needs. `DiffOptions` holds the first step (chosen from the point by default), the tolerance two columns must meet and the number of columns (8).

## Measurements

The numbers below are what the methods were measured to do, not bounds, and they come from the runs in `lib/tests/math/numerics/` and the conditions stated with them. Accuracy is against the closed form where there is one, and against the invariants and the published values of the problem where there is not. No comparison against an external numerical library has been made for this area.

Orders of convergence, with the step fixed and halved from 0.4 to 0.025 — the tests insist only on the 16-fold refinement of a method of the stated order:

| Problem | Method | Error at `h = 0.4` | Refinement a halving |
| --- | --- | --- | --- |
| `y′ = −y²` on `[0, 1]` | `Rk4` | 1.98·10⁻⁶ | 83, 74, 69, 67 |
| the same | `DormandPrince5` | 1.28·10⁻⁴ | 120, 82, 60, 47 |
| the manufactured stiff solution `λ(y − sin 20t) + 20 cos 20t`, `λ = −50` | `RadauIia` | 1.62·10⁻⁴ | 30, 68, 78 |

`Rk4` refines faster than its fourth order because the answer it returns is the Richardson extrapolation of the two half-steps.

The forced stiff problem `y′ = 1000(sin t − y)` from `t = 0` to `t = 5` at `rtol = 1e-8`, `atol = 1e-10`, against the forced response with the transient gone: 38 steps and 342 evaluations for `RadauIia`, 1.3·10⁻⁸ off, against 3,759 steps and 26,334 evaluations for `DormandPrince5`. Robertson's chemical kinetics to `t = 40`, whose three concentrations must always sum to one:

| tolerance | steps | evaluations | the invariant |
| --- | --- | --- | --- |
| `rtol = 1e-6`, `atol = 1e-11` | 24 | 225 | 1 within 3·10⁻¹⁶ |
| `rtol = 1e-8`, `atol = 1e-13` | 41 | 387 | 1 within 5·10⁻¹⁶ |
| `rtol = 1e-10`, `atol = 1e-15` | 87 | 801 | 1 within 5·10⁻¹⁶ |

The concentrations land on the published `0.715827069`, `9.18553477·10⁻⁶` and `0.284163746` at every one of these tolerances. `DormandPrince5` never gets there: at `rtol = 1e-4` it is confined to a step of `10⁻⁷` by the `3·10⁷` rate, exhausts its 30,000 steps and returns `None`. On a problem of no stiffness at all, `y′ = −y` to `rtol = 1e-12`, the work is 16 steps and 144 evaluations for `RadauIia`, 62 and 744 for `Rk4` and 69 and 483 for `DormandPrince5`.

For `∫₀¹ 4/(1 + x²) dx = π`, `quad` needs one panel: 21 evaluations, the answer exact to the last bit. `quad_g7k15` takes 45, `clenshaw_curtis` 33 (4.4·10⁻¹⁶ off) and `tanh_sinh` 235. The double exponential rule is what to ask for where the integrand blows up at an end: for `∫₀¹ ln x dx` it is 2.2·10⁻¹⁶ off in 114 evaluations, where G10–K21 needs 1,113 to come within 1.3·10⁻¹¹, and for the branch point of `∫₀¹ x^(−1/2) dx` it is 1.5·10⁻⁸ off in 53 evaluations, where G10–K21 needs some 1,800 for the same. Over an infinite interval `∫₀^∞ e⁻ˣ dx` and `∫₋∞^∞ e^(−x²) dx = √π` each take 189 evaluations and come out exact.

The Gaussian rules integrate every polynomial of degree `2n − 1` or less to the precision of the arithmetic: over all the even degrees below `2n`, the worst error is 1.1·10⁻¹⁶ at `n = 4` and 1.3·10⁻¹⁵ at `n = 128`. `derivative` gives the slope of the sine at 1 within 4.4·10⁻¹⁶ and `second_derivative` its curvature within 1.4·10⁻¹³; `brent` returns `√2` to the last bit where `newton` is two ulps off.

On Runge's function over 21 equally spaced knots in `[0, 1]`, the largest error of the cubic spline is 6.8·10⁻³, of Akima's 2.5·10⁻³ and of PCHIP 5.9·10⁻³ — none of them the 10³ a single polynomial over the interval would give, and PCHIP's is the only one that cannot overshoot at all. The Chebyshev approximant of degree 12 to `eˣ` on `[0, 1]` is 4.4·10⁻¹⁶ off in the interior.

On Rosenbrock's valley from (−1.2, 1), `nelder_mead` takes 159 iterations and reaches 5.7·10⁻¹⁷, `conjugate_gradient` 39 and 5.2·10⁻¹², `bfgs` 38 and 4.2·10⁻²¹. `levenberg_marquardt` fits `a·e^(−kt)` to five exact points of the halving curve in 5 iterations, to `a = 1.0000000000` and `k = 0.6931471806` and a residual sum of squares of 2.2·10⁻²³.
