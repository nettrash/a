[A](../../../README.md) › [Math](../../../README.md#math)

# Dual numbers

`a::math::dual` differentiates functions exactly, by forward-mode automatic differentiation. A dual number `Dual<T>` is `re + eps·ε` with `ε² = 0`, so a function evaluated at `x + ε` gives `f(x) + f′(x)·ε`; a hyper-dual number `HyperDual<T>` is `re + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂` with `ε₁² = ε₂² = 0`, so `f(x + ε₁ + ε₂)` carries `f″(x)` as well. Each derivative is exact up to the rounding of the arithmetic — there is no step to choose, as there is for the differences of [numerical differentiation](numerics.md#interpolation-and-differentiation), and nothing cancels — but for one rule at `0·∞`, which keeps gradients usable where a variable sits at a point of infinite derivative and gives a wrong 0 where the true limit is not 0 (see *The derivatives at the edges* and *Limits*). The parts are public fields; nothing here is constant-time.

Both types are generic over the trait `DualNum`, which `f32`, `f64`, and `Dual<T>` and `HyperDual<T>` of every `T: DualNum` implement, so they nest: `Dual<Dual<f64>>` and `Dual<HyperDual<f64>>` carry second and third derivatives. A function is written once, generic over `D: DualNum`, and gives its value for `D = f64` and its derivatives for the dual types. The free functions run the passes and return what the solvers of [`a::math::numerics`](numerics.md) take: a gradient as an `n × 1` column, a Jacobian as an `m × n` matrix, a Hessian as a symmetric `n × n` one.

| Item | What it gives | Passes |
| --- | --- | --- |
| `Dual<T>` | `re + eps·ε`: a value and one derivative | |
| `HyperDual<T>` | `re + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂`: a value, two first derivatives and a second | |
| `DualNum` | the trait of `f32`, `f64`, `Dual<T>` and `HyperDual<T>`: arithmetic, comparisons, constants (`ZERO`, `ONE`, `constant`, `from_f64`), the value `re()`, and the elementary functions | |
| `derivative`, `value_and_derivative` | `f′(x)` | one `Dual` pass |
| `second_derivative`, `value_and_derivatives` | `f″(x)` (and `f(x)`, `f′(x)`) | one `HyperDual` pass |
| `gradient`, `value_and_gradient` | `∇f(x)`, `n × 1` | `n` `Dual` passes |
| `jacobian`, `value_and_jacobian` | `J(x)`, `m × n` (and `f(x)`, `m × 1`) | `n` `Dual` passes, a column each |
| `hessian`, `value_gradient_and_hessian` | `∇²f(x)`, `n × n` symmetric (and `f(x)`, `∇f(x)`) | `n(n + 1)/2` `HyperDual` passes, an entry each |

- `+`, `-`, `*` and `/` take two numbers of a type, owned or borrowed, or one and a float of its underlying type on either side (`2.0 * x`, `x / 2.0`, `1.0 - x` for `Dual<f64>`); the assignments, `Neg`, `Sum` and `Product` too. A quotient is solved part after part (`q₁ = (a₁ − q₀·b₁)/b₀`), which does not square the divisor.
- `==`, `<` and `partial_cmp` compare the values alone, with each other and with a float: `Dual::new(1.0, 2.0) == Dual::new(1.0, 3.0)`, as in num-dual. A branch on a comparison then takes the same path for the value and for its derivatives, and `partial_cmp` agrees with `==`.
- `Display` writes `1.5-2ε` and `1-2ε₁+0.5ε₂+3ε₁ε₂`, each part with the precision given, a part that is itself a dual number in parentheses: `(1+1ε)+(1+0ε)ε`.
- The elementary functions, each a method of `DualNum`, with the value from the function of the standard library and the derivative parts from the derivatives of the function at the value: `recip`, `sqrt`, `cbrt`, `powi`, `powf` (a float exponent), `pow` (an exponent that varies too), `exp`, `exp2`, `exp_m1`, `ln`, `log2`, `log10`, `ln_1p`, `log` (any base), `sin`, `cos`, `tan`, `sin_cos`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `abs`, `hypot`, `mul_add`, `min`, `max` and `signum`. The one exception to the standard library is `atanh` of a negative `f32` or `f64`, taken as `−atanh(−x)`: the standard library's `0.5·ln_1p(2x/(1 − x))` is wrong by about 10¹⁴ ulps at `−1 + 2⁻⁵³` (and by 3.6·10⁵ at `−1 + 2⁻²⁴` in `f32`), though not near 1.

## Examples

```rust
use a::math::dual::{Dual, DualNum, HyperDual, derivative, gradient, hessian, second_derivative};
use a::math::matrix::Matrix;

// By hand: f(x) = x·sin x at x = 2, its value and derivative sin x + x·cos x.
let x = Dual::new(2.0_f64, 1.0);
let f = x * x.sin();
assert_eq!(f.re, 2.0 * 2.0f64.sin());
assert!((f.eps - (2.0f64.sin() + 2.0 * 2.0f64.cos())).abs() < 1e-15);

// A function written once, for the value and for every derivative.
fn g<D: DualNum<Float = f64>>(x: D) -> D {
    (x * x + 1.0).ln() - x.atan() * 2.0
}
assert_eq!(derivative(g, 1.0), 0.0);                        // 2x/(1 + x²) − 2/(1 + x²) at 1
assert_eq!(second_derivative(g, 1.0), 1.0);                 // (2 + 4x − 2x²)/(1 + x²)² at 1
let h = g(HyperDual::new(1.0, 1.0, 1.0, 0.0));
assert_eq!((h.eps1, h.eps1eps2), (0.0, 1.0));

// The third derivative by nesting: of x⁴ at 1.5, 24x = 36.
let t = Dual::new(HyperDual::new(1.5, 1.0, 1.0, 0.0), HyperDual::new(1.0, 0.0, 0.0, 0.0)).powi(4);
assert_eq!(t.eps.eps1eps2, 36.0);

// Several variables: f(x, y) = x²y + eʸ at (1, 0).
fn f2<D: DualNum<Float = f64>>(v: &[D]) -> D {
    v[0] * v[0] * v[1] + v[1].exp()
}
let p = Matrix::from_column_major(2, 1, vec![1.0, 0.0]);
assert_eq!(gradient(f2, &p).as_slice(), &[0.0, 2.0]);
assert_eq!(hessian(f2, &p), Matrix::from_rows(&[[0.0, 2.0], [2.0, 1.0]]));
```

The sections below give how forward mode works and what it costs, how to write a function once, the derivatives at the edges, the functions with the solvers of `a::math::numerics`, what was verified against what, and the limits; the times against num-dual are [last](#measured).

## Forward mode and what it costs

A dual number carries one direction. Every operation applies its rule to the value and to the `ε` part: `(a + bε)(c + dε) = ac + (ad + bc)ε`, and a function `f(a + bε) = f(a) + f′(a)·b·ε`. Evaluating a function at `x + v·ε` (each variable `xᵢ + vᵢ·ε`) therefore gives the derivative along `v` in the `ε` part, exactly as the chain rule composes the derivatives of the operations, each rounded once or a few times. A hyper-dual number carries two directions and their cross term: a function is `f(a) + f′(a)·(b·ε₁ + c·ε₂) + (f′(a)·d + f″(a)·b·c)·ε₁ε₂` (Fike and Alonso, "The development of hyper-dual numbers for exact second-derivative calculations", AIAA 2011-886: the Taylor series of `f` about `a`, which ends there as `ε₁² = ε₂² = 0`), so with `ε₁` on `xᵢ` and `ε₂` on `xⱼ` the `ε₁ε₂` part of `f` is `∂²f/∂xᵢ∂xⱼ`. A function of two numbers (`atan2`, `hypot`, `pow`, `mul_add`) takes its two first and three second partial derivatives in the same way.

So a derivative is one dual pass, a gradient or a Jacobian of `n` variables `n` passes (one per variable, giving a column of the Jacobian, all `m` rows at once), a second derivative one hyper-dual pass, and a Hessian `n(n + 1)/2` hyper-dual passes, one for each entry on and above the diagonal; the diagonal passes carry the gradient too, which `value_gradient_and_hessian` returns. Within a pass a product of dual numbers takes three products of floats and a product of hyper-dual numbers nine, and an elementary function its value and its derivatives (`sin` and `cos` from one `sin_cos`, `exp` once for all three parts). `Dual<HyperDual<f64>>` holds eight floats and a product takes 27 float products.

## Writing a function once

A function generic over `D: DualNum` gives its value for `f64` and its derivatives for the dual types; it is passed to the free functions by name (`derivative(f, x)`, `gradient(rosen, &x)`), which pick the dual type, and called with a slice for the value (`rosen(x.as_slice())`, as `Matrix::as_slice` gives a column). Closures are not generic in Rust, so a closure passed to them is written for the type they ask: `|x: Dual<f64>| x.sin()`, `|v: &[Dual<f64>]| v[0] * v[1]`, `|v: &[HyperDual<f64>]| …`.

With the float fixed, `D: DualNum<Float = f64>`, float literals work on the right of an operator (`x * 2.0 - 1.0`) and in comparisons (`x > 0.0`, `x.re() > 0.0`). On the left they do not by themselves: `2.0 * x` needs `f64: Mul<D>`, which the trait cannot promise (a bound on its associated type would hold of `D::Float`, and is not used once that is known to be `f64`), and the orphan rule forbids an implementation of it for every `D`. A function may ask for it, `where f64: Mul<D, Output = D>`, which every dual type of `f64` meets; or a float on the left is written `D::constant(2.0) * x`, or the operation turned round. With the float generic too, `D: DualNum`, literals are `D::from_f64(2.0)`. The concrete types take a float on either side. `Dual<T>` and `HyperDual<T>` are `DualNum` for every `T: DualNum` with no further condition, so code generic over the float can build dual numbers of its own, and the free functions nest: `derivative(|x| second_derivative(f, x), x0)` is a third derivative, with `f` generic.

`re()` is the innermost value, the same for `f64` and for every dual type around it, and the comparisons look at it alone, so a branch takes the same path for the value and for its derivatives: the derivative is that of the branch taken.

## The derivatives at the edges

The derivative of each function is formed so that it keeps its digits where the plain formula would cancel or overflow, and so that the nested types, which differentiate the formula itself, do not lose theirs:

| Function | Derivative | Formed as | At the edges |
| --- | --- | --- | --- |
| `recip` | `−1/x²` | `−r·r`, `r = 1/x` | `−∞` at 0 |
| `sqrt` | `1/(2√x)` | | `+∞` at either zero (from the right); NaN below 0 |
| `cbrt` | `1/(3∛x²)` | `f″` as `f′·(−2/3)/x` | `+∞` at either zero; `f″` is `∓∞` at `±0` |
| `powi(n)` | `n·xⁿ⁻¹` | `xⁿ/xᵏ` where `n − k` leaves `i32`; where `xⁿ⁻¹` alone falls below the normal numbers (subnormal, or 0 where std's `powi` forms a negative power as `1/x^|m|` and `x^|m|` overflows), in two halves with the coefficient between them, `(x^(m − ⌊m/2⌋)·n)·x^⌊m/2⌋` | 0 for `n = 0`, also at 0; `f″` 0 for `n = 1`; normal derivatives of subnormal powers (`(x⁻¹⁰⁰⁰)′` at 2.03) |
| `powf(e)` | `e·xᵉ⁻¹` | `e·x^e₁·(1 + δ·ln x)`, `e₁ = e − 1` rounded and `δ` its exact remainder, corrected only where `x^e₁` is finite and nonzero; `(e·xᵉ)/x` where `x^e₁` alone leaves the range; for `|e| ≥ 2ᵖ − 2` (`2⁵³ − 2` in `f64`, `2²⁴ − 2` in `f32`) or infinite, where every float is an integer and `e − 1` or `e − 2` need not be a float, `e·xᵉ/x` and `e(e − 1)·xᵉ/x²`, and at `x = ±0` and `±∞` `|x|^(e − k)` with the parity of `e − k`; `f″` alike | 0 for `e = 0`; at 0, 0 for `e > 1`, 1 for `e = 1`, `∞` for `e < 1`; below 0 finite for an integer `e`, with the parity of `e − 1` and `e − 2` from `2ᵖ − 2` on (`(x^(2⁵³ + 2))′` at −1 is `−(2⁵³ + 2)`, and `f″` of `x^(−(2⁵³ − 1))` there is negative), NaN otherwise; at 0 also for `|e| ≥ 2ᵖ − 2`; at `x = 1` an infinite `e` gives `f′ = e`, `f″ = +∞`; within a factor `|e|` (`f′`) or `|e(e − 1)|` (`f″`) of the normal numbers only a subnormal's digits, or 0 (see *Limits*) |
| `pow(y)` | `y·xʸ⁻¹·dx + xʸ·ln x·dy` | the `dx` terms as `powf`'s; `∂²/∂x∂y = xʸ⁻¹(1 + y ln x)` | the `dy` terms 0 at `x = 0` where `xʸ` is 0 (their limit for `y > 0`, but for `∂²/∂x∂y` at `0 < y ≤ 1`, whose limit is `−∞`), formed where `xʸ` only underflows, NaN below 0 when `y` varies; with `y` constant the derivative of `x^y` below 0 too |
| `exp_m1` | `eˣ` | | `1` at `10⁻²⁰`; `0` at `−∞` |
| `ln`, `log2`, `log10`, `log` | `1/(x·ln b)` | `1/((x + 0)·ln b)` for `|ln b| ≥ 1`, `(1/(x + 0))/ln b` below, the `+0` making `−0` positive; `f″ = −f′·(f′·ln b)` | `+∞` at either zero (`−∞` for a base below 1); no intermediate overflows before `f″` does (`log` to the base 0.027 at `4.4·10⁻¹⁵⁵`, `f32`'s `log10` at `4.4·10⁻²⁰`) |
| `ln_1p` | `1/(1 + x)` | | `+∞` at −1 |
| `tan` | `1 + tan²x` | | |
| `asin`, `acos` | `±1/√(1 − x²)` | `(1 − x)(1 + x)` for `|x| ≥ ½`, `1 − x²` below | `±∞` at ±1, NaN beyond |
| `atan` | `1/(1 + x²)` | `r²/(1 + r²)`, `r = 1/x`, for `|x| > 1` | 0 at `±∞` and at `±10³⁰⁰`, with no overflow |
| `atan2(y, x)` | `(x·dy − y·dx)/(x² + y²)` | through `r = hypot(x, y)`: `(x/r)/r`; `∂²/∂y∂x = (y − x)(y + x)/r⁴` from the difference of `y` and `x` themselves, of their halves beyond half the largest float | NaN at the origin, where the angle is not continuous; the mixed partial keeps its digits next to the diagonals (at `(1 + 2⁻⁵², 1)`) and is 0, not NaN, at `(10³⁰⁸, ±10³⁰⁸)` |
| `tanh` | `1/cosh²x` | `1/cosh x` as `2e^(−|x|)` beyond 20 | keeps its digits for large `|x|`, where `1 − tanh²x` is 0 (at 30: `3.5·10⁻²⁶`) |
| `asinh` | `1/√(1 + x²)` | `1/hypot(1, x)` | does not overflow for large `|x|` |
| `acosh` | `1/√(x² − 1)` | `1/(√(x − 1)·√(x + 1))` | `+∞` at 1; keeps its digits near 1 and does not overflow |
| `atanh` | `1/(1 − x²)` | as `asin` | `+∞` at ±1 |
| `abs` | the sign | | 0 at either zero, the middle of −1 and 1 |
| `hypot` | `(x·dx + y·dy)/h` | `x/h`, `y/h`; `y²/h³` as `(y/h)·((y/h)/h)` | every derivative part 0 at the origin, the slope of least size; the second partials do not underflow before their result at extreme ratios (`(3·10⁻⁸¹, 1.7·10⁻²⁵⁵)`) |
| `min`, `max` | that of the operand chosen | | of `self` on a tie; a NaN operand is ignored, as by the floats |
| `signum` | 0 | | 0 at the jump too |

In the elementary functions a tangent part that is zero stays zero, whatever the derivative it multiplies: in a pass of a gradient the variables that do not move are constants, and `√y` at `y = 0` would otherwise turn `∂/∂x (x + √y)` into `∞·0 = NaN`; with the rule it is 1. **The rule has a price**: it cannot tell a variable that does not move from an intermediate whose derivative happens to be 0 at the point (or underflows), and where such an intermediate meets an infinite derivative, the true limit of `0·∞` is replaced by 0 — a wrong finite number where the composite is smooth, with nothing to signal it. The derivative of `cbrt(x³)` at 0 comes back as 0, not 1, and so do those of `(x³)^(1/3)` and `cbrt(sin³x)` at 0 and of `cbrt(x³)` at `10⁻³⁰⁰` (where `x³` underflows); the second derivative of `cbrt(x⁶)` at 0 as 0, not 2; the gradient of `cbrt(x³ + y³)` at the origin as `(0, 0)`, not `(1, 1)`. num-dual gives NaN at these points. At a kink, as of `√(x²)` or `(x⁴)^(1/4)` at 0, the 0 is the choice `abs` makes too. The arithmetic is the floats': an infinite value times a zero tangent part is NaN there (see *Limits*). Outside a function's domain the value is NaN and the derivative is what the formula gives there (`ln` at −2: −0.5): check the value. Infinities and NaN in the value propagate as the floats' do.

## With the numerical methods

The solvers of [`a::math::numerics`](numerics.md) take derivatives as closures; the free functions give them in the shapes asked.

```rust
use a::math::dual::{DualNum, derivative, gradient, hessian, jacobian, second_derivative};
use a::math::matrix::Matrix;
use a::math::numerics::{
    OdeOptions, RadauIia, bfgs, halley, integrate_with, levenberg_marquardt, newton, newton_system,
};

fn column(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

// Newton's and Halley's methods with exact first and second derivatives.
fn f<D: DualNum<Float = f64>>(x: D) -> D {
    x.cos() - x
}
let r = newton(0.5, f, |x| derivative(f, x)).unwrap();
let s = halley(0.5, f, |x| derivative(f, x), |x| second_derivative(f, x)).unwrap();
assert!((r - 0.739_085_133_215_160_7).abs() < 1e-15 && (s - r).abs() < 1e-15);

// BFGS on Rosenbrock's function with its gradient (conjugate_gradient and lbfgsb alike).
fn rosen<D: DualNum<Float = f64>>(x: &[D]) -> D {
    let a = D::constant(1.0) - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + b * b * 100.0
}
let m = bfgs(|x| rosen(x.as_slice()), |x| gradient(rosen, x), &column(&[-1.2, 1.0])).unwrap();
assert!(m.converged && (m.x[(0, 0)] - 1.0).abs() < 1e-7);

// Newton's method for a system with its Jacobian (broyden_with_jacobian alike).
fn sys<D: DualNum<Float = f64>>(v: &[D]) -> [D; 2] {
    [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1] - 1.0]
}
let z = newton_system(&column(&[1.5, 0.5]), |x| column(&sys(x.as_slice())), |x| jacobian(sys, x)).unwrap();
assert!(z.converged && (z.x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-15);

// Levenberg–Marquardt with the Jacobian of the residual (gauss_newton alike).
fn res<D: DualNum<Float = f64>>(x: &[D]) -> [D; 2] {
    [(x[1] - x[0] * x[0]) * 10.0, D::constant(1.0) - x[0]]
}
let fit = levenberg_marquardt(|x| column(&res(x.as_slice())), |x| jacobian(res, x), &column(&[-1.2, 1.0])).unwrap();
assert!(fit.converged && (fit.x[(1, 0)] - 1.0).abs() < 1e-10);

// Radau IIA (and the BDFs alike) on Robertson's stiff reactions, with the exact Jacobian.
fn rober<D: DualNum<Float = f64>>(y: &[D]) -> [D; 3] {
    let a = y[0] * -0.04 + y[1] * y[2] * 1e4;
    let c = y[1] * y[1] * 3e7;
    [a, -a - c, c]
}
let rhs = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
    dy.as_mut_slice().copy_from_slice(&rober(y.as_slice()));
};
let jac = |_t: f64, y: &Matrix<f64>, j: &mut Matrix<f64>| *j = jacobian(rober, y);
let opts = OdeOptions::new().with_tolerances(1e-8, 1e-12);
let y = integrate_with(rhs, &column(&[1.0, 0.0, 0.0]), 0.0, 40.0, RadauIia::new().with_jacobian(jac), &opts).unwrap();
assert!((y.y[(0, 0)] - 0.715_827_068_7).abs() < 1e-6 && y.njev > 0);

// Newton's method for a minimum, x ← x − H⁻¹·∇f, with the exact Hessian.
let mut x = column(&[-1.2, 1.0]);
for _ in 0..10 {
    let g = gradient(rosen, &x);
    if g.as_slice().iter().all(|v| v.abs() < 1e-13) {
        break;
    }
    x = &x - &hessian(rosen, &x).solve(&g).unwrap();
}
assert!((x[(0, 0)] - 1.0).abs() < 1e-14 && (x[(1, 0)] - 1.0).abs() < 1e-14);
```

## Verified

- **SymPy and mpmath.** 463 points of 51 functions of one variable and 51 points of `atan2`, `hypot` and `pow` (`lib/tests/math/dual/data/reference.txt`, written by `gen_reference.py`). The functions of one variable are:
  - the 22 functions of one argument alone;
  - `powi` for `n` = −41, −12, −3, 0, 1, 2, 5, 31;
  - `powf` for 12 exponents: 0.5, −1.5, 2.5, 3 and −2 (the last two below 0 too); 0.3, 0.1, ⅓, `−7.5 + 2⁻⁵⁰` and `1 + 2⁻³⁰` in `f64`, whose `e − 1` is not a float, from `10⁻³⁰⁰` to `10³⁰⁰`; and `0.3` and `e` rounded to `f32`, which test `f32` the same way;
  - `log` to the bases 3, ½ and 0.027;
  - six compositions written alike in the generator and the tests: `sin(eˣ)/(1 + x²)`, `ln(1 + x²)·atan³x`, `√cosh x·tanh(x/2) − x^(5/2)`, `e^(−x²)·cos 3x`, `asinh(x³ − x)/(2 + sin x)`, `atan2(sin x, cos 2x + 2)·√(1 + x²)`.

  Each reference gives the value and three derivatives, or the value and the first and second partial derivatives. They come from SymPy 1.14's symbolic differentiation, evaluated by mpmath 1.3 at 600 digits and again at 700, and the two must round to the same `f64`: SymPy's `1 − tanh²x` cancels to 0 at 50 digits for `|x| > 58`, and so does `eˣ − 1` at `10⁻³⁰⁰`. Every exponent and base is the float itself as an exact rational, not the decimal it was written as.

  The grids take in the edges:
  - `1 ± 2⁻⁵³` and `1 ± 2⁻²⁴` for `asin`, `acos`, `atanh`, `acosh` and `ln`, and `−1 + 2⁻⁵³` for `ln_1p`;
  - `10⁻³⁰⁰` to `10³⁰⁰`; `π/2` and `π` rounded for the tangent and the sine; `tanh` to 350, `exp` to 700, and `sinh` and `cosh` to 710;
  - 0 for the functions whose derivatives are finite there (the infinite ones are among the edges below);
  - the axes and quadrants for `atan2`, and points next to its diagonals (`(1 + 2⁻⁵², ±1)`, `(3(1 + 2⁻⁵⁰), 3)`);
  - extreme ratios for `hypot`;
  - `pow` where `xʸ` underflows (`(10⁻²⁰⁰, 2)`), and where `x^(y−1)` alone overflows or underflows though its products do not;
  - the ends of the range where an intermediate used to leave it before the result: `cbrt` at `5.1·10⁻¹⁸⁵`, `log` to the base 0.027 at `4.4·10⁻¹⁵⁵`, `f32`'s `log10` at `4.4·10⁻²⁰`, and `x⁻¹²` and `x⁻⁴¹` where `powi`'s negative power is 0.

  379 of the 463 and 28 of the 51 are floats of `f32` too, with their parameters, and test both types. Every point is evaluated through every form: the value; `f′` by `Dual`; `f′` in both directions and `f″` by `HyperDual`; `f″` by `Dual<Dual<_>>`; `f‴` by `Dual<HyperDual<_>>` and by `HyperDual<Dual<_>>`; and the cross term `f′·d + f″·b·c` of `HyperDual::new(x, 0.75, −1.25, 0.5)`. That makes 8,733 comparisons of components in the normal range of their type.

  The bounds are 4 ulps for values, 8 for first, 16 for second and 32 for third derivatives, eight times that for `powi` of a power beyond ±5 and for the compositions; every component is within its bound. For `powi`, the standard library's power is itself off by up to 7 ulps at `x⁻⁴¹`. In the compositions, their own condition amplifies the rounding of inner values (`sin(eˣ)` at 4). Both tests, the bounds and the printed table, assert the bounds.

  The largest error seen, over all forms and over both platforms, is below. It is in ulps of the reference rounded to the type (the rounded `f64` reference is itself within half an ulp). The platforms are an Apple M4 Max (macOS, aarch64) and x86_64 (Ubuntu 26.04's glibc, in the CI container). Their maxima differ by at most 3 ulps in an entry for the functions (in `tan`'s `f″`), and by up to 48 for the compositions (in one form of the `f‴` of `atan2(sin x, cos 2x + 2)·√(1 + x²)`, 57 on one platform and 9 on the other):

| Function | value f64 | f′ f64 | f″ f64 | f‴ f64 | value f32 | f′ f32 | f″ f32 | f‴ f32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `recip` | 0 | 1 | 2 | 3 | 0.3 | 1.1 | 1.1 | 2.1 |
| `sqrt` | 0 | 1 | 3 | 4 | 0.4 | 1.0 | 3.1 | 4.0 |
| `cbrt` | 0 | 1 | 5 | 5 | 0.5 | 1.9 | 6.2 | 7.6 |
| `powi(−3)`, `(0)`, `(1)`, `(2)`, `(5)` | 1 | 1 | 1 | 1 | 1.6 | 1.7 | 1.8 | 2.1 |
| `powi(31)`, `(−12)`, `(−41)` | 7 | 5 | 8 | 3 | 10 | 16 | 13 | 10 |
| `powf(0.5)`, `(−1.5)`, `(2.5)`, `(3)`, `(−2)` | 0 | 1 | 1 | 1 | 0.4 | 0.9 | 0.8 | 0.8 |
| `powf(0.3)`, `(0.1)`, `(⅓)`, `(−7.5 + 2⁻⁵⁰)`, `(1 + 2⁻³⁰)` (f64) | 0 | 1 | 1 | 2 | – | – | – | – |
| `powf(0.3f32)`, `(e as f32)` | 0 | 1 | 1 | 1 | 0.5 | 0.6 | 2.1 | 2.4 |
| `pow` (both partials) | 0 | 1 | 1 | – | 0.4 | 0.6 | 1 | – |
| `exp`, `exp_m1`, `sinh`, `cosh` | 1 | 1 | 1 | 1 | 0.5 | 0.5 | 0.5 | 0.5 |
| `exp2` | 0 | 0 | 1 | 1 | 0.3 | 0.6 | 0.3 | 0.1 |
| `ln`, `ln_1p` | 0 | 0 | 1 | 2 | 0.4 | 0.5 | 1.1 | 1.9 |
| `log2`, `log10` | 0 | 1 | 3 | 4 | 0.4 | 0.7 | 1.7 | 2.8 |
| `log(3)`, `log(½)`, `log(0.027)` | 1 | 1 | 3 | 3 | 1 | 0.7 | 1.8 | 2.3 |
| `sin`, `cos` | 1 | 1 | 1 | 1 | 0.5 | 0.5 | 0.5 | 0.5 |
| `tan` | 1 | 2 | 5 | 4 | 0.5 | 0.6 | 1.4 | 2.1 |
| `asin`, `acos` | 1 | 1 | 3 | 4 | 0.5 | 0.8 | 2.7 | 2.9 |
| `atan` | 1 | 1 | 2 | 2 | 0.6 | 0.8 | 1.6 | 2.3 |
| `atan2` (both partials) | 0 | 2 | 4 | – | 0.6 | 1.1 | 3 | – |
| `tanh` | 0 | 2 | 3 | 6 | 0.3 | 1.5 | 2.4 | 4.1 |
| `asinh` | 1 | 1 | 2 | 5 | 0.5 | 0.6 | 3.0 | 3.3 |
| `acosh` | 1 | 1 | 4 | 7 | 0.5 | 2.5 | 3.8 | 5.9 |
| `atanh` | 1 | 1 | 2 | 2 | 0.3 | 1.4 | 2.5 | 3.0 |
| `hypot` (both partials) | 0 | 1 | 3 | – | 0.5 | 0.8 | 3.1 | – |
| the six compositions | 9 | 38 | 40 | 57 | 9.8 | 42 | 37 | 59 |

  The cross term is within 5.0 ulps (`f64`) and 9.1 (`f32`) of the larger of its two terms, on both platforms.
- **The hard cases, one by one** (`hard.rs`, each against mpmath at 60 digits):
  - `0.1·x^(−0.9)` at `10³⁰⁰` by `powf` and `pow`, and `(x^(−7.5 + 2⁻⁵⁰))′` at 1000, within 2 ulps. Before the exponent's remainder was kept, these were 145 and 27 ulps off; over the auditor's sweep of 4,000 points of `x ∈ [10⁻³, 10³]`, `e ∈ [−8, 8]`, up to 48 ulps.
  - `pow`'s mixed partial at `(10⁻²⁰⁰, 2)` and `(10⁻¹⁰⁰, 3.5)`, by `HyperDual` and `hessian`, within 4 ulps, where it was 0.
  - `atan2`'s at `(1 + 2⁻⁵², ±1)` and `(3(1 + 2⁻⁵⁰), 3)`, within 4, where it was 41 % and 6 % off.
  - `hypot`'s second partials at `(−3.1·10⁻⁸¹, −1.7·10⁻²⁵⁵)` and `(10⁻¹⁰⁰, 10⁻²⁶⁰)`, where they were 0 and `10⁻⁵` off.
  - `cbrt″` at `5.1·10⁻¹⁸⁵`, `log₀.₀₂₇″` at `4.4·10⁻¹⁵⁵` (also by `Dual<Dual<_>>`), `f32`'s `log10″` at `4.4·10⁻²⁰`, `f32`'s `(x⁻¹²)′` at 978.7 and `(x⁻⁴¹)″` at `−1.5·10⁷`, each near the end of the range, each within 4 ulps (16 for `x⁻⁴¹`, whose power is std's). Before, these were `±∞` or 0.

  - In a second round (`hard.rs`): `atan2`'s mixed partial at `(±10³⁰⁸, ±10³⁰⁸)`, `(1.7·10³⁰⁸, 10³⁰⁸)` and `(10³⁰⁸, 9·10³⁰⁷)` is 0, where the sum of the operands overflowed to NaN or `∞`. The derivatives of `x^(2⁵³ + 2)` at −1 and −1.0000000000000002 have the sign of `x^(e−1)` (−9.007·10¹⁵ and −6.655·10¹⁶, the second within 4 ulps), and so does `f32`'s `x^16777218` at −1. Those of `x^(2⁶⁰)` at 3, `x^(10¹⁷)` at 10¹⁰, `x^(10³⁰⁰)` at 2 and `f32`'s `x^(2³⁰)` at 3 are `+∞`, not `−∞`. `x^±∞` at 1 gives `±∞` and `+∞`. `(x⁻¹⁰⁰⁰)′` at 2.03 and `(x⁻¹⁰⁷⁴)′` at 1.94 (also `f″` by `HyperDual`) and `f32`'s `(x⁻¹⁰⁰)′` at 2.47 are normal numbers, where they were 0. They are 15, 160 and 9.8 ulps of the result off (`f″` 14 and 87), the standard library's own `powi` error carried over from the two halves: its `x⁻⁵³⁷` and `x⁻⁵³⁸` at 1.94 are 55 and 57 ulps off.
  - In a third round: the second derivatives at `x = −1` on both sides of `2⁵³ − 2`, for `e` from `±(2⁵³ − 3)` to `±(2⁵³ + 2)`, by `HyperDual` and by `HyperDual<Dual<_>>`, have the sign of `e(e − 1)·(−1)^(e−2)` (`f″` of `x^(−(2⁵³ − 1))` is −8.11·10³¹, where it was +8.11·10³¹), and so does `f32`'s for `−(2²⁴ − 1)`. For `e` = 2⁵³, 2⁵³ + 2, 2⁶⁰ and 10³⁰⁰, `f′` and `f″` are 0 at ±0 (also by `pow`), `+∞` at `+∞`, and `−∞` and `+∞` at `−∞`, where they were NaN.

  The zero-tangent rule's limit is pinned at the values it gives (0 for `cbrt(x³)′`, `((x³)^(1/3))′`, `cbrt(sin³x)′` at 0, `cbrt(x³)′` at `10⁻³⁰⁰`, `cbrt(x⁶)″` at 0, and `∇cbrt(x³ + y³)` at the origin), beside the case it is for (`∂/∂x (x + √y)` at `y = 0`, 1).
- **The audit's sweep, rerun.** An independent audit swept 210,000 seeded points (and 30,500 more for the logarithms and for `powf` at moderate arguments). It covered every function in `f64` and `f32`, through the same forms as above, against closed forms in mpmath at 50 and 75 digits stored as hi + lo. Its generator and data are not in the crate. Rerun against this code, the findings it raised are gone:
  - `powf`'s `f′`, `f″` and `f‴` are within 2.3, 3.7 and 3.8 ulps, where they were 95, 158 and 480 (48 to 53 at moderate arguments);
  - `pow`'s `fx`, `fxx` and `fxy` by `HyperDual` are within 2.3, 2.6 and 7.3 ulps, the last but next to the zeros of `1 + y ln x` (see *Limits*);
  - `atan2`'s mixed partial by `HyperDual` is within 6.8 ulps, where it was 5.7·10¹⁴;
  - `hypot`'s second partials are within 5.3 ulps, where 11 points were beyond 100;
  - `cbrt″` is within 4.6 ulps;
  - the second derivatives of `log`, `log2` and `log10` are within 4 ulps, by `HyperDual` and by `Dual<Dual<_>>`, where some were infinite;
  - `powi`'s derivatives are within 36 ulps, std's own error carried along, where some were 0 or 10¹⁵ ulps off.

  Its findings that remain are the limits stated below: the nested forms at the ends of the range and `atan2`'s mixed partial by `Dual<Dual<_>>`.
- **The rules on random pairs.** 2,000 random pairs `(f, g)` of 14 smooth pieces at random points of `[−2, 2]`. The pieces are the circular and hyperbolic functions and their inverses, logarithms, roots and powers of `1 + x²`, `hypot`, `atan2` and `exp_m1`. The hyper-dual product, quotient (by `g + 3`) and composition agree with `(fg)″ = f″g + 2f′g′ + fg″`, the quotient rule to second order, and `(f∘g)″ = f″(g)·g′² + f′(g)·g″`. The comparison is formed from the pieces' own hyper-dual values, within 64 `ε` of the sizes of the terms.
- **Richardson extrapolation.** 300 random compositions of three pieces at random points:
  - The dual derivative agrees with `numerics::derivative` within 10⁻⁷ relative wherever Richardson answers (at least 90 % of the points); the largest difference seen is 7.6·10⁻¹².
  - The hyper-dual second derivative agrees with `numerics::second_derivative` within 10⁻⁴; the largest seen is 1.3·10⁻⁹.
  - On 300 more, the third derivative four ways agrees within 10⁻¹² (10⁻⁶ for Richardson): `Dual<HyperDual<_>>`, `HyperDual<Dual<_>>`, `derivative` of `second_derivative` nested through the free functions, and Richardson on the exact second derivative.
  - The same compositions give bit for bit the same value in `f64`, `Dual`, `HyperDual` and `Dual<Dual<_>>`, and in `f32` and `Dual<f32>`.
- **Gradients, Jacobians and Hessians.**
  - `e^(xy)·sin z + ln(1 + x² + y²)·z + atan(x − z²)` at four points against SymPy and mpmath (the `g` lines of the data): value within 4 ulps, gradient within 8, Hessian within 16, and in `f32` within 64 `f32` ulps. `gradient`, `value_and_gradient` and `hessian` give the same numbers as `value_gradient_and_hessian`.
  - Each Hessian entry agrees, within 8 ulps, with the pass whose `ε₁` and `ε₂` are swapped and with `∂/∂xᵢ(∂f/∂xⱼ)` by `Dual<Dual<_>>`. The Jacobian of the gradient is the Hessian within 10⁻¹⁴.
  - Exact cases: the gradient `Ax + b` and the Hessian `A` of `½xᵀAx + bᵀx`, and the Jacobians of a linear map, of polar coordinates, and of spherical ones (whose determinant `r² sin θ` is within 10⁻¹⁵).
  - A function that counts its calls is called 5, 5 and 15 times for the gradient, the Jacobian and the Hessian of 5 variables.
  - An empty point gives empty results, calling `f` once where a value is asked for.
  - A row, a square matrix, a `0 × 0` matrix and a function whose number of outputs changes each panic with their message.
- **num-dual.** A probe outside the crate (num-dual 0.16 with nalgebra 0.35) evaluated, in `f64`, the 45 functions and parameters of the data that num-dual has (all but the compositions) at their 417 points. Each was evaluated by num-dual's `Dual64` and `HyperDual64` and by this crate's, both against the references.
  - The two agree within 2 ulps in the first derivative and 7 in the second, except where num-dual's own error is the larger:
    - Its `tanh′` and `tanh″` are 0 for large `|x|`, already at 20 (5.5·10¹⁵ ulps off there, against 0 here).
    - Its `acosh′` and `asinh′` are 0 from about `10¹⁵⁴`, where `x²` overflows. Near 1 its `acosh′` matches this crate's at `1 + 2⁻²³` and `1 + 2⁻⁵²` (0 and 1 ulp) but is 129 ulps off at 1.001 and 1.7·10⁷ at 1.000000008, where this crate's is within 1.
    - With an exponent whose `e − 1` is not a float, its `powf′` is off by up to 9.0 ulps for `e` = 0.1, 28.6 for ⅓, 77.0 for `−7.5 + 2⁻⁵⁰` and 0.8 for `1 + 2⁻³⁰` (1 here), and it returns 0 at `10³⁰⁰` and `∞` at `10⁻³⁰⁰` where the derivative is finite.
    - Its `x⁻⁴¹` derivatives at `−1.5·10⁷` are 0.
  - num-dual's `gradient`, `jacobian` and `hessian` of the function of three variables, and of a map of it to two values, at five points, agree with this crate's within 0, 0 and 8 ulps.
  - At the `0·∞` points of the zero-tangent rule, num-dual gives NaN and this crate 0.
- **The solvers.**
  - `newton` with `derivative` on `eˣ − 3x²` from 1 gives the root of the hand-written derivative within 2 `ε`. `halley` with `derivative` and `second_derivative` finds the root of `cos x = x`, and `√2` within four steps (`max_iter` 4), as its documentation says. `newton` in `f32` finds `∛2`.
  - `bfgs` with `gradient` converges on Rosenbrock's function from `(−1.2, 1)`, on Beale's from `(1, 1)`, and on Rosenbrock's in four variables with `gtol` 10⁻¹⁰, where the gradient at the answer is below 10⁻¹⁰. `conjugate_gradient` converges on Rosenbrock's and Beale's. `lbfgsb` converges on Rosenbrock's held to `x₀ ≤ ½` (the answer `(½, ¼)` on the bound, by the projected gradient test) and in ten variables.
  - `newton_system` with `jacobian` finds `(1, 2, 3)` of `x + y + z = 6, x² + y² + z² = 14, xyz = 6` within 10⁻¹⁴, and `broyden_with_jacobian`, from the exact first Jacobian, within 10⁻¹².
  - `levenberg_marquardt` and `gauss_newton` with the Jacobian of the residual fit `a·e^(−λt)` to 10⁻¹⁰, and Levenberg–Marquardt solves Rosenbrock's as least squares.
  - `RadauIia` and `Bdf` with `with_jacobian` and the `jacobian` of Robertson's reactions reach t = 40 within 10⁻⁶ of SciPy's Radau at `rtol` 10⁻¹², with mass conserved to 10⁻¹⁰.
  - Newton's method for a minimum with `hessian` reaches `(1, 1)` from `(−1.2, 1)` within 10⁻¹⁴ in at most ten steps, the gradient squaring at each step at the end.
- **Edges.** Each entry of the table of edges above is a test (`edges.rs`):
  - both zeros for the roots and the logarithms;
  - `x^e` at 0 for `e` = 0, ½, 1, 2 and −1, and below 0;
  - `powi` at `i32::MIN`, `i32::MIN + 1` and `i32::MAX`;
  - `pow` with a negative base and a constant exponent, with a varying one, and at 0;
  - `asin`, `acos`, `atanh` and `acosh` at their branch points and beyond, and `acos` at `1 − 2⁻⁵³`;
  - `tanh` at 30, ±1000 and ±∞, and in `f32` at 100;
  - `atan2` on the negative axis with both zeros, and at the origin;
  - `hypot` at the origin and at 10³⁰⁰;
  - `abs` and `signum` at both zeros, and `min` and `max` at ties and with a NaN;
  - a tangent of zero at a singular point, infinities and NaN.

  The arithmetic is tested in every form of operand, with the quotient at `10³⁰⁰` and `10⁻³⁰⁰`, `Sum` and `Product` of none, the comparisons and `Display`.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 58 tests pass, and the 20 doctests of the module and of this page.

To run the tests: `cargo test --test math -- dual::`, with `-- --nocapture` on `dual::reference::largest_errors_seen` for the table. The data file is written again by `python3 gen_reference.py > reference.txt` in `lib/tests/math/dual/data`, with SymPy and mpmath installed.

## Limits

- **The zero-tangent rule at `0·∞`.** A tangent part that is exactly zero contributes nothing in the elementary functions, even against an infinite derivative. That keeps a gradient, Jacobian or Hessian usable where another variable sits at a point of infinite derivative (`∂/∂x (x + √y)` at `y = 0` is 1, not NaN). But the rule cannot tell a variable that does not move from an intermediate whose derivative happens to be 0 at the point, or underflows. Where such an intermediate meets an infinite derivative, the true limit of `0·∞` is replaced by 0, a wrong finite answer with no signal where the composite is smooth:
  - `derivative(|x| (x * x * x).cbrt(), 0.0)` gives 0, not 1, and so do `x.powi(3).powf(1.0 / 3.0)` and `cbrt(sin³x)` at 0, and `cbrt(x³)` at `10⁻³⁰⁰`, where `x³` underflows;
  - `second_derivative` of `cbrt(x⁶)` at 0 gives 0, not 2;
  - the gradient of `cbrt(x³ + y³)` at the origin is `(0, 0)`, not `(1, 1)`.

  In general this happens with odd roots of zeros of odd order, `cbrt(x³·g(x))` with `g(0) ≠ 0`. num-dual gives NaN there: a signal, where this crate gives 0. At a kink (`√(x²)`, `(x⁴)^(1/4)` at 0) the 0 is the choice `abs` makes too.
- **Branches take their own derivative.** The comparisons look at the values, so code that branches differentiates the branch taken:
  - `if x == 0.0 { D::ONE } else { x.sin() / x }` has the derivative 0 at 0 (the constant's), which here is right by chance;
  - `if x == 1.0 { D::constant(2.0) } else { (x * x - 1.0) / (x - 1.0) }` has 0 at 1, where `x + 1` has 1.

  A function that is smooth only through such a branch needs the branch written so that both sides have the derivative.
- **Non-smooth points.** At a kink, a tie or a jump the derivative is the one stated in the table of edges, a choice among the one-sided ones (or 0); `atan2` at the origin is NaN. `pow`'s derivatives in `y` at `x = 0` are taken as 0, the limit of `xʸ·ln x` and `xʸ·ln²x` for `y > 0`, but a choice for the mixed `∂²(xʸ)/∂x∂y = xʸ⁻¹(1 + y ln x)`, whose limit is 0 only for `y > 1` and `−∞` for `0 < y ≤ 1` (at `x = 10⁻³⁰⁰`, −3.4·10¹⁵² for `y = ½`, −690 for `y = 1`).
- **Outside the domain** the value is NaN and the derivative is whatever its formula gives, not NaN as a rule: check the value.
- **Infinite values.** The arithmetic is the floats': where a value is infinite (`1/x` at a constant 0, then multiplied by another constant), its product with a zero tangent part is NaN, unlike in the elementary functions.
- **Nesting and the ends of the range.** The nested types carry the lower derivatives as values of their own, and differentiate the formulas of the derivatives in turn, where `HyperDual` forms its second derivatives directly. This has three consequences:
  - Where a lower derivative overflows, its product with a zero tangent part is NaN: `x⁵` at 10¹⁰ in `f32` (whose value overflows) gives a NaN second derivative by `Dual<Dual<f32>>`, where `HyperDual`, which forms `f″ = 20x³` directly, gives `2·10³¹`. The reference comparisons skip the nested forms where a lower derivative overflows.
  - Where it underflows, its terms are lost: `∂²(xʸ)/∂x∂y` at `(10⁻²⁰⁰, 2)` by `Dual<Dual<_>>` misses the term `xʸ/x` with `xʸ`, 0.1 % here, where `HyperDual` and `hessian` are within 4 ulps. Within a few ulps of subnormal, an intermediate of the third derivative of `powi` by `HyperDual<Dual<_>>` can be subnormal (918 ulps at `x⁻⁵³` in the auditor's sweep).
  - Where the derivative part of an intermediate overflows that a small coefficient would bring back, the nested types overflow inside it: the test that brings a power back into range reads the values alone. `x^(10⁻³⁰⁰)` at `10⁻³⁰⁰` has `f″ = −10³⁰⁰`, which `HyperDual` gives; `Dual<Dual<_>>` differentiates `e·x^(e−1)`, whose derivative part `(e − 1)·x^(e−2) = −10⁶⁰⁰` overflows before `e = 10⁻³⁰⁰` multiplies it, and gives `−∞` (and the third derivatives of the same power by the nested types, for which there is no other form, overflow likewise, at `3.7·10⁻¹⁴⁸`). Pinned by a test.
  - `atan2`'s mixed partial by `Dual<Dual<_>>` differentiates `x/r²` and cancels next to the diagonals (`(y² − x²)/r⁴` from `1/r² − 2x²/r⁴`), as any such formula does; normwise the error is about `ε/r²`. `hessian` (by `HyperDual`) forms it from `(y − x)(y + x)` and keeps its digits.

  Use `HyperDual` or `hessian` for second derivatives.
- **Powers next to the subnormal numbers.** Where the power `xᵉ⁻¹` (of `f′`) or `xᵉ⁻²` (of `f″`) of `powf` or `pow` is subnormal though the derivative is a normal number — within a factor `|e|` or `|e(e − 1)|` of the smallest normal number — the derivative has only the subnormal's digits, by every form, `Dual` and `HyperDual` included (`f32`'s `(x^16777215)′` at 0.99999404, where `xᵉ` is subnormal, is 2.4·10⁵ ulps off, its 6.35·10⁻³⁷ against 6.24·10⁻³⁷). Where that power, or `xᵉ` for an exponent beyond `2ᵖ − 2`, underflows to 0, the derivative is 0: `f″` of `x^(−2⁶⁰)` at `1 + 3·2⁻⁵²` is 0, not `3.8·10⁻²⁹⁸` (`xᵉ` is `2.9·10⁻³³⁴`), and `f32`'s `f″` of `x^(2³⁰)` at 0.99999988 is 0, not `3.0·10⁻³⁸`; both are pinned by a test. `powi` takes its powers in halves there instead (see the table of edges).
- **Ill-conditioned points.** Next to a zero of a derivative, its relative error is unbounded for any formula from rounded inputs: `pow`'s mixed partial `xʸ⁻¹·(1 + y ln x)` where `y ln x ≈ −1` (250 ulps at `(5.2·10⁻³, 0.19)` in the auditor's sweep), and third derivatives next to their zeros (`tanh‴` at `tanh x = 1/√3`, 121 `f32` ulps).
- **What nesting costs.** `Dual<Dual<T>>` carries four parts and computes the first derivative twice (in `re.eps` and `eps.re`); `Dual<HyperDual<T>>` and `HyperDual<Dual<T>>` carry eight, and a product takes 27 products of floats. Higher derivatives by nesting grow as 3ᵏ products a product for `k` levels, where Taylor arithmetic would grow as `k²`; there are no dedicated types for third and higher derivatives.
- **Forward mode only.** A gradient takes `n` passes and a Hessian `n(n + 1)/2`, each a full evaluation of the function. For many variables reverse mode (one backward pass for a gradient) would be cheaper; it is not offered, nor are vector dual numbers that carry all `n` directions in one pass.
- **Floats on the left in generic code.** In a function generic over `D`, `2.0 * x` does not compile (see *Writing a function once*): write `x * 2.0` or `D::constant(2.0) * x`.
- **`==` compares values.** `assert_eq!` on two dual numbers passes when their values are equal whatever their derivatives; compare the parts.
- **The values are the standard library's** (but `atanh` below 0), and so inherit their accuracy: `powi` of a large power multiplies its errors up (10 `f32` ulps at `x³¹`, 7 `f64` ulps at `x⁻⁴¹`, up to 40 at `|n| ≤ 60` in the auditor's sweep), and so do its derivatives.

## Measured

Measured on an Apple M4 Max (macOS 27.2) on 2026-11-26 between 07:55 and 07:57, with nothing else running: the areas of the release's final run were timed one after another, and every timed process ran alone on the machine. One thread; `a` as a user builds it, `cargo build --release` with the default profile (opt-level 3, no LTO, the default codegen units, no `target-cpu`), on Rust 1.100.0 stable, and the Rust references the same way, by the harness in the repository's [`bench/numtypes`](../../../bench/numtypes) (see [`bench/README.md`](../../../bench/README.md)). Each figure is the median time an operation of 21 samples of at least 1.5 ms, each a pass over a pool of fresh random inputs (64 bytes to 96 KiB), all the contenders of a row alternating sample by sample in one process; every reference's results were compared with `a`'s on the whole pool before the timing, to `10⁻¹⁰`. Each row gives the fastest reference that does the same work, its time and `a`'s time over it: below 1, `a` is faster. Times below a nanosecond are a few cycles, and their spreads are wide. The reference: num-dual 0.16.0 (`Dual64`, `HyperDual64`, `Dual2_64` and `Dual3_64`, and for the gradients, Jacobians and Hessians one pass of its vector dual numbers, `DualSVec64<n>` and `Dual2SVec64<n>` of static length, and `DualDVec64` on the heap), in `f64`.

### Elementary functions

Each function of `Dual<f64>` and `HyperDual<f64>` against num-dual's `Dual64` and `HyperDual64`, on arguments in the ranges given, every tangent part non-zero:

| Function | Arguments | `Dual` | num-dual | `a`/num-dual | `HyperDual` | num-dual | `a`/num-dual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `add` | `[−2, 2)` | 0.981 ns | 1.63 ns | 0.60 | 1.55 ns | 2.15 ns | 0.72 |
| `mul` | `[−2, 2)` | 1.15 ns | 1.22 ns | 0.94 | 1.54 ns | 2.40 ns | 0.64 |
| `div` | `[0.5, 2)` | 1.11 ns | 1.33 ns | 0.83 | 1.77 ns | 3.30 ns | 0.54 |
| `mul_add` | `[−2, 2)` | 1.15 ns | 1.19 ns | 0.97 | 1.54 ns | 2.37 ns | 0.65 |
| `recip` | `[0.5, 3)` | 1.06 ns | 1.20 ns | 0.88 | 1.65 ns | 2.54 ns | 0.65 |
| `sqrt` | `[0.1, 4)` | 1.21 ns | 1.21 ns | 1.00 | 1.70 ns | 2.80 ns | 0.61 |
| `cbrt` | `[0.1, 4)` | 1.96 ns | 2.54 ns | 0.77 | 2.45 ns | 6.50 ns | 0.38 |
| `powi(3)` | `[−2, 2)` | 1.13 ns | 1.17 ns | 0.96 | 1.67 ns | 2.49 ns | 0.67 |
| `powi(-5)` | `[0.5, 2)` | 1.36 ns | 1.24 ns | 1.09 | 2.85 ns | 2.87 ns | 0.99 |
| `powf(2.5)` | `[0.1, 3)` | 10.2 ns | 6.05 ns | 1.69 | 11.2 ns | 7.90 ns | 1.42 |
| `pow (variable exponent)` | `[0.5, 2)` | 12.8 ns | 4.68 ns | 2.74 | 22.7 ns | 7.43 ns | 3.05 |
| `exp` | `[−3, 3)` | 2.24 ns | 2.76 ns | 0.81 | 2.52 ns | 4.47 ns | 0.56 |
| `exp2` | `[−3, 3)` | 1.75 ns | 2.25 ns | 0.78 | 2.14 ns | 4.47 ns | 0.48 |
| `exp_m1` | `[−3, 3)` | 4.01 ns | 5.00 ns | 0.80 | 4.21 ns | 7.19 ns | 0.58 |
| `ln` | `[0.1, 5)` | 2.37 ns | 2.89 ns | 0.82 | 2.62 ns | 4.44 ns | 0.59 |
| `log2` | `[0.1, 5)` | 2.32 ns | 2.98 ns | 0.78 | 2.61 ns | 4.56 ns | 0.57 |
| `log10` | `[0.1, 5)` | 2.30 ns | 3.12 ns | 0.74 | 2.81 ns | 4.54 ns | 0.62 |
| `ln_1p` | `[−0.5, 3)` | 2.80 ns | 3.91 ns | 0.72 | 3.06 ns | 6.70 ns | 0.46 |
| `log(3)` | `[0.1, 5)` | 2.35 ns | 3.14 ns | 0.75 | 2.60 ns | 4.63 ns | 0.56 |
| `sin` | `[−3, 3)` | 3.62 ns | 4.23 ns | 0.85 | 3.86 ns | 5.43 ns | 0.71 |
| `cos` | `[−3, 3)` | 3.56 ns | 4.33 ns | 0.82 | 3.81 ns | 5.55 ns | 0.69 |
| `tan` | `[−1.4, 1.4)` | 2.62 ns | 5.99 ns | 0.44 | 3.25 ns | 7.67 ns | 0.42 |
| `sin_cos` | `[−3, 3)` | 3.73 ns | 4.44 ns | 0.84 | 4.02 ns | 6.04 ns | 0.67 |
| `asin` | `[−0.9, 0.9)` | 2.97 ns | 3.44 ns | 0.86 | 3.68 ns | 6.30 ns | 0.58 |
| `acos` | `[−0.9, 0.9)` | 2.94 ns | 3.46 ns | 0.85 | 3.33 ns | 6.04 ns | 0.55 |
| `atan` | `[−3, 3)` | 3.74 ns | 4.08 ns | 0.92 | 4.26 ns | 5.96 ns | 0.72 |
| `atan2` | `[−2, 2)` | 6.50 ns | 9.13 ns | 0.71 | 9.06 ns | 10.3 ns | 0.88 |
| `sinh` | `[−3, 3)` | 5.69 ns | 6.39 ns | 0.89 | 6.02 ns | 8.69 ns | 0.69 |
| `cosh` | `[−3, 3)` | 5.52 ns | 6.22 ns | 0.89 | 5.92 ns | 8.69 ns | 0.68 |
| `tanh` | `[−3, 3)` | 5.33 ns | 11.9 ns | 0.45 | 6.00 ns | 14.7 ns | 0.41 |
| `asinh` | `[−3, 3)` | 5.92 ns | 4.92 ns | 1.20 | 5.73 ns | 8.94 ns | 0.64 |
| `acosh` | `[1.1, 5)` | 3.64 ns | 4.06 ns | 0.90 | 4.21 ns | 7.75 ns | 0.54 |
| `atanh` | `[−0.9, 0.9)` | 3.88 ns | 4.95 ns | 0.78 | 4.41 ns | 8.20 ns | 0.54 |
| `abs` | `[−2, 2)` | 0.924 ns | 1.23 ns | 0.75 | 1.40 ns | 2.49 ns | 0.56 |
| `signum` | `[−2, 2)` | 1.15 ns | 1.20 ns | 0.96 | 1.57 ns | 2.23 ns | 0.71 |

`hypot`, `min` and `max` have no reference.

### Derivatives

The free functions against num-dual's: `derivative` against `first_derivative`, `second_derivative` (one `HyperDual` pass) against `second_derivative` (`Dual2_64`, three parts), the third derivative by `Dual<HyperDual<f64>>` against `third_derivative` (`Dual3_64`), all of `sin(eˣ)/(1 + x²)`; `gradient` (`n` `Dual` passes), `jacobian` (`n` passes, all `m` rows at once) and `hessian` (`n(n + 1)/2` `HyperDual` passes) against one pass of num-dual's vector dual numbers, the faster of the static and the heap ones, of Rosenbrock's function, of `Σ sin(xᵢ)·exp(xᵢ₊₁) + ln(1 + xᵢ²)` and of Broyden's tridiagonal map, at `n` variables:

| Function | n | `a` | Fastest | Its time | `a`/fastest |
| --- | --- | ---: | --- | ---: | ---: |
| `derivative` |  | 6.12 ns | num-dual | 6.08 ns | 1.01 |
| `second_derivative` |  | 10.9 ns | num-dual | 7.64 ns | 1.42 |
| third derivative, nested |  | 19.1 ns | num-dual | 12.5 ns | 1.53 |
| `gradient`, Rosenbrock | 2 | 15.8 ns | num-dual | 3.62 ns | 4.37 |
|  | 4 | 25.3 ns | num-dual | 14.7 ns | 1.71 |
|  | 8 | 75.2 ns | num-dual | 208 ns | 0.36 |
|  | 32 | 1.20 µs | num-dual | 4.75 µs | 0.25 |
| `gradient`, `sin·exp + ln` | 2 | 36.0 ns | num-dual | 11.6 ns | 3.12 |
|  | 4 | 115 ns | num-dual | 31.2 ns | 3.68 |
|  | 8 | 440 ns | num-dual | 153 ns | 2.89 |
|  | 32 | 7.24 µs | num-dual | 3.98 µs | 1.82 |
| `hessian`, Rosenbrock | 2 | 33.8 ns | num-dual | 16.9 ns | 2.00 |
|  | 4 | 94.1 ns | num-dual | 366 ns | 0.26 |
|  | 8 | 642 ns | num-dual | 2.34 µs | 0.28 |
|  | 32 | 39.4 µs | num-dual | 259 µs | 0.15 |
| `hessian`, `sin·exp + ln` | 2 | 64.5 ns | num-dual | 21.5 ns | 3.00 |
|  | 4 | 353 ns | num-dual | 350 ns | 1.01 |
|  | 8 | 2.71 µs | num-dual | 2.09 µs | 1.30 |
|  | 32 | 172 µs | num-dual | 263 µs | 0.65 |
| `value_gradient_and_hessian`, Rosenbrock | 2 | 35.0 ns | num-dual | 17.0 ns | 2.06 |
|  | 4 | 97.4 ns | num-dual | 366 ns | 0.27 |
|  | 8 | 641 ns | num-dual | 2.32 µs | 0.28 |
|  | 32 | 39.5 µs | num-dual | 260 µs | 0.15 |
| `jacobian`, Broyden | 2 | 50.9 ns | num-dual | 16.2 ns | 3.14 |
|  | 4 | 93.7 ns | num-dual | 35.2 ns | 2.66 |
|  | 8 | 210 ns | num-dual | 218 ns | 0.96 |
|  | 32 | 1.63 µs | num-dual | 4.25 µs | 0.38 |

- 64 of the 70 elementary functions are at par or faster (0.38 to 1.00), the `HyperDual` ones at 0.38 to 0.99 of num-dual's time.
- The Hessians of Rosenbrock's function take 0.15 to 0.28 of num-dual's time from `n = 4`, its gradients 0.20 to 0.36 from 8, and the Jacobians 0.38 to 0.42 from 16: num-dual's static vectors carry `n` tangents in every operation, which grows costly past a few.

### Where the references are faster

Every case above 1.05 times num-dual's time, by cause: 23 of the 108 cases.

- **Gradients, Jacobians and Hessians of few variables, and of transcendental functions** (at `n = 2` 2.00 to 4.37, at `n = 4` 1.71 to 3.68, the gradient of `sin·exp + ln` at every `n` to 32 1.76 to 3.68, and its Hessian at `n = 8` 1.30): `n` passes, each evaluating every elementary function again, against num-dual's one pass with `n` tangents; at small `n` also the `Matrix` results and each pass's set-up. Vector dual numbers in one pass, which would close it, are not offered (a decision of #25); the per-pass costs are what is left to reduce.
- **`second_derivative` and the third derivative** (1.42 to 1.53 against `Dual2_64` and `Dual3_64`): `HyperDual` carries four parts where `Dual2` carries the three a second derivative needs, and the nested `Dual<HyperDual<f64>>` eight where `Dual3` carries four.
- **Powers and `asinh`** (`powf(2.5)` 1.42 to 1.69, `pow` of a variable exponent 2.74 to 3.05, `Dual` `powi(-5)` 1.09 and `asinh` 1.20): `powf` corrects the exponent's remainder (`e − 1` rounded and its exact remainder, `ln x` and the edge tests) on every call, and `pow` takes `ln x` and both partial derivatives, for the accuracy and the edges of the table above; `asinh` keeps its digits for large and tiny arguments.

