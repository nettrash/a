# Changelog

All notable changes to [`nettrash-a`](https://crates.io/crates/nettrash-a) are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `a::math::numerics`: the standard problems of numerical analysis on `f32` and `f64`, generic over the sealed trait `Real`, its linear algebra on `a::math::matrix`. A method returns `None`, or its best point with `converged` false, rather than an answer it has not verified; each comes with the default options and a `_with` form taking an options struct.
  - The zeros of a function of one variable by bisection, Brent, the secant method, Newton and Halley, and of a system by Newton with a Jacobian and by Broyden's rank-one update.
  - Definite integrals by adaptive Gauss–Kronrod (G7–K15 and G10–K21), Clenshaw–Curtis and tanh-sinh quadrature, over infinite intervals too, in several dimensions by cubature and by Sobol' Monte Carlo, and the nodes and weights of the Gauss–Legendre, Hermite, Laguerre and Jacobi rules by Golub and Welsch's method.
  - The initial value problem `y′ = f(t, y)` by the classical fourth-order Runge–Kutta method, by Dormand and Prince's 5(4) pair and by the L-stable three-stage Radau IIA method, whose collocation system Newton's method solves on the LU decomposition, all with step doubling and the usual step-size controller.
  - Minima by the golden section and by Brent's method in one variable, by Nelder and Mead's simplex, the Polak–Ribière conjugate gradient and BFGS in several, the last two on a strong-Wolfe line search, and sums of squares by Gauss–Newton and by Levenberg and Marquardt.
  - Interpolation by the cubic, clamped, not-a-knot, Akima and PCHIP splines, by Chebyshev approximation evaluated with Clenshaw's recurrence, and by barycentric interpolation; differentiation by Richardson extrapolation on symmetric differences.

## [0.1.29](https://crates.io/crates/nettrash-a/0.1.29) - 2026-11-04

The first release. Algorithms in pure Rust with no dependencies, each with a page of its own that gives its method, examples and measurements against the reference implementations.

### Added

- `a::sort`: 23 sorting algorithms and a default sort, each taking a mutable slice.
  - By comparison, for any `T: Ord`: bubble, selection, insertion, cocktail shaker, gnome, comb, merge, quick, heap, Tim, intro, Shell, Drift, tree, cycle, bitonic, odd-even and stooge sorts. Bitonic and odd-even sorts also come in parallel forms.
  - By integer or float keys, with `_by_key` forms: counting, radix, bucket and pigeonhole sorts, and sleep sort as a joke.
  - `sort`, the default: Drift sort, chosen over the other general-purpose sorts by measurement.
- `a::math::numbers::Number`: an integer of any size, stored as its sign and the bytes of its magnitude, in either byte order.
  - `+`, `-`, `*`, `/` and `%` with their assignments, with owned or borrowed operands and with primitive integers.
  - Products by the schoolbook method, Karatsuba's, Toom-3, Toom-4 and number-theoretic transforms.
  - Quotients by Knuth's method with Möller and Granlund's reciprocals, by divide and conquer and by Newton's method; `div_rem`, Euclid's division, a prepared `Divisor`, and exact division.
  - Shifts, bitwise operations in two's complement, and bit queries.
  - Text in any radix from 2 to 36.
  - Roots, factorials and binomials, greatest common divisors and modular inverses, powers and modular powers.
  - Primality by Baillie–PSW, the next prime, the Jacobi symbol, and square roots modulo a prime.
  - Comparison, hashing, conversions from and to the primitive integers, to floats and from strings, and `Sum` and `Product`.
  - Assembly kernels on aarch64 and x86_64, and NEON transforms on aarch64.
- `a::math::ec`: elliptic curves over prime fields of any length, on `Number`.
  - Short Weierstrass, Montgomery and twisted Edwards forms.
  - Points that add with `+`, subtract with `-` and multiply by a number with `*`.
  - Compression, `multi_mul` by Straus's and Pippenger's methods, and prepared points for repeated products.
  - Invariants and maps between the forms.
  - secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519 ready made, with field kernels of their own.
  - `k·P` and `k·G` are faster than OpenSSL 3.6 on its curves, and than curve25519-dalek 4.1 on Ed25519 and X25519.
- `a::math::complex`: `Complex<f32>` and `Complex<f64>`.
  - Their arithmetic, with a float of the same type on either side.
  - `abs`, `arg`, `exp`, `ln`, `sqrt` and powers, as principal values with C99's branch cuts.
  - Division and square roots that neither overflow nor underflow before their results do.
- `a::math::matrix`: `Matrix<T>` of `f32`, `f64`, `Complex<f32>`, `Complex<f64>` or `Number`.
  - Arithmetic: `+`, `-` and `*`, by an element, and Hadamard's product. Transposes and conjugate transposes, row and column operations, the trace, and powers.
  - LU, QR, Cholesky, singular value, Schur and eigendecompositions.
  - Determinants, inverses, solutions of linear systems and least squares, ranks, null spaces, characteristic polynomials, six norms and condition numbers.
  - `exp`, `sqrt` and `log`.
  - On `Number`: exact determinants, ranks, characteristic polynomials, adjugates, inverses and null spaces.
  - In `f64` on one core, at least as fast as faer and nalgebra at every size measured, from 8 × 8 to 1024 × 1024.

### Security

- Nothing in the crate is constant-time: `pow_mod` and the elliptic curves in particular branch on the values of their exponents and scalars. Use them on public data, not on secret keys.
