//! Mathematics.
//!
//! * [`complex`]: complex numbers with `f32` or `f64` parts, their arithmetic with `+`, `-`, `*`
//!   and `/`, and their absolute values, arguments, exponentials, logarithms, square roots and
//!   powers.
//! * [`ec`]: elliptic curves over prime fields in Weierstrass, Montgomery and Edwards form, whose
//!   points add with `+`, subtract with `-` and multiply by a number with `*`, and the curves in
//!   common use; on numbers, and not constant-time.
//! * [`fft`]: fast Fourier transforms of real and complex sequences of any length, in one, two
//!   and three dimensions and in batches; cosine and sine transforms of types I to IV; and linear
//!   and circular convolutions and correlations.
//! * [`float`]: binary floats of any precision, each result correctly rounded in four modes as
//!   MPFR rounds it: their arithmetic, roots and powers, conversions and text, exponentials,
//!   logarithms, hyperbolic and circular functions and constants; and complex numbers of two
//!   such parts, as GNU MPC's, with their elementary functions.
//! * [`lattice`]: the reduction of lattice bases by LLL, exactly or by the L² algorithm, by LLL
//!   with deep insertions and by BKZ; shortest and closest vectors by enumeration; and integer
//!   relations among real numbers by PSLQ.
//! * [`matrix`]: matrices of floats, complex numbers, numbers or rationals, their arithmetic with
//!   `+`, `-` and `*`, transposes, row and column operations, determinants, decompositions,
//!   eigenvalues and functions of matrices, the exact determinants, solutions and inverses of
//!   matrices of numbers and rationals, the Hermite and Smith normal forms, and matrices over the
//!   prime fields `GF(p)` and over `GF(2)`.
//! * [`numbers`]: integers of any size, stored as a sign and the bytes of the absolute value
//!   in either byte order; they add with `+` and `+=`, subtract with `-` and `-=`, multiply
//!   with `*` and `*=`, divide with `/`, `/=`, `%` and `%=`, shift with `<<`, `>>`, `<<=` and
//!   `>>=`, and are written and read as text in any radix from 2 to 36.
//! * [`numerics`]: numerical methods on `f32` and `f64`: the zeros of functions and of systems of
//!   equations, integrals in one and several dimensions and the Gaussian rules, initial value
//!   problems for stiff and non-stiff ordinary differential equations with dense output and
//!   events, minima with and without bounds, least squares and linear programming, splines,
//!   Chebyshev and rational approximation, and derivatives.
//! * [`poly`]: polynomials in one variable with coefficients that are numbers, rationals, floats,
//!   complex numbers or residues modulo an integer, their arithmetic, division and powers modulo a
//!   polynomial, greatest common divisors and resultants, evaluation at many points and
//!   interpolation, roots in double precision, certified or correctly rounded, and factorisation.
//! * [`rational`]: rational numbers in lowest terms over [`numbers`], their arithmetic with `+`,
//!   `-`, `*` and `/`, with numbers and primitive integers too, comparisons, rounding, and their
//!   conversions from and to integers, text and floats; continued fractions, best approximations
//!   and rational reconstruction; and the Bernoulli and harmonic numbers.
//! * [`sparse`]: sparse matrices of floats and complex numbers in compressed row, compressed
//!   column and coordinate form, read and written as Matrix Market files, their products and
//!   other operations, the orderings that keep the fill of a factorisation low, the sparse
//!   Cholesky, `L·D·Lᴴ`, LU and QR factorisations, the Krylov methods with their preconditioners,
//!   and a few eigenvalues by the Lanczos and Arnoldi methods.
//! * [`special`]: special functions of floats, and of complex numbers for some: the gamma family
//!   with the incomplete gamma and beta functions and their inverses, the error functions,
//!   Bessel and Airy functions, the Riemann and Hurwitz zeta functions and polylogarithms, the
//!   Lambert W function, exponential and elliptic integrals and the orthogonal polynomials.

pub mod complex;
pub mod ec;
pub mod fft;
pub mod float;
pub mod lattice;
pub mod matrix;
pub mod numbers;
pub mod numerics;
pub mod poly;
pub mod rational;
pub mod sparse;
pub mod special;
