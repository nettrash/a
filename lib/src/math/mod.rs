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
//! * [`matrix`]: matrices of floats, complex numbers, numbers or rationals, their arithmetic with
//!   `+`, `-` and `*`, transposes, row and column operations, determinants, decompositions,
//!   eigenvalues and functions of matrices, and the exact determinants, solutions and inverses of
//!   matrices of numbers and rationals.
//! * [`numbers`]: integers of any size, stored as a sign and the bytes of the absolute value
//!   in either byte order; they add with `+` and `+=`, subtract with `-` and `-=`, multiply
//!   with `*` and `*=`, divide with `/`, `/=`, `%` and `%=`, shift with `<<`, `>>`, `<<=` and
//!   `>>=`, and are written and read as text in any radix from 2 to 36.
//! * [`poly`]: polynomials in one variable with coefficients that are numbers, rationals, floats,
//!   complex numbers or residues modulo an integer, their arithmetic, division and powers modulo a
//!   polynomial, greatest common divisors and resultants, evaluation at many points and
//!   interpolation, roots in double precision, certified or correctly rounded, and factorisation.
//! * [`rational`]: rational numbers in lowest terms over [`numbers`], their arithmetic with `+`,
//!   `-`, `*` and `/`, with numbers and primitive integers too, comparisons, rounding, and their
//!   conversions from and to integers, text and floats; continued fractions, best approximations
//!   and rational reconstruction; and the Bernoulli and harmonic numbers.
//! * [`special`]: special functions of floats, and of complex numbers for some: the gamma family
//!   with the incomplete gamma and beta functions and their inverses, the error functions,
//!   Bessel and Airy functions, the Riemann and Hurwitz zeta functions and polylogarithms, the
//!   Lambert W function, exponential and elliptic integrals and the orthogonal polynomials.

pub mod complex;
pub mod ec;
pub mod fft;
pub mod float;
pub mod matrix;
pub mod numbers;
pub mod poly;
pub mod rational;
pub mod special;
