//! Mathematics.
//!
//! * [`complex`]: complex numbers with `f32` or `f64` parts, their arithmetic with `+`, `-`, `*`
//!   and `/`, and their absolute values, arguments, exponentials, logarithms, square roots and
//!   powers.
//! * [`ec`]: elliptic curves over prime fields in Weierstrass, Montgomery and Edwards form, whose
//!   points add with `+`, subtract with `-` and multiply by a number with `*`, and the curves in
//!   common use; on numbers, and not constant-time.
//! * [`matrix`]: matrices of floats, complex numbers, numbers or rationals, their arithmetic with
//!   `+`, `-` and `*`, transposes, row and column operations, determinants, decompositions,
//!   eigenvalues and functions of matrices, and the exact determinants, solutions and inverses of
//!   matrices of numbers and rationals.
//! * [`numbers`]: integers of any size, stored as a sign and the bytes of the absolute value
//!   in either byte order; they add with `+` and `+=`, subtract with `-` and `-=`, multiply
//!   with `*` and `*=`, divide with `/`, `/=`, `%` and `%=`, shift with `<<`, `>>`, `<<=` and
//!   `>>=`, and are written and read as text in any radix from 2 to 36.
//! * [`rational`]: rational numbers in lowest terms over [`numbers`], their arithmetic with `+`,
//!   `-`, `*` and `/`, with numbers and primitive integers too, comparisons, rounding, and their
//!   conversions from and to integers, text and floats; continued fractions, best approximations
//!   and rational reconstruction; and the Bernoulli and harmonic numbers.

pub mod complex;
pub mod ec;
pub mod float;
pub mod matrix;
pub mod numbers;
pub mod rational;
