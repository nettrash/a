//! Mathematics.
//!
//! * [`ec`]: elliptic curves over prime fields in Weierstrass, Montgomery and Edwards form, whose
//!   points add with `+`, subtract with `-` and multiply by a number with `*`, and the curves in
//!   common use; on numbers, and not constant-time.
//! * [`numbers`]: integers of any size, stored as a sign and the bytes of the absolute value
//!   in either byte order; they add with `+` and `+=`, subtract with `-` and `-=`, multiply
//!   with `*` and `*=`, divide with `/`, `/=`, `%` and `%=`, shift with `<<`, `>>`, `<<=` and
//!   `>>=`, and are written and read as text in any radix from 2 to 36.

pub mod ec;
pub mod numbers;
