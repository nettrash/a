//! Mathematics.
//!
//! * [`numbers`]: integers of any size, stored as a sign and the bytes of the absolute value
//!   in either byte order; they add with `+` and `+=`, subtract with `-` and `-=`, multiply
//!   with `*` and `*=`, divide with `/`, `/=`, `%` and `%=`, shift with `<<`, `>>`, `<<=` and
//!   `>>=`, and are written and read as text in any radix from 2 to 36.

pub mod numbers;
