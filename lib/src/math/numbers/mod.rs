//! Integers of any size: a [`Number`] holds the sign of an integer and the bytes of its
//! absolute value, in the [`ByteOrder`] it was created with.
//!
//! Numbers add with `+` and `+=`, subtract with `-` and `-=` and multiply with `*` and `*=`,
//! each operand owned or borrowed, whatever their signs and byte orders: the result is in the
//! shortest form and in the byte order of the left operand, and it is built in the vector of an
//! operand taken by value, or of the target of `+=`, `-=` or `*=`, whenever that vector has room
//! for it (the left operand's first).
//!
//! They divide with `/` and `/=`, truncated toward zero, and take the remainder with `%` and `%=`,
//! which has the sign of the dividend, as Rust's signed integers do, and dividing by zero panics;
//! [`Number::div_rem`] gives both, and [`Number::div_euclid`] and [`Number::rem_euclid`] Euclid's
//! division, whose remainder is never negative. A [`Divisor`] prepares a number to divide many
//! others by, [`Number::div_exact`] divides by a number known to divide exactly, and
//! [`Number::is_multiple_of`] tells whether it does.
//!
//! They shift by a `usize` number of bits with `<<`, `>>`, `<<=` and `>>=`, owned or borrowed, as
//! Rust's signed integers do but with no bits lost at the top: `a << k` is `a · 2^k`, and `a >> k`
//! is `a / 2^k` rounded towards negative infinity. The result keeps the byte order of the number
//! shifted, and is built in its vector, when it is owned, whenever that has room for it.
//!
//! They are written as text in decimal by `Display`, in hexadecimal, octal and binary by the other
//! formatting traits, and in any radix from 2 to 36 by [`Number::to_str_radix`], and read by
//! [`Number::from_str_radix`]: the sign and the digits of the magnitude.

mod add;
mod byte_order;
mod div;
mod divisor;
mod mul;
mod number;
mod radix;
mod shift;
mod sub;

pub use byte_order::ByteOrder;
pub use divisor::Divisor;
pub use number::Number;
pub use radix::{NumberErrorKind, ParseNumberError};
